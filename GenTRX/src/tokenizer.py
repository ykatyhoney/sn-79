# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Order tokenizer — per-field binning, no composite tokens.

Each order field is binned independently. No flat vocabulary.
Fields: order_type, price, vol_int, vol_dec, interval.
Conditioning (not tokenized): lob_volumes, time_of_day, mid_delta.

Price binning uses symmetric log scale: bins are split into negative and positive
halves, with log-spaced edges within each half. This gives high resolution near
mid-price (where most orders cluster) and coarser bins deeper in the book.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class BinConfig:
    """Defines bin edges for a single field."""

    n_bins: int
    lo: float
    hi: float
    log_scale: bool = False  # use log-spaced bins (for heavy-tailed data)
    symmetric_log: bool = False  # signed log: split into neg/pos halves

    def edges(self) -> np.ndarray:
        if self.symmetric_log:
            return _symmetric_log_edges(self.n_bins, self.hi)
        if self.log_scale:
            # Log-spaced bins: lo must be > 0, we add a zero-bin at the front
            lo = max(self.lo, 1.0)
            return np.logspace(np.log10(lo), np.log10(self.hi), self.n_bins)[1:]
        return np.linspace(self.lo, self.hi, self.n_bins + 1)[1:-1]

    def digitize(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(values, self.edges()), 0, self.n_bins - 1)

    def center(self, bin_idx: int) -> float:
        """Return a representative natural value for a given bin id.

        Inverse of digitize for inference reconstruction. Respects
        log_scale / symmetric_log so the reconstructed value matches
        the magnitude the model trained on (a linear midpoint mis-
        reads log bins by orders of magnitude).
        """
        import math
        idx = max(0, min(self.n_bins - 1, int(bin_idx)))
        if self.symmetric_log:
            half = self.n_bins // 2
            log_hi = math.log10(max(1.0, self.hi))
            if idx < half:
                t = (idx + 0.5) / half
                return -(10.0 ** (log_hi * (1.0 - t)))
            t = (idx - half + 0.5) / half
            return 10.0 ** (log_hi * t)
        if self.log_scale:
            lo = max(self.lo, 1.0)
            log_lo = math.log10(lo)
            log_hi = math.log10(max(self.hi, lo + 1.0))
            t = (idx + 0.5) / self.n_bins
            return 10.0 ** (log_lo + t * (log_hi - log_lo))
        return self.lo + (idx + 0.5) / self.n_bins * (self.hi - self.lo)


def _symmetric_log_edges(n_bins: int, hi: float) -> np.ndarray:
    """Symmetric log bin edges for signed values in [-hi, +hi].

    Layout (100 bins example):
        bins  0..48  → negative side: [-hi .. -1] log-spaced (coarse→fine)
        bin   49     → zero band: [-1 .. +1]
        bins 50..99  → positive side: [+1 .. +hi] log-spaced (fine→coarse)

    Half the bins cover each side. Within each side, edges are log-spaced
    from 1 to hi, giving dense coverage near mid (±1..±20 ticks) and
    progressively coarser bins toward the tail.
    """
    half = n_bins // 2  # 50 for 100 bins
    # Log-spaced edges from 1 to hi on the positive side
    # half bins need (half - 1) internal edges on the positive side,
    # plus the boundary at +1 (shared with zero band)
    pos_edges = np.logspace(0, np.log10(hi), half)  # [1, ..., hi], length=half
    # Negative side is the mirror
    neg_edges = -pos_edges[::-1]  # [-hi, ..., -1]
    # Combine: neg_edges define the left boundaries, pos_edges the right
    # Full edge set: [-hi, ..., -1, +1, ..., +hi]
    # np.digitize uses these as thresholds; values in [-1, +1) land in the zero band
    edges = np.concatenate([neg_edges, pos_edges])
    return edges


@dataclass
class TokenizerConfig:
    n_types: int = 5
    price: BinConfig = field(
        default_factory=lambda: BinConfig(100, -500, 500, symmetric_log=True)
    )
    vol_int: BinConfig = field(
        default_factory=lambda: BinConfig(64, 0, 100, log_scale=True)
    )
    vol_dec: BinConfig = field(default_factory=lambda: BinConfig(8, 0.0, 1.0))
    interval: BinConfig = field(
        default_factory=lambda: BinConfig(64, 0, 50_000_000, log_scale=True)
    )
    lob_depth: int = 10

    # Conditioning embeddings
    max_mid_delta: int = 2000
    time_bin_seconds: int = 5
    # Multi-scale regime context: rolling realized-vol / order-flow imbalance /
    # momentum over this many events (0 = disabled). Derived from mid_price +
    # order_type, fed as model conditioning (gives momentum/mean-reversion memory).
    regime_window: int = 0

    # Cross-asset normalization (asset_norm). When normalize=True and asset_ref is set,
    # encode bins NORMALIZED values on the SAME 5 heads (no model change): price head
    # bins p_norm (bps-of-mid / price_scale_bps); vol_int/vol_dec bin int/dec of
    # qty/median_qty (ratio-to-median ~O(1), same range across assets). Set price /
    # vol_int bin RANGES to the shared normalized ranges when using this. Decode
    # (inference) inverts via asset_norm + the row's mid. int+dec stays parquet storage.
    normalize: bool = False
    asset_ref: dict | None = None  # {median_qty, price_scale_bps, pair_decimals}

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizerConfig":
        """Reconstruct from a dict (e.g., from checkpoint via asdict)."""
        kw = dict(d)
        for key in ("price", "vol_int", "vol_dec", "interval"):
            if key in kw and isinstance(kw[key], dict):
                kw[key] = BinConfig(**kw[key])
        return cls(**kw)

    @property
    def lob_dim(self) -> int:
        return self.lob_depth * 2

    @property
    def mid_delta_buckets(self) -> int:
        return self.max_mid_delta * 2 + 1

    @property
    def time_of_day_buckets(self) -> int:
        return 86400 // self.time_bin_seconds + 1

    @property
    def field_sizes(self) -> dict[str, int]:
        """Output head sizes per field."""
        return {
            "order_type": self.n_types,
            "price": self.price.n_bins,
            "vol_int": self.vol_int.n_bins,
            "vol_dec": self.vol_dec.n_bins,
            "interval": self.interval.n_bins,
        }


class OrderTokenizer:
    """Bins order fields independently. No composite tokens."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

    def encode(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Encode a parquet DataFrame into per-field bin arrays + conditioning arrays.

        Thin wrapper over ``encode_columns``: materialises the polars columns
        to numpy first, then delegates. Prefer ``encode_columns`` from hot
        paths to avoid the polars allocator entirely.

        Returns dict with:
            Per-field bins (model input + label targets):
                order_types, price_bins, vol_int_bins, vol_dec_bins, interval_bins
            Conditioning (embedding inputs, not predicted):
                lob_volumes, time_of_day, mid_deltas
        """
        cols: dict[str, np.ndarray] = {
            name: df.get_column(name).to_numpy() for name in df.columns
        }
        return self.encode_columns(cols)

    def encode_columns(
        self, cols: dict[str, np.ndarray], asset_ref: dict | None = None
    ) -> dict[str, np.ndarray]:
        """Encode pre-materialised numpy columns into per-field bin arrays.

        Equivalent output to ``encode(df)`` but reads from a dict of numpy
        arrays — letting callers source the columns via pyarrow (system
        malloc, no caching pool) instead of polars (mimalloc, retains pool).

        ``asset_ref`` overrides the config's reference for this call (per-file
        cross-asset normalization); falls back to ``config.asset_ref``.
        """
        c = self.config
        ref = asset_ref if asset_ref is not None else c.asset_ref

        # Per-field bins
        order_types = cols["order_type"].astype(np.int32)
        rel_price = cols["rel_price"].astype(np.float64)

        # Volume raw — handle both old (single "volume") and new (int+dec) schemas
        if "volume_int" in cols:
            vol_int_raw = cols["volume_int"].astype(np.float64)
            vol_dec_raw = cols["volume_dec"].astype(np.float64)
        elif "volume" in cols:
            # Legacy: single integer-scaled volume. Approximate split.
            vol_raw = cols["volume"].astype(np.float64)
            vol_int_raw = np.floor(vol_raw).astype(np.float64)
            vol_dec_raw = vol_raw - vol_int_raw
        else:
            raise ValueError(
                "Parquet must have 'volume_int'+'volume_dec' or 'volume' column"
            )

        if c.normalize and ref is not None:
            # Cross-asset: bin NORMALIZED values on the same heads (no model change).
            from GenTRX.src.asset_norm import price_to_norm
            mid = cols["mid_price"].astype(np.float64)
            price_bins = c.price.digitize(price_to_norm(rel_price, mid, ref)).astype(np.int32)
            qty_norm = (vol_int_raw + vol_dec_raw) / max(float(ref["median_qty"]), 1e-12)
            vi = np.floor(qty_norm)
            vol_int_bins = c.vol_int.digitize(vi).astype(np.int32)
            vol_dec_bins = c.vol_dec.digitize(qty_norm - vi).astype(np.int32)
        else:
            price_bins = c.price.digitize(rel_price).astype(np.int32)
            vol_int_bins = c.vol_int.digitize(vol_int_raw).astype(np.int32)
            vol_dec_bins = c.vol_dec.digitize(vol_dec_raw).astype(np.int32)
        interval_bins = c.interval.digitize(
            cols["interval_ns"].astype(np.float64)
        ).astype(np.int32)

        # LOB volumes
        lob_cols = []
        for side in ("ask", "bid"):
            for i in range(1, c.lob_depth + 1):
                lob_cols.append(cols[f"lob_{side}_vol_{i}"])
        lob_volumes = np.column_stack(lob_cols).astype(np.float32)

        # Conditioning: time_of_day, mid_delta
        time_of_day = self._compute_time_of_day_cols(cols)
        mid_deltas = self._compute_mid_delta_cols(cols)

        out = {
            "order_types": order_types,
            "price_bins": price_bins,
            "vol_int_bins": vol_int_bins,
            "vol_dec_bins": vol_dec_bins,
            "interval_bins": interval_bins,
            "lob_volumes": lob_volumes,
            "time_of_day": time_of_day,
            "mid_deltas": mid_deltas,
        }
        if c.regime_window > 0:
            out["regime"] = self._compute_regime_cols(cols, c.regime_window)
        return out

    def _compute_regime_cols(self, cols: dict[str, np.ndarray], w: int) -> np.ndarray:
        """Rolling (causal, expanding then window-w) regime features:
        realized vol, order-flow imbalance, momentum — from mid_price + order_type."""
        mid = cols["mid_price"].astype(np.float64)
        ot = cols["order_type"]
        sign = np.where(ot == 0, 1.0, np.where(ot == 1, -1.0, 0.0))
        dmid = np.empty_like(mid)
        dmid[0] = 0.0
        dmid[1:] = np.diff(mid)
        n = len(mid)

        def roll_mean(x: np.ndarray) -> np.ndarray:
            cs = np.concatenate([[0.0], np.cumsum(x)])
            idx = np.arange(n)
            lo = np.maximum(0, idx - w + 1)
            return (cs[idx + 1] - cs[lo]) / (idx - lo + 1)

        vol = np.sqrt(roll_mean(dmid * dmid))
        imb = roll_mean(sign)
        if n > w:
            mom = mid - np.concatenate([np.full(w, mid[0]), mid[:-w]])
        else:
            mom = mid - mid[0]
        return np.column_stack([vol, imb, mom]).astype(np.float32)

    def _compute_time_of_day(self, df: pl.DataFrame) -> np.ndarray:
        c = self.config
        if "time_of_day_s" in df.columns:
            tod_s = df.get_column("time_of_day_s").to_numpy().astype(np.int64)
        else:
            ts = df.get_column("timestamp")
            tod_s = (
                ts.dt.hour().cast(pl.Int64) * 3600
                + ts.dt.minute().cast(pl.Int64) * 60
                + ts.dt.second().cast(pl.Int64)
            ).to_numpy()
        return np.clip(
            tod_s // c.time_bin_seconds, 0, c.time_of_day_buckets - 1
        ).astype(np.int32)

    def _compute_time_of_day_cols(self, cols: dict[str, np.ndarray]) -> np.ndarray:
        c = self.config
        if "time_of_day_s" in cols:
            tod_s = cols["time_of_day_s"].astype(np.int64)
        else:
            # pyarrow's to_numpy() on a timestamp column yields datetime64[ns]
            ts = cols["timestamp"]
            ts_ns = ts.astype("datetime64[ns]").view(np.int64)
            tod_s = (ts_ns % (86400 * 10**9)) // 10**9
        return np.clip(
            tod_s // c.time_bin_seconds, 0, c.time_of_day_buckets - 1
        ).astype(np.int32)

    def _compute_mid_delta(self, df: pl.DataFrame) -> np.ndarray:
        c = self.config
        if "mid_price_delta" in df.columns:
            delta = df.get_column("mid_price_delta").to_numpy().astype(np.int64)
        else:
            mid = df.get_column("mid_price").to_numpy().astype(np.int64)
            valid = mid[mid > 0]
            ref = int(valid[0]) if len(valid) > 0 else 0
            delta = mid - ref
        clipped = np.clip(delta, -c.max_mid_delta, c.max_mid_delta)
        return (clipped + c.max_mid_delta).astype(np.int32)

    def _compute_mid_delta_cols(self, cols: dict[str, np.ndarray]) -> np.ndarray:
        c = self.config
        if "mid_price_delta" in cols:
            delta = cols["mid_price_delta"].astype(np.int64)
        else:
            mid = cols["mid_price"].astype(np.int64)
            valid = mid[mid > 0]
            ref = int(valid[0]) if len(valid) > 0 else 0
            delta = mid - ref
        clipped = np.clip(delta, -c.max_mid_delta, c.max_mid_delta)
        return (clipped + c.max_mid_delta).astype(np.int32)
