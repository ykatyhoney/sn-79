"""End-to-end GenTRX scoring on real code paths (no fake scorer).

Composes the miner one-pass path with the aggregator's held-out scoring:

    train_incremental (miner)  ->  compress  ->  evaluate_gradient (scorer)
                                                 -> _eval_loss / compute_loss
                                                    (whole-row nonfinite mask)

on a real n_types=5 OrderModel with a DISJOINT synthetic train / held-out
split. Unlike test_scoring.py (which stubs _score_and_aggregate), this drives
the actual scoring functions, so it covers the interaction of:
  - the miner one-pass gradient (train_incremental, budget-governed),
  - held-out scoring on data the miner did not train on,
  - the single-pass baseline reuse (loss_before=... == inline baseline),
  - per-miner rollback (the base model the shared baseline depends on is
    left byte-identical after scoring).

Run: pytest GenTRX/tests/test_score_end_to_end.py -v
"""
import copy

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from GenTRX.src.dataloader import ChunkSampler, OrderDataset
from GenTRX.src.distributed import (
    WindowConfig,
    _eval_loss,
    evaluate_gradient,
    train_incremental,
)
from GenTRX.src.gradient import compress
from GenTRX.src.model import ModelConfig, OrderModel
from GenTRX.src.tokenizer import OrderTokenizer
from GenTRX.src.util.schema import LOB_DEPTH, order_stream_schema

SEQ = 16
BATCH = 4


def _write_parquet(path, n=240, *, seed=0):
    """Write one order-stream page. `seed` makes train vs held-out disjoint
    (different realizations of the same schema, like different books)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = {
            "timestamp": 1_000_000_000 * (i + 1),
            # n_types=5: bid / ask / cancel / exec_buy / exec_sell
            "order_type": int(rng.integers(0, 5)),
            "rel_price": int(rng.integers(-50, 50)),
            "volume_int": int(rng.integers(0, 64)),
            "volume_dec": float(rng.random()),
            "interval_ns": int(rng.integers(0, 100_000)),
            "mid_price": 100_000 + int(rng.integers(-50, 50)),
            "time_of_day_s": int(rng.integers(0, 86400)),
            "mid_price_delta": int(rng.integers(-5, 6)),
        }
        for j in range(LOB_DEPTH):
            r[f"lob_ask_vol_{j + 1}"] = float(rng.integers(1, 50))
            r[f"lob_bid_vol_{j + 1}"] = float(rng.integers(1, 40))
        rows.append(r)
    cols = {
        "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.timestamp("ns")),
        "order_type": np.array([r["order_type"] for r in rows], dtype=np.int8),
        "rel_price": np.array([r["rel_price"] for r in rows], dtype=np.int32),
        "volume_int": np.array([r["volume_int"] for r in rows], dtype=np.int32),
        "volume_dec": np.array([r["volume_dec"] for r in rows], dtype=np.float32),
        "interval_ns": np.array([r["interval_ns"] for r in rows], dtype=np.int64),
        "mid_price": np.array([r["mid_price"] for r in rows], dtype=np.int64),
        "time_of_day_s": np.array([r["time_of_day_s"] for r in rows], dtype=np.int32),
        "mid_price_delta": np.array([r["mid_price_delta"] for r in rows], dtype=np.int64),
    }
    for k in range(LOB_DEPTH):
        cols[f"lob_ask_vol_{k + 1}"] = np.array(
            [r[f"lob_ask_vol_{k + 1}"] for r in rows], dtype=np.float64
        )
        cols[f"lob_bid_vol_{k + 1}"] = np.array(
            [r[f"lob_bid_vol_{k + 1}"] for r in rows], dtype=np.float64
        )
    pq.write_table(pa.table(cols, schema=order_stream_schema()), str(path))
    return path


def _model():
    return OrderModel(
        ModelConfig(
            d_model=32, n_layers=2, n_heads=2, d_ff=64,
            film_layers=(0,), film_d_cond=16,
        )
    )


def _loader(path, tok):
    # shuffle=False: scoring must be deterministic so baseline/after are comparable.
    ds = OrderDataset([path], seq_len=SEQ, tokenizer=tok, max_cached=1)
    return DataLoader(ds, batch_size=BATCH, sampler=ChunkSampler(ds, shuffle=False))


@pytest.fixture
def assets(tmp_path):
    tok = OrderTokenizer()  # n_types=5 by default
    train_loaders = [
        _loader(_write_parquet(tmp_path / f"train{k}.parquet", seed=k), tok)
        for k in range(2)
    ]
    held_loader = _loader(_write_parquet(tmp_path / "held.parquet", seed=99), tok)
    return tok, train_loaders, held_loader


def _train_gradient(base, train_loaders):
    """Miner side: one budget-governed pass over the assigned pages, on a COPY
    so the scorer's base model is never touched."""
    miner_model = copy.deepcopy(base)
    delta = train_incremental(
        miner_model, train_loaders,
        WindowConfig(n_steps=0, budget_s=None, lr=3e-4), "cpu",
    )
    return delta


def test_miner_gradient_scores_on_disjoint_heldout(assets):
    """train_incremental -> compress -> evaluate_gradient on a disjoint held-out
    set produces a finite score and leaves the base model byte-identical."""
    _tok, train_loaders, held_loader = assets
    base = _model()
    base.eval()

    delta = _train_gradient(base, train_loaders)
    assert delta.metadata.steps_trained > 0
    assert delta.norm > 0.0  # weights actually moved
    comp = compress(delta, top_k_frac=0.5)

    before = {k: v.clone() for k, v in base.state_dict().items()}
    score = evaluate_gradient(base, comp, held_loader, device="cpu", max_batches=50)
    assert np.isfinite(score), f"held-out score must be finite, got {score}"

    # Per-miner isolation: evaluate_gradient rolls the base model back exactly,
    # so a per-round cached baseline stays valid across miners.
    after = base.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"param {k} not rolled back"


def test_single_pass_baseline_reuse_matches_inline(assets):
    """P0-1 single-pass: scoring with a precomputed shared baseline
    (loss_before=...) is identical to scoring that computes it inline. If they
    ever diverged, cached baselines would misscore every miner."""
    _tok, train_loaders, held_loader = assets
    base = _model()
    base.eval()

    comp = compress(_train_gradient(base, train_loaders), top_k_frac=0.5)

    baseline = _eval_loss(base, held_loader, "cpu", 50, 0.0)
    score_inline = evaluate_gradient(base, comp, held_loader, device="cpu", max_batches=50)
    score_reuse = evaluate_gradient(
        base, comp, held_loader, device="cpu", max_batches=50, loss_before=baseline
    )
    assert np.isfinite(baseline)
    assert score_inline == pytest.approx(score_reuse, abs=1e-6)
