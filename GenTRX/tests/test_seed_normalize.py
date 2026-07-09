"""Automatic seed-checkpoint normalization on deploy (no manual re-stamp step).

normalize_seed_checkpoint() runs at aggregator startup so a seed dropped on any
box is made usable under the current code automatically:
  - current-architecture seed, old/absent regime stamp -> re-stamped in place;
  - architecture-incompatible seed (wrong n_types)      -> quarantined so the
    server fresh-inits a usable model instead of crashing on the first gradient.

Run: pytest GenTRX/tests/test_seed_normalize.py -v
"""
import torch

from GenTRX.src.gradient_server import normalize_seed_checkpoint
from GenTRX.src.version import TRAIN_REGIME_VERSION

CUR_NT = 5  # current build n_types


def _write_ckpt(path, *, n_types, regime):
    torch.save(
        {
            "model_config": {"n_types": n_types, "d_model": 32},
            "tokenizer_config": {"n_types": n_types},
            "train_regime_version": regime,
            "label_smooth_sigma": 0.0,
            "model_state_dict": {"w": torch.zeros(3)},
        },
        str(path),
    )


def test_absent_seed(tmp_path):
    assert normalize_seed_checkpoint(
        tmp_path / "nope.pt", CUR_NT, TRAIN_REGIME_VERSION, 0.0
    ) == "absent"


def test_current_seed_untouched(tmp_path):
    p = tmp_path / "seed.pt"
    _write_ckpt(p, n_types=CUR_NT, regime=TRAIN_REGIME_VERSION)
    mtime_before = p.stat().st_mtime_ns
    assert normalize_seed_checkpoint(p, CUR_NT, TRAIN_REGIME_VERSION, 0.0) == "current"
    assert p.exists() and p.stat().st_mtime_ns == mtime_before  # not rewritten


def test_compatible_old_regime_is_restamped(tmp_path):
    """Same architecture, older regime stamp -> re-stamped to current in place."""
    p = tmp_path / "seed.pt"
    _write_ckpt(p, n_types=CUR_NT, regime=TRAIN_REGIME_VERSION - 1)
    assert normalize_seed_checkpoint(p, CUR_NT, TRAIN_REGIME_VERSION, 0.0) == "restamped"
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    assert ck["train_regime_version"] == TRAIN_REGIME_VERSION
    assert ck["model_config"]["n_types"] == CUR_NT  # weights/arch untouched


def test_incompatible_ntypes_is_quarantined(tmp_path):
    """Wrong n_types (e.g. old n_types=3 seed under n_types=5 code) -> moved
    aside so the caller fresh-inits a compatible model instead of crashing."""
    p = tmp_path / "seed.pt"
    _write_ckpt(p, n_types=3, regime=TRAIN_REGIME_VERSION)
    assert normalize_seed_checkpoint(p, CUR_NT, TRAIN_REGIME_VERSION, 0.0) == "quarantined"
    assert not p.exists()  # original moved out of the way
    quarantined = p.with_name(p.name + ".incompatible-n_types3")
    assert quarantined.exists()


def test_incompatible_takes_priority_over_regime(tmp_path):
    """An incompatible seed that is also regime-behind is quarantined, never
    re-stamped (re-stamping an unusable architecture would be wrong)."""
    p = tmp_path / "seed.pt"
    _write_ckpt(p, n_types=3, regime=TRAIN_REGIME_VERSION - 1)
    assert normalize_seed_checkpoint(p, CUR_NT, TRAIN_REGIME_VERSION, 0.0) == "quarantined"


# ---------------------------------------------------------------------------
# Aggregator bucket re-baseline (existing-bucket migration, e.g. remote)
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal GradientStore stand-in: records publishes, reports a stale
    published regime so _regime_incompatible() reads incompatible."""

    def __init__(self, regime):
        self._regime = regime
        self.puts = []
        self.heads = []

    def get_latest_meta(self, uid):
        return {"train_regime_version": self._regime}

    def put_checkpoint(self, uid, version, data, meta=None):
        self.puts.append((version, meta))
        return "key"

    def put_head_version(self, uid, version, meta=None):
        self.heads.append((version, meta))


def _aggregator(tmp_path):
    from GenTRX.src.gradient_server import GradientAggregator

    return GradientAggregator(
        checkpoint_path=str(tmp_path / "ckpt.pt"),
        val_data_path=str(tmp_path / "val"),
        output_path=str(tmp_path / "out.pt"),
        books_per_miner=1,
        interval=60,
        window_ns=50,
        warmup_rounds=0,
        rollback=False,
    )


def test_rebaseline_publishes_compatible_checkpoint(tmp_path):
    """A stale (pre-regime) published checkpoint is superseded by a new version
    carrying the current regime stamp, so miners bootstrap a compatible model."""
    agg = _aggregator(tmp_path)
    agg.is_aggregator = True
    agg.validator_store = _FakeStore(regime=TRAIN_REGIME_VERSION - 1)
    agg._version = 5
    (tmp_path / "ckpt.pt").write_bytes(b"CKPT")

    assert agg._regime_incompatible() is True
    assert agg._rebaseline_incompatible_bucket() is True

    assert [v for v, _ in agg.validator_store.puts] == [6]  # bumped, additive
    _v, meta = agg.validator_store.puts[0]
    assert meta["train_regime_version"] == TRAIN_REGIME_VERSION
    assert agg.validator_store.heads[0][0] == 6
    assert agg._version == 6


def test_rebaseline_is_aggregator_only(tmp_path):
    """Siblings pull, they never publish — no re-baseline on a non-aggregator."""
    agg = _aggregator(tmp_path)
    agg.is_aggregator = False
    agg.validator_store = _FakeStore(regime=TRAIN_REGIME_VERSION - 1)
    (tmp_path / "ckpt.pt").write_bytes(b"CKPT")
    assert agg._rebaseline_incompatible_bucket() is False
    assert agg.validator_store.puts == []
