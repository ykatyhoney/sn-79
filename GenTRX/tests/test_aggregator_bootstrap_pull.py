"""Aggregator self-heals its scoring model from its own bucket head at boot.

Regression: on the canonical (aggregator) host, `_sync_from_uid0` is a
publish-side no-op, so a missing/quarantined local seed used to fresh-init and
then only *version-track* the bucket head (the "Resumed vN" branch never
reloads weights). Held-out scoring then ran on a fresh, random model and the
next aggregation published fresh-derived weights over a trained head.

`_bootstrap_from_own_bucket` fixes it: when checkpoint_path is absent and the
aggregator's bucket holds an architecture-compatible head, pull it down as the
local scoring checkpoint. Incompatible heads are left to the fresh-init path.

Run: pytest GenTRX/tests/test_aggregator_bootstrap_pull.py -v
"""

import io

import torch

from GenTRX.src.version import TRAIN_REGIME_VERSION


def _make_aggregator(tmp_path, validator_store=None, is_aggregator=True):
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
        validator_store=validator_store,
        is_aggregator=is_aggregator,
        no_startup_cleanup=True,
    )


def _ckpt_bytes(n_types=5, marker="trained"):
    """A real torch-saved checkpoint for the given architecture, tagged with a
    marker so a pulled checkpoint is distinguishable from a fresh-init one."""
    from dataclasses import asdict

    from GenTRX.src.model import ModelConfig, OrderModel
    from GenTRX.src.tokenizer import TokenizerConfig

    cfg = ModelConfig(n_types=n_types)
    model = OrderModel(cfg)
    buf = io.BytesIO()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(cfg),
            "tokenizer_config": asdict(TokenizerConfig()),
            "train_regime_version": TRAIN_REGIME_VERSION,
            "_test_marker": marker,
        },
        buf,
    )
    return buf.getvalue()


class _FakeBucketStore:
    """Minimal GradientStore stand-in exposing just the bootstrap read path."""

    def __init__(self, head_version=0, data=None, regime=TRAIN_REGIME_VERSION):
        self._head = head_version
        self._data = data
        self._regime = regime

    def get_head_version(self, uid):
        return self._head

    def get_head_meta(self, uid):
        return {"version": self._head}

    def get_latest_existing_version(self, uid):
        return self._head

    def get_latest_meta(self, uid):
        return {"train_regime_version": self._regime}

    def get_checkpoint(self, uid, version):
        if self._data is None:
            raise KeyError(version)
        return self._data


def test_bootstrap_pulls_compatible_head(tmp_path):
    store = _FakeBucketStore(head_version=4058, data=_ckpt_bytes(marker="trained-head"))
    agg = _make_aggregator(tmp_path, validator_store=store)
    agg._fresh_start = True  # a quarantine would have set this

    assert not agg.checkpoint_path.exists()
    assert agg._bootstrap_from_own_bucket() is True

    assert agg.checkpoint_path.exists()
    assert agg._version == 4058
    assert agg._fresh_start is False  # trained baseline recovered → no warmup

    loaded = torch.load(str(agg.checkpoint_path), map_location="cpu", weights_only=False)
    # marker proves it is the bucket head, NOT a fresh-init model
    assert loaded["_test_marker"] == "trained-head"
    assert loaded["model_config"]["n_types"] == 5


def test_bootstrap_skips_incompatible_head(tmp_path):
    store = _FakeBucketStore(head_version=100, data=_ckpt_bytes(n_types=3, marker="old-arch"))
    agg = _make_aggregator(tmp_path, validator_store=store)

    # wrong n_types must NOT be pulled — fresh-init handles it downstream
    assert agg._bootstrap_from_own_bucket() is False
    assert not agg.checkpoint_path.exists()


def test_bootstrap_noop_for_sibling(tmp_path):
    store = _FakeBucketStore(head_version=4058, data=_ckpt_bytes())
    agg = _make_aggregator(tmp_path, validator_store=store, is_aggregator=False)

    assert agg._bootstrap_from_own_bucket() is False
    assert not agg.checkpoint_path.exists()


def test_bootstrap_noop_empty_bucket(tmp_path):
    store = _FakeBucketStore(head_version=0, data=None)
    agg = _make_aggregator(tmp_path, validator_store=store)

    assert agg._bootstrap_from_own_bucket() is False
    assert not agg.checkpoint_path.exists()


def test_bootstrap_noop_without_store(tmp_path):
    agg = _make_aggregator(tmp_path, validator_store=None)

    assert agg._bootstrap_from_own_bucket() is False
    assert not agg.checkpoint_path.exists()


def test_quarantine_then_bootstrap_recovers_trained_weights(tmp_path):
    """The full remote scenario: stale n_types=3 local seed quarantined, then
    the aggregator recovers the trained compatible head from its own bucket
    instead of scoring on a fresh model."""
    from GenTRX.src.gradient_server import normalize_seed_checkpoint
    from GenTRX.src.model import ModelConfig

    store = _FakeBucketStore(head_version=4058, data=_ckpt_bytes(marker="trained-head"))
    agg = _make_aggregator(tmp_path, validator_store=store)

    # stale, architecture-incompatible local seed at the aggregator's path
    agg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agg.checkpoint_path.write_bytes(_ckpt_bytes(n_types=3, marker="stale-local"))

    action = normalize_seed_checkpoint(
        agg.checkpoint_path,
        current_n_types=ModelConfig().n_types,
        current_regime=TRAIN_REGIME_VERSION,
        label_smooth_sigma=1.0,
    )
    assert action == "quarantined"
    assert not agg.checkpoint_path.exists()  # moved aside
    agg._fresh_start = True

    assert agg._bootstrap_from_own_bucket() is True
    loaded = torch.load(str(agg.checkpoint_path), map_location="cpu", weights_only=False)
    assert loaded["_test_marker"] == "trained-head"  # trained baseline, not fresh
    assert loaded["model_config"]["n_types"] == 5
    assert agg._fresh_start is False
