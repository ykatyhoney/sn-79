"""Safety tests for gradient deserialization.

Pins the hardened-load contract on the miner -> validator path:

  - torch.load runs with weights_only=True, so a pickled __reduce__
    payload from an untrusted bucket cannot execute on load.
  - After load, every field is validated against the receiver's model:
    top-level keys, metadata types, param names, dtype, shape, finite
    values, and index range. Anything off-spec raises ValueError and
    the caller drops the gradient.

Run: pytest GenTRX/tests/test_gradient_deserialize_safety.py -v
"""

import io
import math
import pickle

import pytest
import torch

from GenTRX.src.gradient import deserialize, safe_torch_load, serialize
from GenTRX.src.gradient import CompressedGradient, GradientMetadata


def _expected_shapes():
    return {
        "layer.weight": torch.Size([4, 4]),
        "layer.bias": torch.Size([4]),
    }


def _legit_compressed(meta_overrides: dict | None = None) -> CompressedGradient:
    sparse = {
        "layer.weight": (
            torch.tensor([0, 5, 10], dtype=torch.int64),
            torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
            torch.Size([4, 4]),
        ),
        "layer.bias": (
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0.4], dtype=torch.float32),
            torch.Size([4]),
        ),
    }
    meta = GradientMetadata(
        window_id=1, miner_uid=42, steps_trained=10,
        loss_before=1.5, loss_after=1.2, loss_trajectory=[1.5, 1.4, 1.2],
    )
    if meta_overrides:
        for k, v in meta_overrides.items():
            setattr(meta, k, v)
    return CompressedGradient(sparse=sparse, metadata=meta)


def _raw_payload(metadata: dict, sparse: dict) -> bytes:
    buf = io.BytesIO()
    torch.save({"metadata": metadata, "sparse": sparse}, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_legit_gradient_round_trips():
    comp = _legit_compressed()
    data = serialize(comp)
    out = deserialize(data, expected_shapes=_expected_shapes())
    assert out.metadata.window_id == 1
    assert out.metadata.miner_uid == 42
    assert set(out.sparse.keys()) == {"layer.weight", "layer.bias"}
    idx, vals, shape = out.sparse["layer.weight"]
    assert torch.equal(idx, torch.tensor([0, 5, 10], dtype=torch.int64))
    assert torch.allclose(vals, torch.tensor([0.1, -0.2, 0.3]))
    assert tuple(shape) == (4, 4)


# ---------------------------------------------------------------------------
# Code-exec via pickle: weights_only=True is the firewall
# ---------------------------------------------------------------------------


class _ReducePayload:
    """A pickle that, when loaded with weights_only=False, would invoke os.system."""

    def __reduce__(self):
        import os
        return (os.system, ("true",))


def test_safe_torch_load_rejects_pickle_reduce_payload():
    blob = pickle.dumps(_ReducePayload(), protocol=2)
    with pytest.raises(Exception):
        safe_torch_load(io.BytesIO(blob))


def test_deserialize_rejects_pickle_reduce_payload():
    blob = pickle.dumps(_ReducePayload(), protocol=2)
    with pytest.raises(Exception):
        deserialize(blob, expected_shapes=_expected_shapes())


# ---------------------------------------------------------------------------
# Schema gate
# ---------------------------------------------------------------------------


def test_rejects_non_dict_payload():
    buf = io.BytesIO()
    torch.save(torch.zeros(4), buf)
    with pytest.raises(ValueError, match="malformed"):
        deserialize(buf.getvalue(), expected_shapes=_expected_shapes())


def test_rejects_unexpected_top_level_keys():
    buf = io.BytesIO()
    torch.save({"metadata": {}, "sparse": {}, "extra": 1}, buf)
    with pytest.raises(ValueError, match="malformed"):
        deserialize(buf.getvalue(), expected_shapes=_expected_shapes())


def test_rejects_unknown_param_name():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "rogue.param": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([0.0], dtype=torch.float32),
                "shape": [4],
            },
        },
    )
    with pytest.raises(ValueError, match="unknown param"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_shape_mismatch():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([0.0], dtype=torch.float32),
                "shape": [8],
            },
        },
    )
    with pytest.raises(ValueError, match="shape"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_non_int_indices_dtype():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([0.0], dtype=torch.float32),
                "values": torch.tensor([0.0], dtype=torch.float32),
                "shape": [4],
            },
        },
    )
    with pytest.raises(ValueError, match="indices dtype"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_index_out_of_range():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([99], dtype=torch.int64),
                "values": torch.tensor([0.0], dtype=torch.float32),
                "shape": [4],
            },
        },
    )
    with pytest.raises(ValueError, match="out of range"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_nan_values():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([float("nan")], dtype=torch.float32),
                "shape": [4],
            },
        },
    )
    with pytest.raises(ValueError, match="non-finite"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_inf_values():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([math.inf], dtype=torch.float32),
                "shape": [4],
            },
        },
    )
    with pytest.raises(ValueError, match="non-finite"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_indices_values_size_mismatch():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([0, 1, 2], dtype=torch.int64),
                "values": torch.tensor([0.0], dtype=torch.float32),
                "shape": [4],
            },
        },
    )
    with pytest.raises(ValueError, match="size mismatch"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_missing_sparse_entry_fields():
    data = _raw_payload(
        metadata={"window_id": 1, "miner_uid": 1, "steps_trained": 1,
                  "loss_before": 0.0, "loss_after": 0.0, "loss_trajectory": []},
        sparse={
            "layer.bias": {
                "indices": torch.tensor([0], dtype=torch.int64),
                "values": torch.tensor([0.0], dtype=torch.float32),
            },
        },
    )
    with pytest.raises(ValueError, match="malformed"):
        deserialize(data, expected_shapes=_expected_shapes())


def test_rejects_non_dict_metadata():
    buf = io.BytesIO()
    torch.save({"metadata": [1, 2, 3], "sparse": {}}, buf)
    with pytest.raises(ValueError, match="metadata must be a dict"):
        deserialize(buf.getvalue(), expected_shapes=_expected_shapes())
