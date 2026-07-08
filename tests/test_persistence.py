# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Tests for state-file loading / backup recovery (_try_load_state_file).

The review flagged load_state as near-untestable. Its core load-with-backup
recovery logic is now an extractable module function; these pin it: primary
load, fallback to the most-recent backup when the primary is corrupt, and
None when nothing is loadable.
"""
import msgpack

from taos.im.validator.persistence import _try_load_state_file


def _write(path, obj):
    path.write_bytes(msgpack.packb(obj, use_bin_type=True))


def test_loads_primary_when_valid(tmp_path):
    f = tmp_path / "validator.mp"
    _write(f, {"scores": [1, 2, 3], "step": 7})
    state = _try_load_state_file(str(f), "validator")
    # validator files decode with use_list=False → sequences come back as tuples
    assert tuple(state["scores"]) == (1, 2, 3) and state["step"] == 7


def test_missing_file_with_no_backups_returns_none(tmp_path):
    assert _try_load_state_file(str(tmp_path / "absent.mp"), "simulation") is None


def test_falls_back_to_most_recent_backup_when_primary_corrupt(tmp_path):
    primary = tmp_path / "validator.mp"
    primary.write_bytes(b"\x00\x01corrupt-not-msgpack\xff")
    # New-format backups: <name>.<numeric-marker>
    _write(tmp_path / "validator.mp.100", {"v": "old"})
    _write(tmp_path / "validator.mp.200", {"v": "newest"})
    state = _try_load_state_file(str(primary), "validator")
    assert state["v"] == "newest"          # most-recent marker wins
    # the good backup is restored as the new primary
    assert _try_load_state_file(str(primary), "validator")["v"] == "newest"


def test_simulation_uses_list_decoding(tmp_path):
    # simulation files decode with use_list=True (arrays stay lists)
    f = tmp_path / "sim.mp"
    _write(f, {"recent_trades": {0: [1, 2, 3]}})
    state = _try_load_state_file(str(f), "simulation")
    assert isinstance(state["recent_trades"][0], list)


def test_build_validator_state_snapshots_miner_stats():
    """miner_stats is persisted in the saved state (survives restart) and its
    per-uid dict + call_time list round-trip cleanly through msgpack."""
    from types import SimpleNamespace

    from taos.im.validator.persistence import build_validator_state

    class _Score:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    self_ = SimpleNamespace(
        step=5,
        simulation_timestamp=123,
        hotkeys=["a"],
        scores=[_Score(0.1)],
        gentrx_scores=[_Score(0.2)],
        activity_factors={},
        pnl_factors={},
        kappa_values={},
        unnormalized_scores={},
        deregistered_uids=[],
        miner_stats={
            0: {"requests": 105, "timeouts": 34, "failures": 0, "rejections": 0, "call_time": [1.0, 2.0]},
            1: {"requests": 50, "timeouts": 1, "failures": 2, "rejections": 0, "call_time": []},
        },
    )
    empty_vs = {
        "volume_sums": {},
        "maker_volume_sums": {},
        "taker_volume_sums": {},
        "self_volume_sums": {},
        "roundtrip_volume_sums": {},
    }
    out = build_validator_state(self_, {}, {}, empty_vs, {}, {}, {})
    assert out["miner_stats"][0]["requests"] == 105
    assert out["miner_stats"][0]["call_time"] == [1.0, 2.0]
    assert out["miner_stats"][1]["failures"] == 2

    rt = msgpack.unpackb(msgpack.packb(out, use_bin_type=True), raw=False, strict_map_key=False)
    assert rt["miner_stats"][0]["timeouts"] == 34
    assert rt["miner_stats"][1]["requests"] == 50


def _pack_via_stream(obj, depth):
    from taos.im.validator.persistence import _stream_pack
    packer = msgpack.Packer(use_bin_type=True)
    buf = bytearray()
    _stream_pack(packer, buf.extend, obj, depth)
    return bytes(buf)


def test_stream_pack_byte_identical_to_packb():
    """_stream_pack must produce EXACTLY the same bytes as msgpack.packb at every
    recursion depth — the save serialization is only safe if the chunked stream
    is byte-for-byte identical to what load_state expects to unpack."""
    obj = {
        "scores": {0: 1.5, 1: -2.25, 2: 0.0},
        "realized_pnl_history": {
            i: {1000 + j: {b: float(i * j * b) for b in range(3)} for j in range(4)}
            for i in range(5)
        },
        "meta": {"hotkeys": ["a", "b", "c"], "stake": [1.0, 2.0], "nested": {"x": None, "y": True}},
        "a_list": [1, "two", 3.0, {"k": "v"}, [4, 5]],
        "a_tuple_val": (1, 2, {"z": 9}),
        "bytes": b"\x00\x01\x02",
        7: "int-key",
        "empty_dict": {},
        "empty_list": [],
    }
    ref = msgpack.packb(obj, use_bin_type=True)
    ref_round = msgpack.unpackb(ref, raw=False, strict_map_key=False)
    for depth in (0, 1, 2, 3, 8):
        got = _pack_via_stream(obj, depth)
        assert got == ref, f"depth={depth}: not byte-identical to packb"
        assert msgpack.unpackb(got, raw=False, strict_map_key=False) == ref_round

    # Non-dict top-level object also byte-identical.
    for top in ([1, 2, {"a": 3}], 42, "scalar", None):
        assert _pack_via_stream(top, 2) == msgpack.packb(top, use_bin_type=True)
