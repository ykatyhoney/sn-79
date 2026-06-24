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
