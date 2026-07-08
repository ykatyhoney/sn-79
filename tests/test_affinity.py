# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Tests for get_core_allocation's simulator-slice + OS-headroom reservation.

The sim slice must be disjoint from the reward and validator (save/event-loop)
cores so the C++ simulator never contends with the backgrounded reward pool /
save-pack. It is sized to blockCount (sim_cores_count), funded from the rounding
gap first and then ONLY from reward; validator/query/reporting/ipc are never
trimmed. Legacy callers (sim_cores_count=0) get the exact previous allocation.
"""
import multiprocessing


def _alloc(total, **kw):
    from taos.im.utils import affinity

    orig = multiprocessing.cpu_count
    multiprocessing.cpu_count = lambda: total
    try:
        return affinity.get_core_allocation(**kw)
    finally:
        multiprocessing.cpu_count = orig


def _assert_disjoint(a):
    seen = set()
    for cores in a.values():
        s = set(cores)
        assert not (s & seen), f"core overlap across slices: {a}"
        seen |= s


def test_legacy_no_sim_unchanged():
    a = _alloc(64)
    assert "sim" not in a
    assert len(a["reward"]) == 16
    _assert_disjoint(a)


def test_mainnet_sim8_headroom4():
    a = _alloc(64, sim_cores_count=8, os_headroom=4)
    assert a["sim"] == list(range(52, 60))  # 8 dedicated cores
    assert len(a["reward"]) == 13  # 16 - 3 deficit
    assert len(a["validator"]) == 12 and len(a["query"]) == 12  # round path untouched
    _assert_disjoint(a)
    # sim disjoint from reward AND validator (the save / event-loop cores)
    assert not (set(a["sim"]) & set(a["reward"]))
    assert not (set(a["sim"]) & set(a["validator"]))
    # 4 cores left entirely unpinned (OS/IRQ lane)
    used = {c for v in a.values() for c in v}
    assert sorted(set(range(64)) - used) == [60, 61, 62, 63]


def test_sim_slice_scales_with_blockcount():
    a8 = _alloc(64, sim_cores_count=8, os_headroom=4)
    a16 = _alloc(64, sim_cores_count=16, os_headroom=4)
    assert len(a16["sim"]) == 16
    assert len(a16["reward"]) < len(a8["reward"])
    _assert_disjoint(a16)


def test_gap_absorbs_small_sim_without_trimming_reward():
    # sim(4)+headroom(2)=6 <= rounding gap(9) -> reward untouched
    a = _alloc(64, sim_cores_count=4, os_headroom=2)
    assert len(a["reward"]) == 16
    assert len(a["sim"]) == 4
    _assert_disjoint(a)
