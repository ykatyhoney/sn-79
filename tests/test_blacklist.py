# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Regression tests for BaseMinerNeuron.blacklist().

Guards the ordering fix: the registration-membership check MUST run before
metagraph.hotkeys.index(), otherwise an unregistered hotkey raises ValueError
instead of returning a clean (True, reason). See taos/common/neurons/miner.py.
"""
from types import SimpleNamespace

import bittensor as bt

from taos.common.neurons.miner import BaseMinerNeuron


def _make_self(hotkeys, validator_permit, allow_non_registered, allow_non_validators):
    """Minimal stand-in exposing only what blacklist() touches."""
    return SimpleNamespace(
        metagraph=SimpleNamespace(hotkeys=list(hotkeys), validator_permit=list(validator_permit)),
        config=SimpleNamespace(
            blacklist=SimpleNamespace(
                allow_non_registered=allow_non_registered,
                allow_non_validators=allow_non_validators,
            )
        ),
    )


def _synapse(hotkey):
    syn = bt.Synapse()
    syn.dendrite = bt.TerminalInfo(hotkey=hotkey)
    return syn


def test_unregistered_hotkey_is_rejected_not_raised():
    """An unregistered hotkey must return (True, ...), never raise ValueError."""
    me = _make_self(
        hotkeys=["5Registered"],
        validator_permit=[True],
        allow_non_registered=False,
        allow_non_validators=False,
    )
    blacklisted, reason = BaseMinerNeuron.blacklist(me, _synapse("5Unknown"))
    assert blacklisted is True
    assert "nrecognized" in reason or "nregistered" in reason.lower()


def test_unregistered_allowed_when_configured():
    """With allow_non_registered, an unknown hotkey is admitted without an index() crash."""
    me = _make_self(
        hotkeys=["5Registered"],
        validator_permit=[True],
        allow_non_registered=True,
        allow_non_validators=False,
    )
    blacklisted, _ = BaseMinerNeuron.blacklist(me, _synapse("5Unknown"))
    assert blacklisted is False


def test_registered_non_validator_blacklisted_when_validator_required():
    me = _make_self(
        hotkeys=["5A", "5B"],
        validator_permit=[True, False],
        allow_non_registered=False,
        allow_non_validators=False,
    )
    blacklisted, reason = BaseMinerNeuron.blacklist(me, _synapse("5B"))
    assert blacklisted is True
    assert "validator" in reason.lower()


def test_registered_validator_allowed():
    me = _make_self(
        hotkeys=["5A", "5B"],
        validator_permit=[True, False],
        allow_non_registered=False,
        allow_non_validators=False,
    )
    blacklisted, _ = BaseMinerNeuron.blacklist(me, _synapse("5A"))
    assert blacklisted is False
