# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Coverage for BaseValidatorNeuron.set_weights (review: weight-setting path untested).

Locks the contract that the submitted version_key is the validator's spec_version
(the standard Bittensor pattern the chain checks against its minimum
WeightsVersionKey) and that observe mode skips submission.
"""
from types import SimpleNamespace

from taos.common.neurons.validator import BaseValidatorNeuron


def _recorder():
    calls = {}

    def set_weights(**kwargs):
        calls.update(kwargs)
        return True, "ok"

    return calls, set_weights


def _make_self(observe=False, spec_version=50, engine="simulation"):
    calls, set_weights = _recorder()
    me = SimpleNamespace(
        config=SimpleNamespace(neuron=SimpleNamespace(observe=observe), netuid=79, engine=engine),
        hyperparams=SimpleNamespace(commit_reveal_weights_enabled=False),
        wallet=object(),
        spec_version=spec_version,
        subtensor=SimpleNamespace(set_weights=set_weights),
        prepare_weights=lambda: ([0, 1], [0.5, 0.5]),
    )
    return me, calls


def test_set_weights_submits_spec_version_as_version_key():
    me, calls = _make_self(spec_version=50)
    result = BaseValidatorNeuron.set_weights(me)
    assert result is True
    assert calls["version_key"] == 50          # spec_version, not a hard-coded/stale value
    assert calls["netuid"] == 79
    assert calls["uids"] == [0, 1]


def test_set_weights_observe_mode_skips_submission():
    me, calls = _make_self(observe=True)
    BaseValidatorNeuron.set_weights(me)
    assert calls == {}                          # subtensor.set_weights never called
