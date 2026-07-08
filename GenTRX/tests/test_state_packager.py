"""Tests for StatePackager — config extraction never fabricates decimals.

Pin the rule: the packet carries a config block only when the state actually
has one. priceDecimals/volumeDecimals are passed through when present and
omitted (with a warning) when absent — never defaulted to a made-up value.

Run: pytest GenTRX/tests/test_state_packager.py -v
"""

from GenTRX.src.state_packager import StatePackager


def _state(config=None, ts: int = 1000):
    s: dict = {"books": {}, "timestamp": ts}
    if config is not None:
        s["config"] = config
    return s


def test_no_config_block_when_state_has_none():
    """State without a config object yields a packet with no config key."""
    packet = StatePackager().extract_state(_state())
    assert "config" not in packet
    assert "sim_id" not in packet


def test_config_passed_through_when_present():
    pkg = StatePackager()
    packet = pkg.extract_state(
        _state({"priceDecimals": 2, "volumeDecimals": 4, "simulation_id": "SIM_A"})
    )
    assert packet["config"] == {
        "priceDecimals": 2,
        "volumeDecimals": 4,
        "simulation_id": "SIM_A",
    }
    assert packet["sim_id"] == "SIM_A"


def test_missing_decimals_omitted_not_defaulted(caplog):
    """Config present but missing decimals: keys omitted, no made-up 8s."""
    packet = StatePackager().extract_state(_state({"simulation_id": "SIM_B"}))
    cfg = packet["config"]
    assert "priceDecimals" not in cfg
    assert "volumeDecimals" not in cfg
    assert cfg["simulation_id"] == "SIM_B"
    assert any("missing decimals" in r.message for r in caplog.records)


def test_config_emitted_every_tick():
    """Config (decimals + sim_id) rides EVERY tick, so a gradient server that
    restarts mid-sim always re-receives it and binds the correct price/volume
    scale instead of falling back to default decimals."""
    pkg = StatePackager()
    cfg = {"priceDecimals": 2, "volumeDecimals": 4, "simulation_id": "SIM_C"}
    first = pkg.extract_state(_state(cfg))
    second = pkg.extract_state(_state(cfg))
    third = pkg.extract_state(_state(cfg))
    for pkt in (first, second, third):
        assert pkt["config"] == {
            "priceDecimals": 2,
            "volumeDecimals": 4,
            "simulation_id": "SIM_C",
        }
        assert pkt["sim_id"] == "SIM_C"


def test_sim_events_extracted_from_uid_keyed_notices():
    """notices is dict[uid]->[events]; ESS is broadcast to every uid."""
    state = _state()
    state["notices"] = {0: [{"y": "ESS"}], 1: [{"y": "ESS"}]}
    packet = StatePackager().extract_state(state)
    assert packet.get("sim_events") == ["ESS"]


def test_sim_events_extracted_from_flat_notices():
    """A flat list of notices is tolerated too."""
    state = _state()
    state["notices"] = [{"y": "ESE"}]
    packet = StatePackager().extract_state(state)
    assert packet.get("sim_events") == ["ESE"]


def test_ess_marker_carries_config():
    """A SimulationStartEvent tick still carries the config block (config now
    rides every tick) and re-binds sim_id for the new sim."""
    pkg = StatePackager()
    cfg = {"priceDecimals": 2, "volumeDecimals": 4, "simulation_id": "SIM_D"}
    pkg.extract_state(_state(cfg))
    restart = _state(cfg)
    restart["notices"] = {0: [{"y": "ESS"}], 1: [{"y": "ESS"}]}
    packet = pkg.extract_state(restart)
    assert "config" in packet
    assert packet["sim_id"] == "SIM_D"
    assert packet.get("sim_events") == ["ESS"]
