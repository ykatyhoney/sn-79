# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""Protocol integrity + return-contract tests.

- GenTRXAssignment binds its payload into the signed body_hash (tamper-evident).
- AgentResponse / SimulationStateUpdate expose the declared return types.
"""
from taos.common.protocol import AgentResponse, SimulationStateUpdate
from taos.im.protocol.gentrx import GenTRXAssignment


def _assignment(**overrides):
    base = dict(
        round=5,
        model_version=2,
        books=["b1", "b2"],
        ts_start=100,
        ts_end=200,
        data=["s3://k1", "s3://k2"],
        data_source="s3",
        data_endpoint="https://acct.r2.cloudflarestorage.com",
        data_bucket="agent-256",
        data_access_key="ak",
        data_secret_key="sk",
        validator_uid=7,
    )
    base.update(overrides)
    return GenTRXAssignment(**base)


def test_gentrx_assignment_binds_payload_into_body_hash():
    a = _assignment()
    h = a.body_hash
    assert isinstance(h, str) and len(h) == 64  # SHA3-256 hex


def test_body_hash_stable_across_roundtrip():
    a = _assignment()
    dumped = a.model_dump()
    b = GenTRXAssignment(**{k: v for k, v in dumped.items() if k in GenTRXAssignment.model_fields})
    assert a.body_hash == b.body_hash


def test_body_hash_detects_tampered_bucket_and_credentials():
    a = _assignment()
    assert a.body_hash != _assignment(data_bucket="evil-bucket").body_hash
    assert a.body_hash != _assignment(data_secret_key="stolen").body_hash
    assert a.body_hash != _assignment(data=["s3://malicious"]).body_hash


def test_required_hash_fields_cover_credentials_and_window():
    fields = set(GenTRXAssignment.required_hash_fields)
    for must in ("data_endpoint", "data_bucket", "data_access_key", "data_secret_key", "ts_start", "ts_end"):
        assert must in fields


def test_agent_response_serialize_returns_list():
    # serialize() is annotated -> list[dict] and builds a list comprehension over
    # instructions; on an empty response it returns an empty list, not a dict.
    out = AgentResponse(agent_id=1).serialize()
    assert isinstance(out, list)


def test_simulation_state_update_deserialize_returns_response():
    class _S(SimulationStateUpdate):
        def environment_state(self):
            return None

        def agent_state(self):
            return None

    s = _S()
    assert s.deserialize() is None  # no response set yet
    resp = AgentResponse(agent_id=3)
    s.response = resp
    assert s.deserialize() is resp
