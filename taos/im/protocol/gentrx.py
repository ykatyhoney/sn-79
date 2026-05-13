# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import json

import bittensor as bt
from pydantic import ConfigDict, model_validator


class GenTRXAssignment(bt.Synapse):
    """Pushed from validator to miner once per GenTRX aggregation round.

    Carries the book IDs, timestamp window, and pre-resolved S3 data keys the
    miner should download and train on.  Miners that don't run GenTRXAgent
    return a non-200 status which the validator silently ignores.

    Fields match the dict returned by GradientAggregator.get_assignment().
    """

    model_config = ConfigDict(protected_namespaces=())

    # bittensor 10.2's parent validator does `values["name"] = ...` and crashes
    # when the dendrite ships the synapse as a raw-bytes body. Decode here.
    @model_validator(mode="before")
    @classmethod
    def set_name_type(cls, values):
        if isinstance(values, (bytes, bytearray)):
            try:
                values = json.loads(values)
            except (json.JSONDecodeError, ValueError):
                return values
        if isinstance(values, dict):
            values["name"] = cls.__name__
        return values

    round: int = 0
    model_version: int = 0
    books: list[str] = []
    ts_start: int = 0
    ts_end: int = 0
    data: list[str] = []        # S3 keys for training parquets
    data_source: str = "s3"     # "s3" or "local"

    # Validator's data bucket 
    # Enables per-validator data buckets access without chain commitments.
    data_endpoint: str = ""
    data_bucket: str = ""
    data_access_key: str = ""
    data_secret_key: str = ""

    # UID of the validator that issued this assignment. 
    validator_uid: int = -1
