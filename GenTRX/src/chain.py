# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# Portions of this code are based on tplr-ai/templar, licensed under MIT.
# Adapted from Templar:
#   - 128-char commitment format (account_id 32 + access_key_id 32 + secret_access_key 64)
#   - BucketInfo field layout (account_id, access_key_id, secret_access_key)
#   - from_commitment() / to_commitment() field slicing and padding logic
#   - get_miner_buckets(): query_map(module="Commitments"), ss58_encode(),
#     hotkey_to_uid dict construction, and commitment bytes decoding
#   - commit_bucket(): existing-commitment check before writing + v9.4 API shim
# See NOTICE or README for full attribution.
"""On-chain bucket commitment for GenTRX distributed training.

Miners commit their S3 bucket read credentials on-chain so the validator
can discover and read gradients from each miner's bucket. Uses bittensor's
built-in Commitments pallet (same pattern as Templar).

Supports Cloudflare R2, Storj, and Hippius S3. Provider is auto-detected
from the account_id slot of the commitment (see BucketInfo).

Commitment format (128 chars):
    account_id (32) + access_key_id (32) + secret_access_key (64)

Usage (miner):
    chain = GenTRXChain(subtensor, netuid, metagraph)
    chain.commit_bucket(wallet, BucketInfo(...))

Usage (validator):
    chain = GenTRXChain(subtensor, netuid, metagraph)
    buckets = await chain.get_miner_buckets()
    # buckets = {uid: BucketInfo, ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

COMMITMENT_LEN = 128
ACCOUNT_ID_LEN = 32
ACCESS_KEY_LEN = 32
SECRET_KEY_LEN = 64

STORJ_PREFIX = "storj:"
STORJ_ENDPOINT = "https://gateway.storjshare.io"
HIPPIUS_ENDPOINT = "https://s3.hippius.com"


_R2_ACCOUNT_ID_RE = None  # compiled lazily


def _is_r2_account_id(value: str) -> bool:
    """Return True if value looks like a Cloudflare R2 account ID (32 hex chars)."""
    global _R2_ACCOUNT_ID_RE
    if _R2_ACCOUNT_ID_RE is None:
        import re
        _R2_ACCOUNT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
    return bool(_R2_ACCOUNT_ID_RE.match(value))


def _resolve_account_id(bucket: str, account_id_env: str, provider_env: str) -> str:
    """Build the account_id slot value for an on-chain commitment.

    Provider hint comes from the GENTRX_*_S3_PROVIDER env var:
      "storj":   prepend the "storj:" sigil to the bucket name
      "r2":      use account_id_env (Cloudflare account ID)
      "hippius": use bucket name (or account_id_env if explicitly set)
      unset:     existing behavior — explicit account_id_env wins, else bucket
    """
    provider = (provider_env or "").strip().lower()
    if provider == "storj":
        return f"{STORJ_PREFIX}{bucket}"
    return account_id_env or bucket


@dataclass
class BucketInfo:
    """S3 bucket credentials parsed from on-chain commitment.

    Supports Cloudflare R2, Hippius S3, and Storj. Provider is auto-detected
    from the account_id field:

        R2      — account_id is a 32-char lowercase hex Cloudflare account ID.
                  Endpoint:  https://{account_id}.r2.cloudflarestorage.com
                  Region:    auto
                  Bucket:    account_id (convention: bucket == account_id for R2)

        Storj   — account_id starts with "storj:" followed by the bucket name.
                  Endpoint:  https://gateway.storjshare.io  (static)
                  Region:    global
                  Bucket:    account_id without the "storj:" prefix

        Hippius — account_id field stores the bucket name (padded to 32 chars).
                  Endpoint:  https://s3.hippius.com  (static)
                  Region:    decentralized
                  Bucket:    account_id.strip()

    The 128-char on-chain format is identical for all providers:
        account_id (32) + access_key_id (32) + secret_access_key (64)

    For local MinIO / other providers set _endpoint_override at runtime.
    """

    account_id: str
    access_key_id: str
    secret_access_key: str
    _endpoint_override: str | None = None
    _bucket_override: str | None = None

    @property
    def is_r2(self) -> bool:
        """True when account_id is a 32-char lowercase hex Cloudflare account ID."""
        return _is_r2_account_id(self.account_id.strip())

    @property
    def is_storj(self) -> bool:
        """True when account_id starts with the Storj prefix sigil."""
        return self.account_id.strip().startswith(STORJ_PREFIX)

    @property
    def endpoint_url(self) -> str:
        """Derive endpoint from provider, or use override for local dev."""
        if self._endpoint_override:
            return self._endpoint_override
        if self.is_r2:
            return f"https://{self.account_id.strip()}.r2.cloudflarestorage.com"
        if self.is_storj:
            return STORJ_ENDPOINT
        return HIPPIUS_ENDPOINT

    @property
    def region(self) -> str:
        """AWS region string for boto3 (provider-specific)."""
        if self._endpoint_override:
            return "auto"
        if self.is_r2:
            return "auto"
        if self.is_storj:
            return "global"
        return "decentralized"

    @property
    def bucket_name(self) -> str:
        """Bucket name.

        R2: account_id by convention (new account per bucket).
        Storj: account_id minus the "storj:" prefix.
        Hippius: account_id field IS the bucket name.
        Override: _bucket_override wins (local dev / MinIO).
        """
        if self._bucket_override:
            return self._bucket_override
        aid = self.account_id.strip()
        if aid.startswith(STORJ_PREFIX):
            return aid[len(STORJ_PREFIX):]
        return aid

    def to_commitment(self) -> str:
        """Serialize to 128-char commitment string."""
        if len(self.account_id) > ACCOUNT_ID_LEN:
            raise ValueError(
                f"account_id length {len(self.account_id)} exceeds slot "
                f"{ACCOUNT_ID_LEN}: {self.account_id!r}"
            )
        if len(self.access_key_id) > ACCESS_KEY_LEN:
            raise ValueError(
                f"access_key_id length {len(self.access_key_id)} exceeds slot "
                f"{ACCESS_KEY_LEN}: {self.access_key_id!r}"
            )
        if len(self.secret_access_key) > SECRET_KEY_LEN:
            raise ValueError(
                f"secret_access_key length {len(self.secret_access_key)} exceeds "
                f"slot {SECRET_KEY_LEN}"
            )
        aid = self.account_id.ljust(ACCOUNT_ID_LEN)
        akid = self.access_key_id.ljust(ACCESS_KEY_LEN)
        sk = self.secret_access_key.ljust(SECRET_KEY_LEN)
        return aid + akid + sk

    @classmethod
    def from_commitment(cls, data: str) -> BucketInfo:
        """Parse a 128-char commitment string back to BucketInfo."""
        if len(data) != COMMITMENT_LEN:
            raise ValueError(
                f"Commitment length {len(data)}, expected {COMMITMENT_LEN}"
            )
        return cls(
            account_id=data[:ACCOUNT_ID_LEN].strip(),
            access_key_id=data[
                ACCOUNT_ID_LEN : ACCOUNT_ID_LEN + ACCESS_KEY_LEN
            ].strip(),
            secret_access_key=data[ACCOUNT_ID_LEN + ACCESS_KEY_LEN :].strip(),
        )

    @classmethod
    def from_aggregator_env(cls) -> BucketInfo | None:
        """Last-resort env-var pointer to uid-0's checkpoint bucket.

        Returns None unless the operator has populated GENTRX_AGGREGATOR_S3_BUCKET
        plus a read key pair. The normal discovery path is the chain commitment;
        these env vars exist only for operator overrides.
        """
        import os

        bucket = os.environ.get("GENTRX_AGGREGATOR_S3_BUCKET", "")
        if not bucket:
            return None

        account_id = _resolve_account_id(
            bucket,
            os.environ.get("GENTRX_AGGREGATOR_S3_ACCOUNT_ID", ""),
            os.environ.get("GENTRX_AGGREGATOR_S3_PROVIDER", ""),
        )

        read_access = (
            os.environ.get("GENTRX_AGGREGATOR_S3_READ_ACCESS_KEY", "")
            or os.environ.get("GENTRX_AGGREGATOR_S3_ACCESS_KEY", "")
        )
        read_secret = (
            os.environ.get("GENTRX_AGGREGATOR_S3_READ_SECRET_KEY", "")
            or os.environ.get("GENTRX_AGGREGATOR_S3_SECRET_KEY", "")
        )

        if not read_access or not read_secret:
            return None

        return cls(
            account_id=account_id,
            access_key_id=read_access,
            secret_access_key=read_secret,
            _bucket_override=bucket if bucket != account_id else None,
        )

    @classmethod
    def from_validator_env(cls) -> BucketInfo | None:
        """Build BucketInfo from unified validator bucket environment variables.

        Used by validators to commit their bucket (data + scores + checkpoints)
        to chain so miners and the aggregator can discover it without pre-config.

        R2: set GENTRX_VALIDATOR_S3_ACCOUNT_ID to the 32-char Cloudflare account
            ID; endpoint is derived automatically.
        Storj: set GENTRX_VALIDATOR_S3_PROVIDER=storj. The bucket name is stored
            in the account_id field with a "storj:" prefix.
        Hippius: set GENTRX_VALIDATOR_S3_BUCKET to the bucket name; it is stored
            in the account_id field (endpoint derived as https://s3.hippius.com).

        Required:
            GENTRX_VALIDATOR_S3_BUCKET — bucket name

        Read-only credentials (committed on-chain):
            GENTRX_VALIDATOR_S3_READ_ACCESS_KEY (falls back to GENTRX_VALIDATOR_S3_ACCESS_KEY)
            GENTRX_VALIDATOR_S3_READ_SECRET_KEY (falls back to GENTRX_VALIDATOR_S3_SECRET_KEY)

        Optional:
            GENTRX_VALIDATOR_S3_PROVIDER   — "r2" | "storj" | "hippius" (provider hint)
            GENTRX_VALIDATOR_S3_ACCOUNT_ID — R2 account ID (defaults to bucket name)

        Returns None if required env vars are missing.
        """
        import os

        bucket = os.environ.get("GENTRX_VALIDATOR_S3_BUCKET", "")
        if not bucket:
            return None

        account_id = _resolve_account_id(
            bucket,
            os.environ.get("GENTRX_VALIDATOR_S3_ACCOUNT_ID", ""),
            os.environ.get("GENTRX_VALIDATOR_S3_PROVIDER", ""),
        )

        read_access = (
            os.environ.get("GENTRX_VALIDATOR_S3_READ_ACCESS_KEY", "")
            or os.environ.get("GENTRX_VALIDATOR_S3_ACCESS_KEY", "")
        )
        read_secret = (
            os.environ.get("GENTRX_VALIDATOR_S3_READ_SECRET_KEY", "")
            or os.environ.get("GENTRX_VALIDATOR_S3_SECRET_KEY", "")
        )

        if not read_access or not read_secret:
            return None

        return cls(
            account_id=account_id,
            access_key_id=read_access,
            secret_access_key=read_secret,
            _bucket_override=bucket if bucket != account_id else None,
        )

    @classmethod
    def from_env(cls) -> BucketInfo | None:
        """Build BucketInfo from miner environment variables.

        Required:
            GENTRX_AGENT_S3_BUCKET — bucket name (also used as account_id for
                                    MinIO; for R2, set GENTRX_AGENT_S3_ACCOUNT_ID;
                                    for Storj, set GENTRX_AGENT_S3_PROVIDER=storj)

        Read-only credentials (committed on-chain):
            GENTRX_AGENT_S3_READ_ACCESS_KEY (falls back to GENTRX_AGENT_S3_ACCESS_KEY)
            GENTRX_AGENT_S3_READ_SECRET_KEY (falls back to GENTRX_AGENT_S3_SECRET_KEY)

        Optional:
            GENTRX_AGENT_S3_PROVIDER   — "r2" | "storj" | "hippius" (provider hint)
            GENTRX_AGENT_S3_ACCOUNT_ID — Cloudflare account ID (R2 only)
            GENTRX_CHAIN_ENDPOINT_OVERRIDE — S3 endpoint override for MinIO
                                             localnet (used by service, not
                                             committed)

        Returns None if required env vars are missing.
        """
        import os

        bucket = os.environ.get("GENTRX_AGENT_S3_BUCKET", "")
        if not bucket:
            return None

        account_id = _resolve_account_id(
            bucket,
            os.environ.get("GENTRX_AGENT_S3_ACCOUNT_ID", ""),
            os.environ.get("GENTRX_AGENT_S3_PROVIDER", ""),
        )

        # Read-only creds (committed) — fall back to write creds for local/MinIO
        read_access = (
            os.environ.get("GENTRX_AGENT_S3_READ_ACCESS_KEY", "")
            or os.environ.get("GENTRX_AGENT_S3_ACCESS_KEY", "")
        )
        read_secret = (
            os.environ.get("GENTRX_AGENT_S3_READ_SECRET_KEY", "")
            or os.environ.get("GENTRX_AGENT_S3_SECRET_KEY", "")
        )

        if not read_access or not read_secret:
            return None

        return cls(
            account_id=account_id,
            access_key_id=read_access,
            secret_access_key=read_secret,
            _bucket_override=bucket if bucket != account_id else None,
        )


class GenTRXChain:
    """Read/write GenTRX bucket commitments on the bittensor chain.

    The Commitments pallet stores one string per (netuid, hotkey).
    Miners call commit_bucket() to publish their bucket read credentials.
    The validator calls get_miner_buckets() to retrieve all of them.
    """

    def __init__(self, subtensor, netuid: int, metagraph) -> None:
        self.subtensor = subtensor
        self.netuid = netuid
        self.metagraph = metagraph
        # Optional endpoint override for all buckets — useful for MinIO localnet
        # where on-chain commitments don't include a usable endpoint URL.
        self._endpoint_override: str | None = None

    def commit_bucket(self, wallet, bucket: BucketInfo) -> None:
        """Commit miner's S3 read credentials on-chain.

        Checks existing commitment first to avoid redundant chain writes.
        """
        new_data = bucket.to_commitment()

        # Check if commitment already matches
        try:
            uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            existing = self.subtensor.get_commitment(self.netuid, uid)
            if existing == new_data:
                logger.debug("Bucket commitment unchanged, skipping")
                return
        except Exception:
            pass  # No existing commitment or lookup failed — commit anyway

        # Bittensor v9.4: subtensor.commit() renamed to set_commitment()
        if hasattr(self.subtensor, "set_commitment"):
            self.subtensor.set_commitment(wallet, self.netuid, new_data)
        else:
            self.subtensor.commit(wallet, self.netuid, new_data)  # legacy
        logger.info(
            "Committed bucket credentials to chain for hotkey %s",
            wallet.hotkey.ss58_address,
        )

    def get_bucket(self, uid: int) -> BucketInfo | None:
        """Read a single miner's bucket commitment from chain."""
        try:
            data = self.subtensor.get_commitment(self.netuid, uid)
        except Exception as exc:
            logger.debug("No commitment for UID %d: %s", uid, exc)
            return None

        if not data or len(data) != COMMITMENT_LEN:
            return None

        try:
            bi = BucketInfo.from_commitment(data)
            if self._endpoint_override:
                bi._endpoint_override = self._endpoint_override
            return bi
        except ValueError as exc:
            logger.warning("Invalid commitment for UID %d: %s", uid, exc)
            return None

    async def get_miner_buckets(
        self, block: int | None = None
    ) -> dict[int, BucketInfo]:
        """Read all miner bucket commitments from chain.

        Returns mapping of uid → BucketInfo for all miners with valid
        128-char commitments.
        """
        # Refresh the metagraph so re-registered uids' NEW hotkeys are visible
        # below. Without this the gradient server holds whatever instance was
        # built at startup, and any uid that was deregistered + re-registered
        # to a different hotkey gets dropped by the
        # `decoded_ss58 not in hotkey_to_uid` filter — the new commitment is
        # unmatchable, _miner_buckets[uid] stays stale (or empty), and every
        # subsequent gradient_get for that uid hits NoSuchBucket. Refresh is
        # already rate-limited upstream by _miner_buckets_refresh_s in the
        # gradient server (30s), so the per-call sync cost is bounded.
        try:
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as exc:
            logger.warning("metagraph sync failed in get_miner_buckets: %s", exc)
        try:
            from bittensor.utils import SS58_FORMAT, ss58_encode

            substrate = self.subtensor.substrate
            query_result = substrate.query_map(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.netuid],
                block_hash=(None if block is None else substrate.get_block_hash(block)),
            )

            hotkey_to_uid = dict(zip(self.metagraph.hotkeys, self.metagraph.uids))
            buckets: dict[int, BucketInfo] = {}

            for key, value in query_result:
                try:
                    # substrate-interface used to wrap the storage key in a
                    # tuple of bytes; newer versions yield the ss58 string
                    # directly. Handle both.
                    if isinstance(key, str):
                        decoded_ss58 = key
                    else:
                        raw_key = key[0]
                        if isinstance(raw_key, str):
                            raw_key = bytes.fromhex(raw_key.removeprefix("0x"))
                        else:
                            raw_key = bytes(raw_key)
                        decoded_ss58 = ss58_encode(raw_key, SS58_FORMAT)
                    info = value["info"] if isinstance(value, dict) else value.value["info"]
                    field = info["fields"][0]
                    if isinstance(field, dict):
                        payload = next(iter(field.values()))
                    else:
                        payload = field[0]
                        if isinstance(payload, dict):
                            payload = next(iter(payload.values()))[0]
                    if isinstance(payload, str):
                        raw = bytes.fromhex(payload.removeprefix("0x")).decode("ascii")
                    elif isinstance(payload, (bytes, bytearray)):
                        raw = bytes(payload).decode("ascii")
                    else:
                        raw = bytes(payload).decode("ascii")
                except Exception as exc:
                    logger.debug("Failed to decode commitment: %s", exc)
                    continue

                if decoded_ss58 not in hotkey_to_uid:
                    continue

                uid = hotkey_to_uid[decoded_ss58]
                if len(raw) != COMMITMENT_LEN:
                    logger.debug(
                        "UID %d commitment length %d (expected %d), skipping",
                        uid,
                        len(raw),
                        COMMITMENT_LEN,
                    )
                    continue

                try:
                    bi = BucketInfo.from_commitment(raw)
                    if self._endpoint_override:
                        bi._endpoint_override = self._endpoint_override
                    buckets[uid] = bi
                except ValueError as exc:
                    logger.warning("UID %d invalid commitment: %s", uid, exc)

            # Only log at INFO when we actually got something — empty results
            # are extremely common during bootstrap / between miner restarts
            # and would otherwise spam the log every refresh tick.
            if buckets:
                logger.info(
                    "Retrieved %d miner bucket commitments from chain", len(buckets)
                )
            else:
                logger.debug("Retrieved 0 miner bucket commitments from chain")
            return buckets

        except Exception as exc:
            logger.error("Failed to query chain commitments: %s", exc)
            return {}


class LocalBucketConfig:
    """Config-file-based bucket discovery for local proxy test (no chain).

    Loads miner bucket info from a JSON file or dict. Same interface as
    GenTRXChain so GradientAggregator can use either.

    JSON format:
        {
          "0": {"endpoint_url": "http://localhost:9000", "bucket": "agent-0",
                "access_key": "minioadmin", "secret_key": "minioadmin"},
          "1": { ... }
        }
    """

    def __init__(self, config: dict[int, BucketInfo] | str) -> None:
        if isinstance(config, str):
            # Load from JSON file path
            import json
            from pathlib import Path

            raw = json.loads(Path(config).read_text())
            self._buckets = {}
            for uid_str, info in raw.items():
                self._buckets[int(uid_str)] = BucketInfo(
                    account_id=info.get("bucket", info.get("account_id", "")),
                    access_key_id=info.get("access_key", info.get("access_key_id", "")),
                    secret_access_key=info.get(
                        "secret_key", info.get("secret_access_key", "")
                    ),
                )
                # Override endpoint_url and bucket name for local dev
                if "endpoint_url" in info:
                    self._buckets[int(uid_str)]._endpoint_override = info[
                        "endpoint_url"
                    ]
                if "bucket" in info:
                    self._buckets[int(uid_str)]._bucket_override = info["bucket"]
        else:
            self._buckets = config

    async def get_miner_buckets(
        self, block: int | None = None
    ) -> dict[int, BucketInfo]:
        """Return pre-configured miner buckets (no chain query)."""
        return dict(self._buckets)

    def commit_bucket(self, wallet, bucket: BucketInfo) -> None:
        """No-op for local testing."""
        pass
