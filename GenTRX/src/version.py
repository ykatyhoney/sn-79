"""Training-regime versioning for checkpoint compatibility.

`TRAIN_REGIME_VERSION` gates whether a stored checkpoint may be resumed. The
gradient server refuses to resume (or sync) a checkpoint stamped below the
current regime and restarts under warmup instead, so a non-backwards-compatible
training change (loss regime, bin layout, data scale) never silently continues
on a model trained under the old rules. `TAOS_SPEC_VERSION` is stamped alongside
for provenance.
"""

from __future__ import annotations

from taos import __spec_version__ as TAOS_SPEC_VERSION

# Bump on any change that makes prior checkpoints incompatible to resume.
# Unstamped (pre-versioning) checkpoints read as regime 0, below this.
# 3: equal order-type loss weights + IID/held-out scoring regime (0.5.3).
TRAIN_REGIME_VERSION = 3


def checkpoint_stamp(label_smooth_sigma: float) -> dict:
    """Version fields embedded in every saved checkpoint and its latest.json."""
    return {
        "train_regime_version": TRAIN_REGIME_VERSION,
        "taos_spec_version": TAOS_SPEC_VERSION,
        "label_smooth_sigma": label_smooth_sigma,
    }
