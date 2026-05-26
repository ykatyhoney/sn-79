"""Filesystem path resolution for GenTRX runtime artifacts.

All default output paths are resolved here so the rest of the code can
import `default_output_dir(...)` without each call site computing its
own `Path(__file__).resolve().parents[N]`.

Resolution order for the agent / miner output directory:

  1. `GENTRX_AGENT_OUTPUT_DIR` env var (must be absolute)
  2. `<repo>/agents/data/<uid>/` when a uid is supplied
  3. `<repo>/data/live/` (generic fallback for direct API usage)

`<repo>` is the directory containing this package, resolved from
`__file__` so the same default applies regardless of CWD at process
launch. Previously several call sites used `../../../agents/data/<uid>`
which only worked when launched from `taos/im/neurons/`.
"""

from __future__ import annotations

import os
from pathlib import Path

# GenTRX/src/util/paths.py -> parents[3] = repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

OUTPUT_DIR_ENV: str = "GENTRX_AGENT_OUTPUT_DIR"


def default_output_dir(uid: int | str | None = None) -> Path:
    """Resolve the default miner / agent output directory.

    `<output_dir>` holds parquets, gradient cache, downloaded checkpoints,
    pending uploads, and log files. Operators can override the default
    by setting `GENTRX_AGENT_OUTPUT_DIR` to an absolute path.
    """
    env = os.environ.get(OUTPUT_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    if uid is not None and str(uid) != "":
        return REPO_ROOT / "agents" / "data" / str(uid)
    return REPO_ROOT / "data" / "live"
