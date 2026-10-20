"""
Which version of the app a machine runs: the git commit it was deployed from.

`update_fleet.py` writes it to a `RELEASE` file at the top of every copy it deploys (the VPS's and each GPU box's),
and the box API reports it in /health, so the gateway can tell an up-to-date box from a stale one. A checkout that was
never deployed has no file and reports "unknown". Standard library only: the gateway and the box both import it.
"""

from pathlib import Path

RELEASE_FILE = "RELEASE"
UNKNOWN = "unknown"

# The repository root: web_deploy/common/release.py -> ../../
ROOT = Path(__file__).resolve().parents[2]


def read_release(root: Path = ROOT) -> str:
    """The deployed commit, or "unknown" when this copy wasn't deployed by update_fleet.py."""
    try:
        value = (root / RELEASE_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return UNKNOWN
    return value or UNKNOWN
