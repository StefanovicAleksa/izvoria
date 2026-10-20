"""
Which version of the app a machine runs: the git commit it was deployed from.

`update_fleet.py` writes it to a `RELEASE` file at the top of every copy it deploys (the VPS's and each GPU box's),
and the box API reports it in /health, so the gateway can tell an up-to-date box from a stale one. A checkout that was
never deployed has no file and reports "unknown". Standard library only: the gateway and the box both import it.
"""

import hashlib
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


# What a box keeps when a release replaces its copy of the app: what it built or downloaded itself (rsync leaves
# excluded paths alone). Shared by update_fleet.py and the box's own self_update.py.
BOX_KEEP = ("/venv/", "/models/", "/onyx_data/", "/.env", "/.env.*", "__pycache__/", "/deploy/.venv/",
            "/web_deploy/.venv/", "/web_deploy/frontend/node_modules/", "/web_deploy/local/local.env",
            "/web_deploy/provision/prod/", "/input_video/")

# A change to these needs provision.sh again (packages, llama-server, supervisord), not just a restart
PROVISION_PATHS = ("requirements.txt", "requirements-nodeps.txt", "web_deploy/api/requirements.txt",
                   "web_deploy/provision/provision.sh", "web_deploy/provision/build_llama_server.sh",
                   "web_deploy/provision/llama_patches/", "web_deploy/provision/supervisord.conf")


def provision_digest(root: Path) -> str:
    """A fingerprint of the files PROVISION_PATHS names: when it changes, a box must run provision.sh again."""
    digest = hashlib.sha256()
    for entry in PROVISION_PATHS:
        path = root / entry
        files = sorted(p for p in path.rglob("*") if p.is_file()) if entry.endswith("/") else [path]
        for file in files:
            if file.is_file():
                digest.update(str(file.relative_to(root)).encode() + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()
