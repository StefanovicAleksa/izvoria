"""
Brings this GPU box to the current version of the app before its services start. onstart.sh runs it at every start.

    python3 web_deploy/provision/self_update.py      # from /workspace/onyx_v3; log: /workspace/logs/self_update.log

1. Asks the gateway which version is current: `GET <RELEASE_URL>/box-release/<instance>`, signed with
   WEB_API_SECRET (web_deploy/gateway/releases.py).
2. When this copy (its RELEASE file) is another version: downloads the package, checks its sha256, and puts it in
   place of the code, keeping what the box built or downloaded itself (release.BOX_KEEP). When the files that
   provision.sh depends on changed (their digest differs from the one this box last provisioned with), runs it.
3. Reports the version it now holds, so the gateway knows this box is up to date.

Exit status: 0 to start the services (also when the gateway can't be reached or the package can't be fetched: the box
then runs the code it has, and the gateway, which knows its version, tries it last); 10 when the gateway has this box
under maintenance (it was started only to be updated: onstart.sh then starts no service, and the gateway stops it);
1 when applying an update failed half-way (onstart.sh starts nothing rather than half-updated code). A failure is
reported to the gateway (`/failed`), so a box started only to be updated is stopped at once.

Standard library only, and run with any Python 3.10+: it may be replacing the code the venv runs.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from web_deploy.common.release import BOX_KEEP, read_release  # noqa: E402
from web_deploy.common.signing import sign_service  # noqa: E402

ENV_FILE = Path(os.environ.get("BOX_ENV_FILE", "/workspace/web/api.env"))
DIGEST_FILE = Path(os.environ.get("BOX_PROVISION_DIGEST", "/workspace/web/provision_digest"))
SCRATCH = Path(os.environ.get("BOX_UPDATE_SCRATCH", "/workspace/tmp/self-update"))
MAINTENANCE = 10
FAILED = 1
TIMEOUT = 60


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} self_update: {message}", flush=True)


def read_env(path: Path) -> Dict[str, str]:
    values = {}
    try:
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def instance_id(env: Dict[str, str], environ=os.environ, proc_environ: Path = Path("/proc/1/environ")) -> Optional[int]:
    """This box's Vast instance id: from api.env, or the variables Vast sets in every container."""
    candidates = [env.get("BOX_INSTANCE_ID"), environ.get("CONTAINER_ID"), environ.get("VAST_CONTAINERLABEL")]
    try:
        for item in proc_environ.read_bytes().split(b"\0"):
            key, _, value = item.decode(errors="replace").partition("=")
            if key in ("CONTAINER_ID", "VAST_CONTAINERLABEL"):
                candidates.append(value)
    except OSError:
        pass
    for value in candidates:
        digits = (value or "").strip().removeprefix("C.")
        if digits.isdigit():
            return int(digits)
    return None


def gateway_url(env: Dict[str, str]) -> str:
    """The gateway's public address: RELEASE_URL (update_fleet.py writes the gateway's PUBLIC_URL there), else
    https://VPS_HOST, which works only when VPS_HOST is the domain its certificate names."""
    if env.get("RELEASE_URL"):
        return env["RELEASE_URL"].rstrip("/")
    return f"https://{env['VPS_HOST']}" if env.get("VPS_HOST") else ""


class Gateway:
    def __init__(self, base_url: str, secret: str, instance: int, opener=urllib.request.urlopen) -> None:
        self._base = base_url.rstrip("/")
        self._secret = secret
        self._instance = instance
        self._open = opener

    def _request(self, method: str, suffix: str = ""):
        path = f"/box-release/{self._instance}{suffix}"
        request = urllib.request.Request(self._base + path, method=method, data=b"" if method == "POST" else None,
                                         headers=sign_service(self._secret, method, path))
        return self._open(request, timeout=TIMEOUT)

    def current(self) -> dict:
        with self._request("GET") as response:
            return json.loads(response.read())

    def download(self, target: Path) -> str:
        """Saves the package to `target`; returns its sha256."""
        digest = hashlib.sha256()
        with self._request("GET", "/package") as response, open(target, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        return digest.hexdigest()

    def applied(self, release: str) -> None:
        with self._request("POST", f"/applied/{release}") as response:
            response.read()

    def failed(self) -> None:
        """Tells the gateway the update failed, so a box started only for it is stopped at once."""
        try:
            with self._request("POST", "/failed") as response:
                response.read()
        except (urllib.error.URLError, OSError) as exc:
            log(f"couldn't report the failure: {exc}")


def apply(package: Path, root: Path) -> None:
    """Puts the package's files in place of `root`'s code, keeping BOX_KEEP."""
    unpacked = package.with_suffix(".d")
    shutil.rmtree(unpacked, ignore_errors=True)
    unpacked.mkdir(parents=True)
    with tarfile.open(package) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(unpacked, filter="data")
        else:  # Python older than 3.10.12: the package comes from our own gateway over a signed request
            tar.extractall(unpacked)
    excludes = [arg for keep in BOX_KEEP for arg in ("--exclude", keep)]
    subprocess.run(["rsync", "-rlpt", "--delete", *excludes, f"{unpacked}/", f"{root}/"], check=True)
    shutil.rmtree(unpacked, ignore_errors=True)


def provision(root: Path) -> None:
    log("the box setup changed: running provision.sh (minutes)")
    with open("/workspace/logs/provision.log", "a") as out:
        subprocess.run(["bash", str(root / "web_deploy/provision/provision.sh")], cwd=root, check=True,
                       stdout=out, stderr=subprocess.STDOUT)


def update(gateway: Gateway, root: Path = ROOT) -> int:
    try:
        meta = gateway.current()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log(f"the gateway can't be reached ({exc}); starting with the code this box has, {read_release(root)}")
        return 0
    release, maintenance = meta["release"], bool(meta.get("maintenance"))
    held = read_release(root)
    package = SCRATCH / "release.tar.gz"
    if held != release:
        log(f"updating {held} -> {release}")
        try:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            if gateway.download(package) != meta["sha256"]:
                raise RuntimeError("the package's sha256 doesn't match")
        except Exception as exc:
            # Nothing was changed yet: the box can still run the code it has (the gateway tries it last)
            log(f"couldn't fetch {release} ({exc}); starting with {held}")
            package.unlink(missing_ok=True)
            if maintenance:
                gateway.failed()
                return MAINTENANCE
            return 0
    try:
        if held != release:
            apply(package, root)
            package.unlink(missing_ok=True)
            if read_release(root) != release:
                raise RuntimeError(f"after the update this copy says {read_release(root)}")
        stored = DIGEST_FILE.read_text().strip() if DIGEST_FILE.exists() else ""
        if held != release and stored != meta.get("provision_digest"):
            provision(root)
        if meta.get("provision_digest"):
            DIGEST_FILE.write_text(meta["provision_digest"] + "\n")
        gateway.applied(release)
        log(f"holds {release}" + (" (started for this update only)" if maintenance else ""))
    except Exception as exc:
        log(f"UPDATE FAILED: {exc}")
        gateway.failed()
        return FAILED
    return MAINTENANCE if maintenance else 0


def main() -> int:
    env = read_env(ENV_FILE)
    secret, base_url = env.get("WEB_API_SECRET", ""), gateway_url(env)
    instance = instance_id(env)
    if len(secret) < 32 or not base_url or instance is None:
        log(f"not set up for updates (WEB_API_SECRET, RELEASE_URL or instance id missing: instance={instance}); "
            f"starting as is")
        return 0
    return update(Gateway(base_url, secret, instance))


if __name__ == "__main__":
    sys.exit(main())
