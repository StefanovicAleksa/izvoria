"""
One ingestion job's files on the GPU box, for as long as the job runs, and nothing at rest.

Every firm's vault and uploads live on the Hetzner VPS (docs/web_deploy.md, "Where data lives"), and the box has no
key that can reach them. It asks the VPS to move a job's files instead (gateway/box_internal.py), and the VPS does,
through a port the box forwards to its own sshd (gateway/staging_server.py): the job's batch and the vault files of
sources with the same content in, the firm's scratch vault back out. The scratch sits at the SAME absolute path as on
the VPS (TENANTS_ROOT, /workspace/tenants), because the database stores absolute file paths.

Around the ingestion child (Supervisor._run_job):
1. `stage_in`: the VPS sends the files; the box writes a marker.
2. the ingestion child writes the vault in scratch, and its rows into the VPS database.
3. `stage_out`: the VPS takes the vault back; then the scratch is wiped and the marker removed.

A marker lives until the results are on the VPS, so a crash, a host stop or a failed transfer leaves the job
recoverable: `recover()` asks for the transfer again at the next start, and the API's /health reports
`staging_pending` meanwhile, which keeps the gateway's idle stop away.

`ensure_access()` installs the VPS's staging key for root, so the VPS can reach the scratch. It runs before every
transfer, not once per start: Vast rewrites authorized_keys when it restarts the container, which once dropped the key
while the worker kept running, and every transfer after it was refused.
"""

import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from web_deploy.common.signing import sign_service
from web_deploy.common.tenancy import firm_root

logger = logging.getLogger(__name__)

AUTHORIZED_KEYS = Path("/root/.ssh/authorized_keys")
KEY_MARK = "izvoria-staging"

# (method, path) -> parsed JSON answer; raises StagingError
Caller = Callable[[str, str], dict]


class StagingError(RuntimeError):
    pass


@dataclass(frozen=True)
class StagingSettings:
    """From api.env. Without BOX_GATEWAY_URL the box and the data share one disk (dev hardware): staging is off."""
    gateway_url: str
    tenants_root: Path
    state_dir: Path
    secret: str = ""
    # The owner's YouTube cookies, sent by the VPS with a YouTube job (gateway/youtube_cookies.py)
    youtube_cookies: Path = Path("/workspace/web/youtube_cookies.txt")

    @property
    def enabled(self) -> bool:
        return bool(self.gateway_url)

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "StagingSettings":
        env = dict(os.environ if env is None else env)
        return cls(
            gateway_url=env.get("BOX_GATEWAY_URL", "").strip().rstrip("/"),
            tenants_root=Path(env.get("TENANTS_ROOT", "/workspace/tenants")),
            state_dir=Path(env.get("STAGING_STATE_DIR", "/workspace/staging-state")),
            secret=env.get("WEB_API_SECRET", ""),
            youtube_cookies=Path(env.get("YOUTUBE_COOKIES_FILE", "/workspace/web/youtube_cookies.txt")),
        )


def pending_markers(state_dir: Path) -> List[Path]:
    """Jobs whose results are not on the VPS yet. The API's /health reads this."""
    return sorted(state_dir.glob("*.json")) if state_dir.is_dir() else []


class Stager:
    def __init__(self, settings: StagingSettings, call: Optional[Caller] = None,
                 authorized_keys: Optional[Path] = None) -> None:
        self._settings = settings
        self._call = call or self._http
        self._authorized_keys = authorized_keys or AUTHORIZED_KEYS

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def has_pending(self) -> bool:
        return self.enabled and bool(pending_markers(self._settings.state_dir))

    # ---------- at start ----------

    def ensure_access(self) -> None:
        """
        Lets the VPS's staging key in as root. Called before every transfer: Vast may rewrite authorized_keys at any
        container restart, and a missing key makes every transfer fail with "Permission denied (publickey)".
        """
        if not self.enabled:
            return
        key = str(self._call("GET", "/internal/box/staging-key").get("key", "")).strip().split()
        if len(key) < 2 or key[0] != "ssh-ed25519":
            raise StagingError("The gateway sent no staging key.")
        line = f"restrict {key[0]} {key[1]} {KEY_MARK}"
        path = self._authorized_keys
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = path.read_text().splitlines() if path.exists() else []
            if line in current:
                return
            kept = [l for l in current if not l.endswith(KEY_MARK)]
            path.write_text("\n".join([*kept, line]) + "\n")
            path.chmod(0o600)
        except OSError as exc:
            raise StagingError(f"Could not let the VPS's staging key in: {exc}") from exc

    # ---------- a job ----------

    def stage_in(self, job_id: str, firm: str, batch_dir: Optional[str]) -> None:
        """The VPS sends the job's files (its batch, unless downloaded here, and matching vault files)."""
        if not self.enabled:
            return
        self._write_marker(job_id, {"firm": firm})
        if batch_dir:
            Path(batch_dir).mkdir(parents=True, exist_ok=True)
        # The glossary comes only from the VPS: one an earlier job left here may since have been cleared there
        _remove_tree(firm_root(self._settings.tenants_root, firm) / "glossaries")
        self.ensure_access()
        answer = self._call("POST", f"/internal/box/stage-in/{job_id}")
        logger.info(f"Job {job_id}: the VPS sent {answer.get('batch_files', 0)} batch file(s) and "
                    f"{answer.get('vault_files', 0)} vault file(s)")

    def stage_out(self, job_id: str) -> None:
        """The VPS takes the results back; then the scratch is wiped. Raises StagingError and keeps it on failure."""
        if not self.enabled:
            return
        marker = self._read_marker(job_id)
        if marker is None:
            return
        self.ensure_access()
        self._call("POST", f"/internal/box/stage-out/{job_id}")
        self.wipe(marker["firm"])
        (self._settings.state_dir / f"{job_id}.json").unlink(missing_ok=True)
        logger.info(f"Job {job_id}: results are on the VPS and the box's copy is wiped")

    def abandon(self, job_id: str) -> None:
        """
        Drops a job whose files never arrived: no ingestion ran, so the scratch holds nothing the VPS lacks and there
        is nothing to push back. Without this its marker would stay, and the worker refuses new work while a push is
        pending: one failed fetch would stop ingestion for every firm.
        """
        if not self.enabled:
            return
        marker = self._read_marker(job_id)
        if marker is not None:
            self.wipe(marker["firm"])
        (self._settings.state_dir / f"{job_id}.json").unlink(missing_ok=True)

    def recover(self) -> int:
        """At start: has the VPS take back what an interrupted job left, then wipes all scratch. Returns what's left."""
        if not self.enabled:
            return 0
        for marker in pending_markers(self._settings.state_dir):
            try:
                self.stage_out(marker.stem)
            except StagingError as exc:
                logger.error(f"Could not hand job {marker.stem}'s results to the VPS yet: {exc}")
        left = pending_markers(self._settings.state_dir)
        if not left:
            self.wipe_all()
        return len(left)

    # ---------- scratch ----------

    def wipe(self, firm: str) -> None:
        """
        A firm's scratch vault, uploads and glossary (its client names). Reports are kept in the database only
        (REPORT_FILES_ENABLED=false).
        """
        root = firm_root(self._settings.tenants_root, firm)
        _remove_tree(root / "glossaries")
        for part in ("vault", "uploads"):
            _remove_tree(root / part)
            (root / part).mkdir(parents=True, exist_ok=True)

    def youtube_cookies_back(self, job_id: str, refused: bool) -> None:
        """
        After a download step: the VPS takes back the cookies yt-dlp refreshed (keeping the sign-in alive), or, when
        YouTube took the box for a bot, hears so and tells the owner. Never fails the job.
        """
        if not self.enabled:
            return
        try:
            if refused:
                self._call("POST", f"/internal/box/youtube-cookies/{job_id}/refused")
            elif self._settings.youtube_cookies.is_file():
                self._call("POST", f"/internal/box/youtube-cookies/{job_id}/refreshed")
        except StagingError as exc:
            logger.warning(f"Job {job_id}: could not hand the YouTube cookies' state to the VPS: {exc}")

    def drop_youtube_cookies(self) -> None:
        """The YouTube account's cookies, once the download step is over: the box keeps no credential at rest."""
        if self.enabled:
            self._settings.youtube_cookies.unlink(missing_ok=True)

    def wipe_all(self) -> None:
        self.drop_youtube_cookies()  # a restart in the middle of a download
        root = self._settings.tenants_root
        if not root.is_dir():
            return
        for firm in sorted(root.iterdir()):
            if firm.is_dir() and not firm.is_symlink():
                _remove_tree(firm / "glossaries")
                for part in ("vault", "uploads", "reports"):
                    _remove_tree(firm / part)
                    (firm / part).mkdir(parents=True, exist_ok=True)

    # ---------- plumbing ----------

    def _http(self, method: str, path: str) -> dict:
        request = urllib.request.Request(self._settings.gateway_url + path, method=method,
                                         data=b"{}" if method == "POST" else None,
                                         headers={**sign_service(self._settings.secret, method, path),
                                                  "Content-Type": "application/json"})
        try:
            # A transfer takes as long as the files do: no read timeout
            with urllib.request.urlopen(request, timeout=None) as response:
                return json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read().decode()).get("error") or exc.reason
            except Exception:
                message = exc.reason
            raise StagingError(f"{path}: {exc.code} {message}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise StagingError(f"{path}: the gateway could not be reached: {exc}") from exc

    def _write_marker(self, job_id: str, marker: dict) -> None:
        self._settings.state_dir.mkdir(parents=True, exist_ok=True)
        target = self._settings.state_dir / f"{job_id}.json"
        partial = target.with_suffix(".tmp")
        partial.write_text(json.dumps(marker))
        partial.replace(target)

    def _read_marker(self, job_id: str) -> Optional[dict]:
        try:
            return json.loads((self._settings.state_dir / f"{job_id}.json").read_text())
        except (OSError, ValueError):
            return None


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)
