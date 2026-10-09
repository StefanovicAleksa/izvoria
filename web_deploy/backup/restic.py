"""
restic, driven as a subprocess: an encrypted, deduplicated snapshot store, here on a Hetzner Storage Box over SFTP.
Only changed blocks travel after the first run, so a nightly backup of a large vault costs minutes, not hours.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from web_deploy.backup.config import BackupSettings


class ResticError(RuntimeError):
    pass


def sftp_target(repository: str) -> Optional[str]:
    """`sftp:user@host:path` -> `user@host`; None for any other kind of repository."""
    if not repository.startswith("sftp:"):
        return None
    rest = repository[len("sftp:"):]
    if rest.startswith("//"):  # sftp://user@host:port//path
        rest = rest[2:].split("/", 1)[0]
        return rest.split(":", 1)[0]
    return rest.split(":", 1)[0]


class Restic:
    def __init__(self, settings: BackupSettings, run: Callable = subprocess.run) -> None:
        self._settings = settings
        self._run = run

    def base(self) -> List[str]:
        command = ["restic"]
        target = sftp_target(self._settings.repository)
        if target:
            # Storage Boxes serve SFTP on port 23; the key is the box's own, never a person's
            known = (f"-o UserKnownHostsFile={self._settings.known_hosts} " if self._settings.known_hosts
                     else "")
            command += ["-o", f"sftp.command=ssh -i {self._settings.ssh_key} -p {self._settings.ssh_port} "
                              f"-o StrictHostKeyChecking=accept-new {known}-o BatchMode=yes {target} -s sftp"]
        return command

    def env(self) -> dict:
        env = dict(os.environ)
        env.update({"RESTIC_REPOSITORY": self._settings.repository, "RESTIC_PASSWORD": self._settings.password})
        return env

    def __call__(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        completed = self._run([*self.base(), *args], env=self.env(), stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if check and completed.returncode != 0:
            raise ResticError(f"restic {args[0]} failed (exit {completed.returncode}): "
                              f"{(completed.stderr or '').strip()[-2000:]}")
        return completed

    def ensure_repository(self) -> None:
        """Creates the repository on first use; afterwards only checks that it opens with this password."""
        if self("cat", "config", check=False).returncode != 0:
            self("init")

    def unlock(self) -> None:
        # A backup killed by a box stop leaves its lock behind; a stale lock would refuse the next run
        self("unlock", check=False)

    def backup(self, paths: Sequence[Path], excludes: Sequence[str], tag: str) -> None:
        args = ["backup", "--host", self._settings.host, "--tag", tag]
        for pattern in excludes:
            args += ["--exclude", pattern]
        self(*args, *[str(p) for p in paths])

    def forget(self, tag: str) -> None:
        args = ["forget", "--host", self._settings.host, "--tag", tag, "--prune"]
        for unit, count in self._settings.keep:
            args += [f"--keep-{unit}", str(count)]
        self(*args)

    def snapshots(self, tag: str) -> list:
        out = self("snapshots", "--host", self._settings.host, "--tag", tag, "--json").stdout
        return json.loads(out or "[]")

    def restore(self, snapshot: str, target: Path, tag: str) -> None:
        self("restore", snapshot, "--host", self._settings.host, "--tag", tag, "--target", str(target))
