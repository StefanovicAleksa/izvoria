"""
Restores the firms' data from a backup onto the Hetzner VPS (docs/web_deploy.md, "Backups"): every firm database and
the ingestion queue, and every firm's vault.

  1. Install the gateway (install_gateway.sh) with the SAME /etc/onyx-web-gateway.env as before: the same database
     password, restic repository and password, and WEB_API_SECRET the GPU box knows.
  2. Install the backup SSH key the old VPS used (or add the new one's key to the Storage Box).
  3. systemctl stop onyx-web-gateway onyx-web-backup        (Postgres keeps running)
  4. cd /opt/onyx_v3 && set -a && . /etc/onyx-web-gateway.env && set +a
     /opt/onyx-web-gateway/bin/python -m web_deploy.backup.restore --list
     /opt/onyx-web-gateway/bin/python -m web_deploy.backup.restore --snapshot latest --yes
  5. systemctl start onyx-web-gateway onyx-web-backup

Snapshots are kept per series (BACKUP_HOST_NAME): onyx-data for the VPS's, onyx-box for those taken on the GPU box
before the data moved to the VPS. Set BACKUP_HOST_NAME=onyx-box to restore one of those. The stored file paths are
the same on both machines (/workspace/tenants), so nothing needs rewriting.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

from web_deploy.backup.config import BackupSettings
from web_deploy.backup.restic import Restic
from web_deploy.backup.runner import TAG


class Restorer:
    def __init__(self, settings: BackupSettings, restic: Optional[Restic] = None,
                 run: Callable = subprocess.run, log: Callable[[str], None] = print) -> None:
        self._settings = settings
        self._restic = restic or Restic(settings, run=run)
        self._run = run
        self._log = log

    def restore(self, snapshot: str, workdir: Path) -> List[str]:
        """Returns the databases restored."""
        self._log(f"Fetching snapshot {snapshot} into {workdir}…")
        self._restic.restore(snapshot, workdir, TAG)
        dumps = sorted((workdir / self._settings.staging.relative_to("/") / "databases").glob("*.dump"))
        if not dumps:
            raise RuntimeError("The snapshot holds no database dumps.")
        for dump in dumps:
            self.restore_database(dump.stem, dump)
        restored_tenants = workdir / self._settings.tenants_root.relative_to("/")
        if restored_tenants.is_dir():
            self._log(f"Restoring the vaults into {self._settings.tenants_root}…")
            for firm in sorted(restored_tenants.iterdir()):
                destination = self._settings.tenants_root / firm.name
                for part in ("vault", "reports"):
                    if (firm / part).is_dir():
                        shutil.rmtree(destination / part, ignore_errors=True)
                        destination.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(firm / part, destination / part, symlinks=True)
                (destination / "uploads").mkdir(parents=True, exist_ok=True)
        for extra in self._settings.extra_paths:
            tls = workdir / extra.relative_to("/") / "tls"
            if tls.is_dir():
                self._log(f"Restoring the box's TLS certificate into {extra / 'tls'}…")
                shutil.rmtree(extra / "tls", ignore_errors=True)
                shutil.copytree(tls, extra / "tls")
        return [d.stem for d in dumps]

    def restore_database(self, name: str, dump: Path) -> None:
        self._log(f"Restoring database {name}…")
        self._pg(["dropdb", "--if-exists", name])
        self._pg(["createdb", name])
        completed = self._pg(["pg_restore", "--no-owner", "-d", name, str(dump)], check=False)
        if completed.returncode != 0:
            # pg_restore exits 1 on harmless warnings too (an extension comment a non-superuser can't set); the
            # data is restored. Show them rather than fail the whole restore.
            self._log(f"  pg_restore reported for {name}: {(completed.stderr or '').strip()[-1500:]}")

    def _pg(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        pg = self._settings.postgres
        args = [command[0], "-h", pg["host"], "-p", pg["port"], "-U", pg["user"]]
        if command[0] in ("dropdb", "createdb"):
            args += ["--maintenance-db", "postgres"]
        completed = self._run([*args, *command[1:]], env=dict(os.environ, PGPASSWORD=pg["password"]),
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if check and completed.returncode != 0:
            raise RuntimeError(f"{command[0]} {command[-1]} failed: {(completed.stderr or '').strip()[-1000:]}")
        return completed


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Restore the firms' data from a backup (replaces it).")
    parser.add_argument("--list", action="store_true", help="list the snapshots and exit")
    parser.add_argument("--snapshot", default="latest")
    parser.add_argument("--yes", action="store_true", help="confirm: every firm database and vault is replaced")
    args = parser.parse_args(argv)

    settings = BackupSettings.from_env()
    if not settings.repository or not settings.password:
        print("error: RESTIC_REPOSITORY and RESTIC_PASSWORD must be set (load /etc/onyx-web-gateway.env).",
              file=sys.stderr)
        return 1
    restic = Restic(settings)
    if args.list:
        for snap in restic.snapshots(TAG):
            print(f"{snap.get('short_id')}  {snap.get('time', '')[:19]}  {', '.join(snap.get('paths', []))}")
        return 0
    if not args.yes:
        print("error: this replaces every firm database and vault here; pass --yes to go ahead.",
              file=sys.stderr)
        return 1
    workdir = Path(f"/workspace/restore-{time.strftime('%Y%m%d-%H%M%S')}")
    names = Restorer(settings, restic).restore(args.snapshot, workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    print(f"Restored {len(names)} database(s): {', '.join(names)}. Start the services again: "
          f"systemctl start onyx-web-gateway onyx-web-backup")
    return 0


if __name__ == "__main__":
    sys.exit(main())
