"""
Backs up the gateway's own database (firms, accounts, password hashes, sign-up requests) to the same restic
repository as the GPU box, as its own snapshot series (host onyx-gateway, tag gateway). Run daily by the
onyx-web-gateway-backup systemd timer that install_gateway.sh installs; a no-op until the backup settings in
/etc/onyx-web-gateway.env are filled in.

SQLite's online backup API copies a consistent database while the gateway keeps running.

Restore (on a new VPS, gateway stopped):
  restic restore latest --host onyx-gateway --tag gateway --target /tmp/r
  cp /tmp/r/var/lib/onyx-web/backup/gateway.sqlite3 /var/lib/onyx-web/gateway.sqlite3; chown onyx: …
"""

import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from web_deploy.backup.config import BackupSettings
from web_deploy.backup.restic import Restic

logger = logging.getLogger(__name__)

TAG = "gateway"


def snapshot_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src, sqlite3.connect(tmp) as dst:
        src.backup(dst)
    os.replace(tmp, target)


def run(env: Optional[dict] = None, restic: Optional[Restic] = None) -> int:
    env = dict(os.environ if env is None else env)
    env.setdefault("BACKUP_HOST_NAME", "onyx-gateway")
    env.setdefault("BACKUP_SSH_KEY", "/etc/onyx-web/backup_ed25519")
    # The service user has no home: ssh's host key and restic's cache live with the gateway's state
    env.setdefault("BACKUP_KNOWN_HOSTS", "/var/lib/onyx-web/known_hosts")
    settings = BackupSettings.from_env(env)
    if not settings.configured:
        logger.info("Gateway backups are off (BACKUP_ENABLED, RESTIC_REPOSITORY, RESTIC_PASSWORD).")
        return 0
    database = Path(env.get("GATEWAY_DB", "/var/lib/onyx-web/gateway.sqlite3"))
    copy = Path(env.get("GATEWAY_BACKUP_DIR", str(database.parent / "backup"))) / database.name
    snapshot_database(database, copy)
    restic = restic or Restic(settings)
    restic.unlock()
    restic.ensure_repository()
    restic.backup([copy.parent], [], TAG)
    restic.forget(TAG)
    logger.info(f"Gateway database backed up ({copy.stat().st_size} bytes)")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return run()
    except Exception as exc:
        logger.error(f"Gateway backup failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
