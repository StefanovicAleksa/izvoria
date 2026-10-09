"""Backup settings, from the same environment file as the rest of the box (/workspace/web/api.env)."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


def _flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class BackupSettings:
    enabled: bool
    # restic's own variables: an sftp repository (a Hetzner Storage Box) and the key that encrypts it
    repository: str
    password: str
    # The SSH key the box logs in to the Storage Box with, and its port (Storage Boxes serve SFTP on 23)
    ssh_key: Path
    ssh_port: int
    # Where ssh keeps the Storage Box's host key; None is ssh's own (~/.ssh), which a system user may not have
    known_hosts: Optional[Path]
    # This snapshot series' name: restic groups and prunes by host, and a Vast container's hostname changes
    host: str
    interval_hours: float
    keep: Tuple[Tuple[str, int], ...]
    tenants_root: Path
    staging: Path
    status_file: Path
    # Also kept: the box's settings and TLS key, so a restored box serves the certificate the gateway pins
    extra_paths: Tuple[Path, ...]
    postgres: Dict[str, str] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.repository) and bool(self.password)

    @classmethod
    def from_env(cls, env: Dict[str, str] = None) -> "BackupSettings":
        env = dict(os.environ if env is None else env)
        web = Path(env.get("WEB_DIR", "/workspace/web"))
        return cls(
            enabled=_flag(env.get("BACKUP_ENABLED", "false")),
            repository=env.get("RESTIC_REPOSITORY", ""),
            password=env.get("RESTIC_PASSWORD", ""),
            ssh_key=Path(env.get("BACKUP_SSH_KEY", str(web / "backup_ed25519"))),
            ssh_port=int(env.get("BACKUP_SSH_PORT", "23")),
            known_hosts=Path(env["BACKUP_KNOWN_HOSTS"]) if env.get("BACKUP_KNOWN_HOSTS") else None,
            host=env.get("BACKUP_HOST_NAME", "onyx-box"),
            interval_hours=float(env.get("BACKUP_INTERVAL_HOURS", "24")),
            keep=(("daily", int(env.get("BACKUP_KEEP_DAILY", "7"))),
                  ("weekly", int(env.get("BACKUP_KEEP_WEEKLY", "4"))),
                  ("monthly", int(env.get("BACKUP_KEEP_MONTHLY", "6")))),
            tenants_root=Path(env.get("TENANTS_ROOT", "/workspace/tenants")),
            staging=Path(env.get("BACKUP_STAGING_DIR", "/workspace/backup-staging")),
            status_file=Path(env.get("BACKUP_STATUS_FILE", str(web / "backup-status.json"))),
            extra_paths=tuple(Path(p) for p in env.get("BACKUP_EXTRA_PATHS", str(web)).split(":") if p),
            postgres={
                "host": env.get("POSTGRES_SERVER", "127.0.0.1"),
                "port": env.get("POSTGRES_PORT", "5432"),
                "user": env.get("POSTGRES_USER", "onyx"),
                "password": env.get("POSTGRES_PASSWORD", ""),
            },
        )
