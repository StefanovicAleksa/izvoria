"""
The box API's own settings, from the environment (provision writes them to /workspace/web/api.env). The appliance's
settings (models, Postgres, the hosted residency flags) are read by app.core.config from the same environment.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

API_PORT = 8095


@dataclass(frozen=True)
class ApiSettings:
    secret: str
    # The box's scratch copy of the tenants (the same absolute path as the VPS's, which holds the data)
    tenants_root: Path
    control_database_url: str
    admin_database: str = "postgres"
    live_workers: int = 8
    report_workers: int = 2
    firm_pool_size: int = 3
    firm_pool_overflow: int = 3
    # Where the ingestion worker marks jobs whose results are not on the VPS yet (STAGING_STATE_DIR); None when the
    # box and the data share one disk
    staging_state_dir: Optional[Path] = None
    # YouTube links on the Ingest page (web only: the appliance never reaches the internet)
    link_ingest_enabled: bool = True
    link_max_videos: int = 200
    link_max_height: int = 720

    @classmethod
    def from_env(cls) -> "ApiSettings":
        secret = os.environ.get("WEB_API_SECRET", "")
        if len(secret) < 32:
            raise RuntimeError("WEB_API_SECRET must be set (32+ characters), the same value as on the gateway.")
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "")
        server = os.environ.get("POSTGRES_SERVER", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        return cls(
            secret=secret,
            tenants_root=Path(os.environ.get("TENANTS_ROOT", "/workspace/tenants")),
            control_database_url=os.environ.get(
                "CONTROL_DATABASE_URL", f"postgresql+psycopg2://{user}:{password}@{server}:{port}/onyx_control"),
            admin_database=os.environ.get("POSTGRES_ADMIN_DB", "postgres"),
            live_workers=int(os.environ.get("WEB_LIVE_WORKERS", "8")),
            report_workers=int(os.environ.get("REPORT_MAX_CONCURRENT", "2")),
            firm_pool_size=int(os.environ.get("FIRM_DB_POOL_SIZE", "3")),
            firm_pool_overflow=int(os.environ.get("FIRM_DB_POOL_OVERFLOW", "3")),
            link_ingest_enabled=os.environ.get("LINK_INGEST_ENABLED", "true").strip().lower() in ("1", "true", "yes"),
            link_max_videos=int(os.environ.get("LINK_MAX_VIDEOS", "200")),
            link_max_height=int(os.environ.get("LINK_MAX_HEIGHT", "720")),
            staging_state_dir=(Path(os.environ.get("STAGING_STATE_DIR", "/workspace/staging-state"))
                               if os.environ.get("BOX_GATEWAY_URL") else None),
        )
