"""The gateway's settings, from /etc/onyx-web-gateway.env (see web_deploy/provision/gateway.env.example)."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from web_deploy.api.config import API_PORT
from web_deploy.gateway.mailer import MailSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewaySettings:
    api_secret: str
    database_path: Path
    static_dir: Path
    vast_api_key: str = ""
    vast_instance_id: int = 0
    idle_minutes: float = 30.0
    https_only: bool = True
    # The box API's self-signed certificate, for BOX_LINK=direct over Vast's public port. Without it the gateway
    # talks plain HTTP: through the tunnel (encrypted by SSH) or on dev hardware.
    box_cert_file: Optional[Path] = None
    # "tunnel": the box dials this VPS over SSH and forwards its API to 127.0.0.1:box_tunnel_port here; the box has
    # no public port. "direct": the API at Vast's public address (the layout from before the data moved here).
    box_link: str = "direct"
    box_tunnel_port: int = 18095
    # Whether the box's SSH keys are accepted on this VPS: "open" while the gateway has the box started, "closed"
    # otherwise (read by sshd's AuthorizedKeysCommand, izvoria-box-keys). None: not managed (local runs).
    box_access_file: Optional[Path] = None
    # Local end-to-end runs: treat this host as the "GPU box" instead of calling Vast
    fake_box_host: str = ""
    api_port: int = API_PORT
    scheduling_grace_seconds: float = 60.0
    # "vast": a rented box started on demand. "local": the API runs on this machine (dev hardware), always on.
    box_mode: str = "vast"
    # The one address everyone signs in at (https://onyx.example.com): what browsers must send as Origin, and the
    # link in emails. Empty on a LAN, where the address varies: the request's own scheme and host are trusted then.
    public_url: str = ""
    mail: MailSettings = field(default_factory=MailSettings)

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        secret = os.environ.get("WEB_API_SECRET", "")
        if len(secret) < 32:
            raise RuntimeError("WEB_API_SECRET must be set (32+ characters), the same value as on the GPU box.")
        public_url = os.environ.get("PUBLIC_URL", "").strip().rstrip("/")
        base_domain = os.environ.get("BASE_DOMAIN", "").strip().lower()
        if not public_url and base_domain and base_domain != "localhost":
            public_url = f"https://{base_domain}"  # env files from before PUBLIC_URL
        fake = os.environ.get("FAKE_BOX_HOST", "")
        key = os.environ.get("VAST_API_KEY", "")
        box_mode = os.environ.get("BOX_MODE", "vast")
        if box_mode not in ("vast", "local"):
            raise RuntimeError("BOX_MODE is 'vast' or 'local'.")
        https_only = os.environ.get("GATEWAY_HTTPS_ONLY", "1") != "0"
        # Production (a rented box, HTTPS) must name its address; a local run trusts whatever address it is opened at
        if box_mode == "vast" and https_only and not fake and not public_url:
            raise RuntimeError("PUBLIC_URL must be set (e.g. https://onyx.example.com): the address everyone signs in at.")
        if "FIRM_ROUTING" in os.environ:
            logger.warning("FIRM_ROUTING is no longer used: every firm signs in at one address. Remove it.")
        if box_mode == "vast" and not fake and not key:
            raise RuntimeError("Set VAST_API_KEY and VAST_INSTANCE_ID (or BOX_MODE=local on dev hardware).")
        cert = os.environ.get("BOX_CERT_FILE", "")
        box_link = os.environ.get("BOX_LINK", "tunnel" if box_mode == "vast" and not fake else "direct")
        if box_link not in ("tunnel", "direct"):
            raise RuntimeError("BOX_LINK is 'tunnel' or 'direct'.")
        access = os.environ.get("BOX_ACCESS_FILE", "/var/lib/onyx-web-box/access" if box_link == "tunnel" else "")
        return cls(
            api_secret=secret,
            database_path=Path(os.environ.get("GATEWAY_DB", "/var/lib/onyx-web/gateway.sqlite3")),
            static_dir=Path(os.environ.get("STATIC_DIR", str(Path(__file__).parent / "static"))),
            vast_api_key=key,
            vast_instance_id=int(os.environ.get("VAST_INSTANCE_ID", "0") or 0),
            idle_minutes=float(os.environ.get("IDLE_MINUTES", "30")),
            https_only=https_only,
            box_cert_file=Path(cert) if cert else None,
            box_link=box_link,
            box_tunnel_port=int(os.environ.get("BOX_TUNNEL_PORT", "18095")),
            box_access_file=Path(access) if access else None,
            fake_box_host=fake,
            api_port=int(os.environ.get("BOX_API_PORT", str(API_PORT))),
            box_mode=box_mode,
            public_url=public_url,
            mail=mail_settings_from_env(),
        )


def mail_settings_from_env() -> MailSettings:
    signup_to = os.environ.get("SIGNUP_TO", "").strip()
    return MailSettings(
        smtp_host=os.environ.get("SMTP_HOST", "").strip(),
        smtp_port=int(os.environ.get("SMTP_PORT", "587") or 587),
        smtp_user=os.environ.get("SMTP_USER", "").strip(),
        smtp_password=os.environ.get("SMTP_PASSWORD", ""),
        smtp_starttls=os.environ.get("SMTP_STARTTLS", "1") != "0",
        smtp_ssl=os.environ.get("SMTP_SSL", "0") == "1",
        mail_from=os.environ.get("MAIL_FROM", "").strip() or os.environ.get("SMTP_USER", "").strip(),
        signup_to=signup_to,
        support_to=os.environ.get("SUPPORT_TO", "").strip() or signup_to,
    )
