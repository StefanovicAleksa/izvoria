import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# One database holds every language: each chunk is keyword-indexed with its own language's stemmer. Must match the
# packs in data/languages/ (and deploy/gateway/config.py's LANGUAGES).
DEFAULT_DATABASE = "onyx"
DEFAULT_LANGUAGES = ("en", "sr")


@dataclass(frozen=True)
class AgentSettings:
    secret: str
    onyx_root: Path
    upload_root: Path
    display: str = ":1"
    state_file: Path = Path("/var/run/onyx-agent/app.json")
    log_dir: Path = Path("/workspace/logs")
    database: str = DEFAULT_DATABASE
    languages: Tuple[str, ...] = DEFAULT_LANGUAGES
    # Where the appliance keeps every ingested file and clip (its FileManager's ONYX_STORAGE_PATH, relative to
    # onyx_root). The only folder a cited source may be served from. None means onyx_root / "onyx_data".
    vault_root: Optional[Path] = None
    # Browser-playable transcodes and cited-passage cuts, derived by the agent; safe to delete at any time.
    cache_root: Path = Path("/workspace/source-cache")
    # The address the desktop app reaches the agent on, from inside the box.
    offer_url: str = "http://127.0.0.1:8090/offers"

    @property
    def vault(self) -> Path:
        return self.vault_root if self.vault_root is not None else self.onyx_root / "onyx_data"

    @property
    def python(self) -> Path:
        return self.onyx_root / "venv" / "bin" / "python"

    @classmethod
    def from_env(cls) -> "AgentSettings":
        secret = os.environ.get("AGENT_SECRET", "")
        if len(secret) < 32:
            raise RuntimeError("AGENT_SECRET must be set to at least 32 characters.")
        onyx_root = Path(os.environ.get("ONYX_ROOT", "/workspace/onyx_v3"))
        return cls(
            secret=secret,
            onyx_root=onyx_root,
            upload_root=Path(os.environ.get("UPLOAD_ROOT", "/workspace/uploads")),
            display=os.environ.get("ONYX_DISPLAY", ":1"),
            state_file=Path(os.environ.get("AGENT_STATE_FILE", "/var/run/onyx-agent/app.json")),
            log_dir=Path(os.environ.get("AGENT_LOG_DIR", "/workspace/logs")),
            vault_root=Path(os.environ["ONYX_STORAGE_ROOT"]) if os.environ.get("ONYX_STORAGE_ROOT") else None,
            cache_root=Path(os.environ.get("SOURCE_CACHE_ROOT", "/workspace/source-cache")),
            offer_url=os.environ.get("SOURCE_OFFER_URL", "http://127.0.0.1:8090/offers"),
        )
