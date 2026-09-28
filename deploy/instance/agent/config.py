import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# Appliance language -> its own database. An appliance serves one language per database: switching
# APPLIANCE_LANGUAGE on an existing database fails OnyxEngine.start() (IndexLanguageMismatchError).
DEFAULT_DATABASES = {"sr": "onyx_sr", "sl": "onyx_sl"}


@dataclass(frozen=True)
class AgentSettings:
    secret: str
    onyx_root: Path
    upload_root: Path
    display: str = ":1"
    state_file: Path = Path("/var/run/onyx-agent/app.json")
    log_dir: Path = Path("/workspace/logs")
    databases: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_DATABASES))

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
        )
