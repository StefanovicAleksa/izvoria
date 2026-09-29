import os
from dataclasses import dataclass
from typing import Dict

# The languages the demo offers: one per pack in data/languages/. They share one database, and the desktop app
# switches between them itself; the choice here is only the language it starts in and names the upload folder.
LANGUAGES: Dict[str, str] = {"sr": "Serbian", "en": "English"}

# Container ports the instance exposes (Vast maps each to a random host port, read back from the API).
AGENT_PORT = 8090
NOVNC_PORT = 6080


@dataclass(frozen=True)
class GatewaySettings:
    password: str
    session_secret: str
    agent_secret: str
    websockify_token: str
    vast_api_key: str = ""
    vast_instance_id: int = 0
    vast_api_url: str = "https://console.vast.ai"
    idle_minutes: float = 30.0
    # Wait this long in Vast's "scheduling" state before telling the tester the GPU is taken.
    scheduling_grace_seconds: float = 60.0
    # Local end-to-end runs point the gateway at a box on this host and never call Vast.
    fake_vast_host: str = ""
    https_only: bool = True

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        def required(name: str, min_length: int = 1) -> str:
            value = os.environ.get(name, "")
            if len(value) < min_length:
                raise RuntimeError(f"{name} must be set (at least {min_length} characters).")
            return value

        fake_host = os.environ.get("FAKE_VAST_HOST", "")
        return cls(
            password=required("GATEWAY_PASSWORD", 12),
            session_secret=required("SESSION_SECRET", 32),
            agent_secret=required("AGENT_SECRET", 32),
            websockify_token=required("WEBSOCKIFY_TOKEN", 32),
            vast_api_key="" if fake_host else required("VAST_API_KEY"),
            vast_instance_id=0 if fake_host else int(required("VAST_INSTANCE_ID")),
            vast_api_url=os.environ.get("VAST_API_URL", "https://console.vast.ai"),
            idle_minutes=float(os.environ.get("IDLE_MINUTES", "30")),
            fake_vast_host=fake_host,
            https_only=os.environ.get("GATEWAY_HTTPS_ONLY", "1") != "0",
        )
