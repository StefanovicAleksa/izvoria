# Canvas: engine_enums_v1
# Path: app/engine/domain/enums.py

from enum import Enum

class EngineState(str, Enum):
    OFFLINE = "offline"
    BOOTING = "booting"
    ONLINE = "online"
    INGESTING = "ingesting"  # Hardware locked, VRAM prioritized for batch processing
    ERROR = "error"