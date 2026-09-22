# Canvas: engine_enums_v1
# Path: app/engine/domain/enums.py

from enum import Enum

class EngineState(str, Enum):
    OFFLINE = "offline"
    BOOTING = "booting"
    ONLINE = "online"
    INGESTING = "ingesting"  # Hardware locked, VRAM prioritized for batch processing
    # A Deep Research Report run holds the GPU for tens of minutes. It gets its own state rather than
    # reusing INGESTING so a caller can tell a user what the appliance is actually busy with.
    RESEARCHING = "researching"
    ERROR = "error"