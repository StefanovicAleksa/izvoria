from pathlib import Path
from pydantic import BaseModel, ConfigDict

class ScannedFile(BaseModel):
    path: Path
    name: str
    extension: str
    size_bytes: int

    model_config = ConfigDict(frozen=True)