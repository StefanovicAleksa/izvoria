from pathlib import Path
from pydantic import BaseModel, ConfigDict

class ScannedFile(BaseModel):
    path: Path
    file_name: str
    extension: str
    size_bytes: int

    model_config = ConfigDict(frozen=True)