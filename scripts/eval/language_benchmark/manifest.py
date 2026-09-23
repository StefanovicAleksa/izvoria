# Path: scripts/eval/language_benchmark/manifest.py

import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_PATH = Path(__file__).with_name("sources.toml")


class Lecture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    youtube_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    title: str
    topic: str
    max_minutes: int = Field(gt=0, le=240)

    @property
    def stem(self) -> str:
        return f"lecture_{self.youtube_id}"


class LanguageSources(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fleurs_config: str
    fleurs_minutes: int = Field(gt=0)
    parlaspeech_shard: Optional[str] = None
    parlaspeech_minutes: int = 0
    lectures: List[Lecture] = Field(default_factory=list)


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, LanguageSources]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {code: LanguageSources.model_validate(table) for code, table in data.items()}
