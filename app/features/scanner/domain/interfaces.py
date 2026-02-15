from abc import ABC, abstractmethod
from typing import Generator
from pathlib import Path
from app.features.scanner.domain.entities import ScannedFile

class IScaner(ABC):
    @abstractmethod
    def scan(self, root_path: str | Path) -> Generator[ScannedFile, None, None]:
        pass