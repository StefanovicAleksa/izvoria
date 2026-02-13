from abc import ABC, abstractmethod
from typing import Generator
from contextlib import contextmanager
from sqlalchemy.orm import Session


class IDatabaseManager(ABC):
    @abstractmethod
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope."""
        pass

    @abstractmethod
    def dispose(self) -> None:
        """Close connections."""
        pass