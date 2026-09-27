# Path: app/features/document_conversion/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

from app.features.document_conversion.domain.entities import ConvertedDocument, OcrPage


class IPdfDocument(ABC):
    """An open PDF. Close it (or use it as a context manager) to release the file."""

    @abstractmethod
    def page_count(self) -> int:
        pass

    @abstractmethod
    def page_as_pdf(self, index: int) -> bytes:
        """Page `index` (from 0) as a one-page PDF, so a converter sees exactly that page."""
        pass

    @abstractmethod
    def image_coverage(self, index: int) -> float:
        """Fraction (0..1) of page `index`'s area covered by images: close to 1 for a scanned page."""
        pass

    @abstractmethod
    def render_page(self, index: int, dpi: int) -> Image.Image:
        """Page `index` (from 0) rendered to an RGB image at `dpi`, for OCR."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> "IPdfDocument":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class IPdfReader(ABC):
    @abstractmethod
    def open(self, path: str) -> IPdfDocument:
        """Opens a PDF; raises for a missing, corrupt or password-protected file."""
        pass


class IMarkdownConverter(ABC):
    @abstractmethod
    def convert(self, data: bytes, extension: str) -> str:
        """Converts a file's bytes to Markdown. Never reads a path or a URL."""
        pass


class ITextDecoder(ABC):
    @abstractmethod
    def decode(self, data: bytes) -> str:
        """A plain-text file's bytes as text, whatever its encoding. Never adds, drops or joins a line."""
        pass


class IRtfReader(ABC):
    @abstractmethod
    def to_text(self, data: bytes) -> str:
        """An RTF file's bytes as plain text, one paragraph per line."""
        pass


class IPageLayoutExtractor(ABC):
    @abstractmethod
    def extract(self, page_pdf: bytes) -> Optional[str]:
        """
        A one-page PDF's text with its tables as Markdown tables, or None when the page has no table (its
        converter's text is then used unchanged).
        """
        pass


class IOcrEngine(ABC):
    """Reads text from a page image. Runs on the CPU; models load on first use and stay until `close()`."""

    @abstractmethod
    def recognize(self, image: Image.Image) -> OcrPage:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class IConvertedDocumentRepository(ABC):
    @abstractmethod
    def save(self, document: ConvertedDocument) -> ConvertedDocument:
        pass

    @abstractmethod
    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ConvertedDocument]:
        pass
