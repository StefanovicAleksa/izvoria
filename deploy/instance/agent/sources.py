"""
Cited sources offered to the tester's browser.

The desktop app, asked to open a citation on the hosted box, posts the file's path and anchor here instead of
handing it to the box's own player. The agent checks the path is a file in the appliance's vault, and files it
under a random offer id. The browser only ever sees that id: no path it sends is ever joined onto the filesystem.

Pure: no Starlette, no subprocesses, so the boundary that matters is unit-tested on its own.
"""

import asyncio
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from deploy.instance.agent.uploads import ALLOWED_SUFFIXES

MAX_OFFERS = 100
OFFER_TTL_SECONDS = 12 * 3600
# Seconds of context either side of a cited span in a recording's cut, so the passage doesn't start mid-word.
EXCERPT_PADDING_SECONDS = 5.0
MAX_TEXT_BYTES = 5 * 1024 * 1024
MAX_EXCERPT_CHARS = 20_000

OFFER_ID = re.compile(r"^[A-Za-z0-9_-]{22}$")


class SourceRejected(ValueError):
    pass


class SourceKind(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    PDF = "pdf"
    TEXT = "text"          # TXT / Markdown: shown with its cited lines highlighted
    DOCUMENT = "document"  # DOCX / EPUB / RTF / HTML / CSV: the browser can't render it safely, so download only
    IMAGE = "image"


class Variant(str, Enum):
    MEDIA = "media"        # what the browser plays or shows inline (transcoded when it can't play the original)
    EXCERPT = "excerpt"    # a recording's cited span plus padding, cut to a small file
    TEXT = "text"          # a TXT/Markdown file decoded to UTF-8, line breaks as the appliance counts them
    ORIGINAL = "original"  # the vault file itself, as a download


# The one place a served file's Content-Type comes from. Never sniffed: what a file claims to be is client data.
_TYPES: Dict[str, Tuple[SourceKind, str]] = {
    ".mp4": (SourceKind.VIDEO, "video/mp4"),
    ".webm": (SourceKind.VIDEO, "video/webm"),
    ".mkv": (SourceKind.VIDEO, "video/x-matroska"),
    ".mov": (SourceKind.VIDEO, "video/quicktime"),
    ".avi": (SourceKind.VIDEO, "video/x-msvideo"),
    ".mp3": (SourceKind.AUDIO, "audio/mpeg"),
    ".wav": (SourceKind.AUDIO, "audio/wav"),
    ".flac": (SourceKind.AUDIO, "audio/flac"),
    ".m4a": (SourceKind.AUDIO, "audio/mp4"),
    ".aac": (SourceKind.AUDIO, "audio/aac"),
    ".ogg": (SourceKind.AUDIO, "audio/ogg"),
    ".opus": (SourceKind.AUDIO, "audio/ogg"),
    ".wma": (SourceKind.AUDIO, "audio/x-ms-wma"),
    ".alac": (SourceKind.AUDIO, "audio/mp4"),
    ".aiff": (SourceKind.AUDIO, "audio/aiff"),
    ".amr": (SourceKind.AUDIO, "audio/amr"),
    ".mka": (SourceKind.AUDIO, "audio/x-matroska"),
    ".pdf": (SourceKind.PDF, "application/pdf"),
    ".txt": (SourceKind.TEXT, "text/plain"),
    ".md": (SourceKind.TEXT, "text/plain"),
    ".markdown": (SourceKind.TEXT, "text/plain"),
    ".docx": (SourceKind.DOCUMENT,
              "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".epub": (SourceKind.DOCUMENT, "application/epub+zip"),
    ".rtf": (SourceKind.DOCUMENT, "application/rtf"),
    ".html": (SourceKind.DOCUMENT, "text/html"),
    ".htm": (SourceKind.DOCUMENT, "text/html"),
    ".csv": (SourceKind.DOCUMENT, "text/csv"),
    ".jpg": (SourceKind.IMAGE, "image/jpeg"),
    ".jpeg": (SourceKind.IMAGE, "image/jpeg"),
    ".png": (SourceKind.IMAGE, "image/png"),
    ".webp": (SourceKind.IMAGE, "image/webp"),
    ".tif": (SourceKind.IMAGE, "image/tiff"),
    ".tiff": (SourceKind.IMAGE, "image/tiff"),
    ".bmp": (SourceKind.IMAGE, "image/bmp"),
}


def resolve_vault_path(raw: str, vault_root: Path) -> Path:
    """
    The real file behind `raw`, if and only if it is a regular file inside the vault with a suffix the appliance
    ingests. Symlinks are resolved first, so one pointing out of the vault is refused like any other escape.
    """
    if not raw or "\x00" in raw:
        raise SourceRejected("No source path was given.")
    try:
        path = Path(raw).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SourceRejected("The cited file is no longer on the appliance.") from exc
    if not path.is_relative_to(vault_root.resolve()):
        raise SourceRejected("Only the appliance's own stored files can be opened.")
    if not path.is_file():
        raise SourceRejected("The cited source is not a file.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES or path.suffix.lower() not in _TYPES:
        raise SourceRejected(f"'{path.suffix}' files are not served.")
    return path


def classify(path: Path) -> Tuple[SourceKind, str]:
    return _TYPES[path.suffix.lower()]


def download_name(doc_label: str, location: str, suffix: str) -> str:
    """`Izvoria Doc 3 - p. 7 - Article 7 Termination.pdf`: what the tester saves, recognisable without the app."""
    label = " - ".join(part for part in (doc_label.strip("[] "), location) if part)
    label = label.replace(":", ".").replace("–", "-").replace("·", "-")
    label = re.sub(r"[^\w.,()\- ]", "_", label, flags=re.UNICODE)
    label = re.sub(r"\s+", " ", label).strip(" .")[:150]
    return f"Izvoria {label}{suffix.lower()}" if label else f"Izvoria source{suffix.lower()}"


@dataclass(frozen=True)
class OfferRequest:
    """What the desktop app posts. Every field but the path is display data, shown with textContent only."""
    path: str
    media_kind: str = ""
    anchor_kind: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    doc_label: str = ""
    location: str = ""
    excerpt: str = ""
    speakers: Tuple[str, ...] = ()

    @classmethod
    def from_json(cls, body: dict) -> "OfferRequest":
        if not isinstance(body, dict):
            raise SourceRejected("Expected a JSON object.")

        def number(name: str) -> float:
            try:
                return max(0.0, float(body.get(name) or 0.0))
            except (TypeError, ValueError):
                raise SourceRejected(f"'{name}' must be a number.")

        def whole(name: str) -> Optional[int]:
            value = body.get(name)
            if value is None:
                return None
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                raise SourceRejected(f"'{name}' must be a whole number.")

        speakers = body.get("speakers") or ()
        return cls(
            path=str(body.get("path") or ""),
            media_kind=str(body.get("media_kind") or "")[:32],
            anchor_kind=str(body.get("anchor_kind") or "")[:32],
            start_sec=number("start_sec"),
            end_sec=number("end_sec"),
            page_start=whole("page_start"),
            page_end=whole("page_end"),
            doc_label=str(body.get("doc_label") or "")[:64],
            location=str(body.get("location") or "")[:300],
            excerpt=str(body.get("excerpt") or "")[:MAX_EXCERPT_CHARS],
            speakers=tuple(str(s)[:64] for s in speakers)[:20] if isinstance(speakers, (list, tuple)) else (),
        )


@dataclass(frozen=True)
class Offer:
    id: str
    seq: int
    path: Path
    kind: SourceKind
    mime: str
    request: OfferRequest
    created_at: float
    created_wall: str
    variants: Tuple[Variant, ...] = field(default=())

    @property
    def is_clip(self) -> bool:
        # A per-chunk clip already is the cited span; only a whole recording needs cutting and seeking.
        return self.request.media_kind == "clip"

    @property
    def excerpt_window(self) -> Tuple[float, float]:
        start = max(0.0, self.request.start_sec - EXCERPT_PADDING_SECONDS)
        end = max(self.request.end_sec, self.request.start_sec) + EXCERPT_PADDING_SECONDS
        return start, end

    def public(self) -> dict:
        """What the gateway and browser see. Never the path."""
        r = self.request
        return {
            "id": self.id,
            "seq": self.seq,
            "kind": self.kind.value,
            "clip": self.is_clip,
            "variants": [v.value for v in self.variants],
            "doc_label": r.doc_label,
            "location": r.location,
            "anchor_kind": r.anchor_kind,
            "start_sec": r.start_sec,
            "end_sec": r.end_sec,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "excerpt": r.excerpt,
            "speakers": list(r.speakers),
            "excerpt_start_sec": self.excerpt_window[0] if Variant.EXCERPT in self.variants else None,
            "download_name": download_name(r.doc_label, r.location, self.path.suffix),
            "size_bytes": self.path.stat().st_size if self.path.exists() else None,
            "created": self.created_wall,
        }


def variants_for(kind: SourceKind, path: Path, request: OfferRequest) -> Tuple[Variant, ...]:
    if kind in (SourceKind.VIDEO, SourceKind.AUDIO):
        whole_recording = request.media_kind != "clip" and request.end_sec > 0
        return ((Variant.EXCERPT,) if whole_recording else ()) + (Variant.MEDIA, Variant.ORIGINAL)
    if kind is SourceKind.TEXT:
        fits = path.stat().st_size <= MAX_TEXT_BYTES
        return ((Variant.TEXT,) if fits else ()) + (Variant.ORIGINAL,)
    if kind in (SourceKind.PDF, SourceKind.IMAGE):
        return Variant.MEDIA, Variant.ORIGINAL
    return (Variant.ORIGINAL,)


class OfferStore:
    """
    The last MAX_OFFERS sources, for OFFER_TTL_SECONDS, in memory: an agent restart forgets them, and the desktop
    app's next click offers the source again. `seq` restarts at 1 then, which the gateway notices.
    """

    def __init__(self, vault_root: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self._vault_root = vault_root
        self._clock = clock
        self._offers: Dict[str, Offer] = {}
        self._seq = 0
        self._changed = asyncio.Condition()

    @property
    def latest_seq(self) -> int:
        return self._seq

    async def add(self, request: OfferRequest) -> Offer:
        path = resolve_vault_path(request.path, self._vault_root)
        kind, mime = classify(path)
        async with self._changed:
            self._purge()
            self._seq += 1
            offer = Offer(
                id=secrets.token_urlsafe(16),
                seq=self._seq,
                path=path,
                kind=kind,
                mime=mime,
                request=request,
                created_at=self._clock(),
                created_wall=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                variants=variants_for(kind, path, request),
            )
            self._offers[offer.id] = offer
            while len(self._offers) > MAX_OFFERS:
                self._offers.pop(next(iter(self._offers)))
            self._changed.notify_all()
        return offer

    def get(self, offer_id: str) -> Optional[Offer]:
        if not OFFER_ID.match(offer_id or ""):
            return None
        self._purge()
        return self._offers.get(offer_id)

    def after(self, seq: int) -> List[Offer]:
        self._purge()
        return [offer for offer in self._offers.values() if offer.seq > seq]

    async def wait_after(self, seq: int, timeout: float) -> List[Offer]:
        """
        Offers newer than `seq`, waiting up to `timeout` for one when there are none yet (a long-poll). A `seq` past
        the latest one comes from before an agent restart: everything held now is newer than what the caller saw.
        """
        if seq > self._seq:
            seq = 0
        async with self._changed:
            try:
                await asyncio.wait_for(self._changed.wait_for(lambda: bool(self.after(seq))), timeout)
            except asyncio.TimeoutError:
                return []
        return self.after(seq)

    def _purge(self) -> None:
        cutoff = self._clock() - OFFER_TTL_SECONDS
        for offer_id in [i for i, o in self._offers.items() if o.created_at < cutoff]:
            del self._offers[offer_id]
