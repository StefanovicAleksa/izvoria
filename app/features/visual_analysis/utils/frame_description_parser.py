# Path: app/features/visual_analysis/utils/frame_description_parser.py

import logging
import re
from typing import List, Optional

from app.features.visual_analysis.domain.entities import FrameObservation

logger = logging.getLogger(__name__)

# "Frame 2: ...", tolerating markdown emphasis, list bullets and dash separators ("- **Frame 2** - ...")
_FRAME_MARKER = re.compile(
    r"^[\s>*_#-]*frame\s*(\d+)\s*[*_]*\s*[:.\-–—)]\s*[*_]*\s*",
    re.IGNORECASE | re.MULTILINE
)

# Field labels are upper-case by instruction; lower-case "text:" inside a value must not start a field
_FIELD = re.compile(r"[*_]*\b(SHOWS|TEXT|MARKS|LINK)\b[*_]*\s*:\s*[*_]*")
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
_EMPTY_VALUES = {"", "NONE", "N/A", "NA", "NOTHING", "-"}


def parse_frame_descriptions(raw_output: str, frame_count: int) -> List[str]:
    """
    Splits a multi-image VLM answer into one text section per frame.

    Text after a "Frame N:" marker, up to the next marker, belongs to frame N (whitespace is collapsed and
    repeated markers are appended). Markers outside 1..frame_count and text before the first marker are
    dropped. If the model ignored the format entirely, the whole answer is attributed to the first frame
    so the evidence is not lost.
    """
    if frame_count <= 0:
        return []

    text = raw_output.strip()
    descriptions = [""] * frame_count
    markers = list(_FRAME_MARKER.finditer(text))

    if not markers:
        if frame_count > 1 and text:
            logger.warning(
                f"VLM answer for {frame_count} frames had no 'Frame N:' markers; attributing it to the first frame."
            )
        descriptions[0] = " ".join(text.split())
        return descriptions

    for position, marker in enumerate(markers):
        end = markers[position + 1].start() if position + 1 < len(markers) else len(text)
        body = " ".join(text[marker.end():end].split())
        index = int(marker.group(1)) - 1

        if not body or not 0 <= index < frame_count:
            continue
        descriptions[index] = f"{descriptions[index]} {body}".strip()

    return descriptions


def _clean(value: str) -> str:
    return value.strip().strip("*_").strip()


def _is_empty(value: str) -> bool:
    """NONE-like values, and placeholders the model copied from the format ("<what is on screen>", "...")."""
    cleaned = _clean(value)
    return (cleaned.strip(".\"' ;").upper() in _EMPTY_VALUES
            or bool(re.fullmatch(r"<[^>]*>", cleaned.rstrip(". ")))
            or not re.sub(r"[.\"'“”;,\s…]", "", cleaned))


def _legible_items(value: str) -> List[str]:
    if _is_empty(value):
        return []
    quoted = [a or b for a, b in _QUOTED.findall(value)]
    items = (" ".join(item.split()).strip(" ;,") for item in (quoted if quoted else value.split(";")))
    return [item for item in items if item and not _is_empty(item)]


def parse_frame_observation(section: str) -> Optional[FrameObservation]:
    """
    Reads the SHOWS / TEXT / MARKS / LINK fields of one frame's section.
    Returns None when the section has no SHOWS field: an unstructured answer carries no checkable evidence.
    """
    matches = list(_FIELD.finditer(section))
    fields = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(section)
        fields.setdefault(match.group(1), section[match.end():end])

    if "SHOWS" not in fields or _is_empty(fields["SHOWS"]):
        return None

    marks = fields.get("MARKS", "")
    link = fields.get("LINK", "")
    return FrameObservation(
        shows=_clean(fields["SHOWS"]).rstrip("."),
        legible_text=_legible_items(fields.get("TEXT", "")),
        marks=None if _is_empty(marks) else _clean(marks).rstrip("."),
        speech_link=None if _is_empty(link) else _clean(link).rstrip("."),
    )


def parse_frame_observations(raw_output: str, frame_count: int) -> List[Optional[FrameObservation]]:
    """One observation per frame, in order; None for a frame whose section is missing or unstructured."""
    sections = parse_frame_descriptions(raw_output, frame_count)
    observations = [parse_frame_observation(section) if section else None for section in sections]
    unstructured = sum(1 for section, observation in zip(sections, observations) if section and observation is None)
    if unstructured:
        logger.warning(f"{unstructured} of {frame_count} VLM frame section(s) had no SHOWS field; they are discarded.")
    return observations
