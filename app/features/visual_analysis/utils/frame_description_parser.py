# Path: app/features/visual_analysis/utils/frame_description_parser.py

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# "Frame 2: ...", tolerating markdown emphasis, list bullets and dash separators ("- **Frame 2** - ...")
_FRAME_MARKER = re.compile(
    r"^[\s>*_#-]*frame\s*(\d+)\s*[*_]*\s*[:.\-–—)]\s*[*_]*\s*",
    re.IGNORECASE | re.MULTILINE
)


def parse_frame_descriptions(raw_output: str, frame_count: int) -> List[str]:
    """
    Splits a multi-image VLM answer into one description per frame.

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
