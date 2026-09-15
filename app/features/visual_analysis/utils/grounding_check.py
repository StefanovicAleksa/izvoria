# Path: app/features/visual_analysis/utils/grounding_check.py

import re
from typing import List, Optional, Set

from app.features.visual_analysis.domain.entities import FrameObservation

_NOTHING_SHOWN = re.compile(
    r"\b(speaker only|only the speaker|talking head|presenter only|person speaking|nothing relevant|blank|black screen)\b",
    re.IGNORECASE
)
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_WEBSITE = re.compile(r"(www\.|https?://|\.com\b|\.net\b|\.org\b)", re.IGNORECASE)
# Video player and navigation chrome ("16 / 31", "00:21 / 04:32", "PREV") is legible but isn't content
_INTERFACE_TEXT = re.compile(
    r"^(prev|previous|next|search\.*|outline|menu|play|pause|marker tools|\d+\s*/\s*\d+|"
    r"\d{1,2}:\d{2}(:\d{2})?\s*/\s*\d{1,2}:\d{2}(:\d{2})?)$",
    re.IGNORECASE
)
# Marks that say nothing about what is pointed at can't anchor a description
_GENERIC_MARKS = re.compile(r"\b(different|various|several|multiple)\s+(parts|areas|places|points|levels|elements)\b",
                            re.IGNORECASE)
# "box pointing to chart", "arrow pointing to a specific point on the ladder": the target names only the frame itself
_MARK_TARGET = re.compile(r"\b(?:pointing|points|point|around|circling|highlighting|at)\s+(?:to|at|towards)?\s*(.+)$",
                          re.IGNORECASE)
_GENERIC_TARGET_WORDS = {
    "a", "an", "the", "this", "that", "it", "some", "specific", "particular", "certain", "one", "on", "of", "in", "to",
    "point", "points", "area", "areas", "part", "parts", "spot", "place", "location", "section", "region", "element",
    "chart", "ladder", "slide", "screen", "image", "frame", "graph", "picture", "window", "panel", "something",
    # Chart furniture: "arrows pointing to lines on the chart" singles nothing out
    "line", "lines", "bar", "bars", "candle", "candles", "candlestick", "candlesticks", "level", "levels", "price",
    "prices", "value", "values", "data"
}
_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "by", "with", "from", "up",
    "down", "is", "are", "was", "were", "be", "we", "you", "it", "that", "this", "can", "will", "so", "as", "our"
}
# A TEXT item with a number whose content words all sit this densely in one stretch of speech was copied from it
SPEECH_COPY_COVERAGE = 0.8
SPEECH_COPY_WINDOW_SLACK = 4
MAX_LEGIBLE_ITEMS = 8
# A presentation player's outline sidebar: several short numbered entries ("15. BOE Reversal 2/8/18", "18. Continuation")
_OUTLINE_ENTRY = re.compile(r"^\d{1,2}\.\s+\S+(\s+\S+){0,3}$")
MIN_OUTLINE_ENTRIES = 3


def _numbers(text: str) -> Set[str]:
    return {n.strip(".,:/-") for n in _NUMBER.findall(text)}


def _content_tokens(text: str) -> List[str]:
    tokens = (t.strip(".,:;!?'\"()").lower() for t in text.split())
    return [t for t in tokens if t and t not in _FUNCTION_WORDS]


def _is_generic_mark(marks: str) -> bool:
    """True when every listed mark points at nothing more specific than the frame itself or "different parts"."""
    if _GENERIC_MARKS.search(marks):
        return True
    for mark in re.split(r"\s*[;,]\s*", marks):
        target = _MARK_TARGET.search(mark)
        words = re.findall(r"[a-z0-9'.-]+", (target.group(1) if target else mark).lower())
        if target is None or any(word not in _GENERIC_TARGET_WORDS for word in words):
            return False
    return True


def _copied_from_speech(item: str, transcript_tokens: List[str]) -> bool:
    """
    An item whose content words (at least 80%, all of a short item) occur together in one stretch of the speech:
    "160.84 if we can get back above 63" or "160.84 downtrend" read into a chart's text, or the spoken sentence
    "There appears to be a shadow in the lower lobe" reported as text on an X-ray with no text at all.
    Items need 2+ content words with a number, or 3+ without, so short titles the speaker also says ("POSITION
    SQUEEZE") stay. Slide bullets the speaker paraphrases ("Result 7-2 for hike BUT" said as "a 7-2 vote for a rate
    hike") keep enough wording of their own to survive; a bullet read out word for word is lost, which removes
    evidence but never adds a false one.
    """
    tokens = _content_tokens(item)
    if _NUMBER.search(item) and len(tokens) < 2:
        # A bare value such as "142.04" listed on a chart whose axis is unreadable: the speaker says it, the frame
        # doesn't show it legibly. A spoken value that really is on screen loses nothing but this one item.
        return _numbers(item) <= _numbers(" ".join(transcript_tokens))
    if len(tokens) < (2 if _NUMBER.search(item) else 3):
        return False
    needed = set(tokens)
    window = len(tokens) + SPEECH_COPY_WINDOW_SLACK
    best = max((len(needed & set(transcript_tokens[i:i + window]))
                for i in range(max(1, len(transcript_tokens) - window + 1))), default=0)
    return best / len(needed) >= SPEECH_COPY_COVERAGE


def ground_observation(observation: Optional[FrameObservation], transcript: str) -> Optional[FrameObservation]:
    """
    Keeps a frame only if it carries visible evidence: something is shown other than the speaker, and at least one
    legible text item or specific drawn mark remains after removing:
    - website addresses, watermarks, video player chrome and outline sidebars (3+ short numbered entries);
    - repeated items;
    - text items copied from the speech;
    - marks that don't say what they point at, or that name a spoken number the frame's text doesn't show.
    A frame without evidence yields no description at all.

    The speech link is dropped (the visible evidence kept) if it names a number the speaker never said: a link may
    repeat the speech, but it can't invent a value.
    """
    if observation is None or not observation.shows.strip() or _NOTHING_SHOWN.search(observation.shows):
        return None

    transcript_tokens = _content_tokens(transcript)
    outline = [item for item in observation.legible_text if _OUTLINE_ENTRY.match(item.strip())]
    skip_outline = len(outline) >= MIN_OUTLINE_ENTRIES
    legible_text: List[str] = []
    for item in observation.legible_text:
        if skip_outline and item in outline:
            continue
        if (_WEBSITE.search(item) or _INTERFACE_TEXT.match(item.strip())
                or item.casefold() in (kept.casefold() for kept in legible_text)
                or _copied_from_speech(item, transcript_tokens)):
            continue
        legible_text.append(item)
    legible_text = legible_text[:MAX_LEGIBLE_ITEMS]

    legible_numbers = _numbers(" ".join(legible_text))
    marks = observation.marks
    if marks and (_is_generic_mark(marks) or (_numbers(marks) & _numbers(transcript)) - legible_numbers):
        marks = None
    if not legible_text and not marks:
        return None

    link = observation.speech_link
    if link and not _numbers(link) <= _numbers(transcript):
        link = None

    return observation.model_copy(update={"legible_text": legible_text, "marks": marks, "speech_link": link})


def drop_items_already_reported(observation: FrameObservation, reported: Set[str]) -> Optional[FrameObservation]:
    """
    Removes text items an earlier frame of the same chunk already reported, and records the new ones in `reported`.
    Frames described in one multimodal call are often given the first frame's text; distinct frames that really show
    the same slide lose nothing, since the earlier frame keeps it. Returns None if no new evidence is left.
    """
    new_items = [item for item in observation.legible_text if item.casefold() not in reported]
    reported.update(item.casefold() for item in new_items)
    if not new_items and not observation.marks:
        return None
    return observation.model_copy(update={"legible_text": new_items})


def drop_marks_copied_from_other_frames(
        observations: List[Optional[FrameObservation]]
) -> List[Optional[FrameObservation]]:
    """
    Clears the marks of a frame when they point at text that another frame of the same multimodal call reported and
    this frame didn't ("blue arrow pointing to spike up on no major change" on the slide before the one that shows
    it). Frames that lose their only evidence become None.
    """
    result: List[Optional[FrameObservation]] = []
    for index, observation in enumerate(observations):
        if observation is None or not observation.marks:
            result.append(observation)
            continue
        own_items = {item.casefold() for item in observation.legible_text}
        mark_tokens = set(_content_tokens(observation.marks))
        copied = any(
            item.casefold() not in own_items
            and len(item_tokens := _content_tokens(item)) >= 3 and set(item_tokens) <= mark_tokens
            for other_index, other in enumerate(observations) if other is not None and other_index != index
            for item in other.legible_text
        )
        if not copied:
            result.append(observation)
        elif observation.legible_text:
            result.append(observation.model_copy(update={"marks": None}))
        else:
            result.append(None)
    return result
