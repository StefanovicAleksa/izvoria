# Path: app/features/visual_analysis/utils/observation_renderer.py

import re

from app.features.visual_analysis.domain.entities import FrameObservation


def _sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].upper() + text[1:] + "." if text else ""


def render_observation(observation: FrameObservation) -> str:
    """
    Writes the stored description from an observation's fields. What was read on screen, what is marked, and what
    the speaker refers to each get their own sentence, so an indexed description never presents a spoken detail as
    something visible.

    Example: 'Slide with on-screen text "BoE REVERSAL 22/3/18"; "Result 7-2 for hike BUT". Marked on screen: arrow at
    the spike. Speaker's reference: the 7-2 vote.'
    """
    # "Slide with text" would read "Slide with text with on-screen text ..."
    shows = re.sub(r"\s+with\s+(some\s+)?text$", "", observation.shows.strip().rstrip("."), flags=re.IGNORECASE)
    if observation.legible_text:
        quoted = "; ".join(f'"{item}"' for item in observation.legible_text)
        parts = [f"{_sentence(shows)[:-1]} with on-screen text {quoted}."]
    else:
        parts = [_sentence(shows)]
    if observation.marks:
        parts.append(f"Marked on screen: {observation.marks.strip().rstrip('.')}.")
    if observation.speech_link:
        parts.append(f"Speaker's reference: {observation.speech_link.strip().rstrip('.')}.")
    return " ".join(part for part in parts if part)
