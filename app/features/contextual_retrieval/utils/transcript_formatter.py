# Path: app/features/contextual_retrieval/utils/transcript_formatter.py

from typing import List

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence


def format_timestamp(seconds: float) -> str:
    """Formats seconds as HH:MM:SS (floored), e.g. 751.4 -> '00:12:31'."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def format_speaker_turns(sentences: List[ReconstructedSentence], include_timestamps: bool = True) -> str:
    """
    Renders sentences as speaker turns, one line per turn:
        [Speaker_0 @ 00:12:31]: First sentence. Second sentence.
        [Speaker_1 @ 00:12:40]: Reply.

    Consecutive sentences from the same speaker are merged into a single turn stamped with the
    turn's start time. Speaker attribution and time are the context a legal/medical reader needs
    ("who said it, when"), so they are kept in both the indexed text and the document the
    context model reads.

    With `include_timestamps=False` turns render as `Speaker_0: text`. That lighter form is what
    the context SLM reads: given bracketed timestamp labels, Llama 3.2 3B tended to answer by
    echoing a label instead of writing context.
    """
    lines: List[str] = []
    current_speaker = None
    turn_start = 0.0
    turn_parts: List[str] = []

    def _flush() -> None:
        if turn_parts:
            label = f"[{current_speaker} @ {format_timestamp(turn_start)}]" if include_timestamps else current_speaker
            lines.append(f"{label}: {' '.join(turn_parts)}")

    for sentence in sentences:
        text = sentence.text.strip()
        if not text:
            continue

        if sentence.speaker_id != current_speaker or not turn_parts:
            _flush()
            current_speaker = sentence.speaker_id
            turn_start = sentence.start_sec
            turn_parts = [text]
        else:
            turn_parts.append(text)

    _flush()
    return "\n".join(lines)


def ordered_speaker_labels(sentences: List[ReconstructedSentence]) -> List[str]:
    """Distinct speaker ids in order of first appearance."""
    return list(dict.fromkeys(s.speaker_id for s in sentences if s.text.strip()))
