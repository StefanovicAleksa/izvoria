# Path: app/features/visual_hallucination_correction/utils/topic_overlap.py

import re
from typing import List, Set

_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
_WORD = re.compile(r"[a-z][a-z0-9&'-]{2,}")
# Numbers with inner punctuation ("7-2", "160.63", "22/3/18") are specific enough to tie a slide to the speech
_SPECIFIC_NUMBER = re.compile(r"\d+(?:[.:/-]\d+)+")
_FUNCTION_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "you", "your", "have", "has", "but",
    "not", "into", "onto", "over", "after", "before", "then", "than", "our", "its", "will", "can", "all", "any", "out",
    "off", "per", "what", "which", "who", "how", "when", "why", "there", "here", "they", "them", "his", "her"
}


def _words(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _FUNCTION_WORDS}


def shared_on_screen_topic(transcript: str, visual_descriptions: List[str]) -> Set[str]:
    """
    Words and specific numbers that the quoted on-screen text of the descriptions shares with the transcript.
    Only quoted text counts: it was read off the frame, whereas the rest of a description ("chart", "arrow pointing
    at the lower left") is generic enough to overlap with any speech.
    """
    on_screen = " ".join(a or b for d in visual_descriptions for a, b in _QUOTED.findall(d))
    return (_words(on_screen) & _words(transcript)) | (
        set(_SPECIFIC_NUMBER.findall(on_screen)) & set(_SPECIFIC_NUMBER.findall(transcript))
    )


def shares_topic(transcript: str, visual_descriptions: List[str], min_shared: int) -> bool:
    """
    True when the frames' on-screen text visibly belongs to what is being said: at least `min_shared` shared topic
    words, or one shared specific number. Such evidence can't be "completely unrelated" to the speech.
    """
    shared = shared_on_screen_topic(transcript, visual_descriptions)
    return len(shared) >= min_shared or any(_SPECIFIC_NUMBER.fullmatch(item) for item in shared)
