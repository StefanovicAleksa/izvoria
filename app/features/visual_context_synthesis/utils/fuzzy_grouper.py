# Canvas: visual_synthesis_fuzzy_grouper_v3
# Path: app/features/visual_context_synthesis/utils/fuzzy_grouper.py

import re
import uuid
import nltk
from nltk.corpus import stopwords
from typing import List, Dict, Any, Set

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

CORE_STOP_WORDS: Set[str] = set(stopwords.words('english'))
VISION_NOISE: Set[str] = {
    "showing", "depicting", "displaying", "image", "shows",
    "features", "picture", "photo", "screen", "frame", "appears"
}
FINAL_STOP_WORDS = CORE_STOP_WORDS.union(VISION_NOISE)


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    words1 = {w for w in re.findall(r'\b\w+\b', text1.lower()) if w not in FINAL_STOP_WORDS}
    words2 = {w for w in re.findall(r'\b\w+\b', text2.lower()) if w not in FINAL_STOP_WORDS}

    if not words1 or not words2:
        return 1.0 if not words1 and not words2 else 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def build_fuzzy_visual_timeline(
        visual_analyses: List[Any],
        accepted_frame_ids: Set[uuid.UUID],
        similarity_threshold: float = 0.65,
        base_timestamp_sec: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Groups stochastic VLM outputs into chronological time-spans.
    Safely falls back to base_timestamp_sec if the entities lack explicit timestamps.
    """
    # If the entity has a timestamp, use it for sorting; otherwise, preserve list order
    sorted_analyses = sorted(visual_analyses, key=lambda x: getattr(x, "timestamp_sec", base_timestamp_sec))
    valid_analyses = [a for a in sorted_analyses if getattr(a, "frame_id") in accepted_frame_ids]

    if not valid_analyses:
        return []

    visual_timeline: List[Dict[str, Any]] = []
    current_event: Dict[str, Any] = {}

    for analysis in valid_analyses:
        frame_id = getattr(analysis, "frame_id")

        # FIX: Safely retrieve timestamp or use the provided fallback
        timestamp = getattr(analysis, "timestamp_sec", base_timestamp_sec)
        description = getattr(analysis, "description", "")

        if not current_event:
            current_event = {
                "representative_description": description,
                "start_sec": timestamp,
                "end_sec": timestamp,
                "frame_ids": [frame_id]
            }
        else:
            similarity = calculate_jaccard_similarity(description, current_event["representative_description"])

            if similarity >= similarity_threshold:
                current_event["end_sec"] = max(current_event["end_sec"], timestamp)
                current_event["frame_ids"].append(frame_id)

                if len(description) > len(current_event["representative_description"]):
                    current_event["representative_description"] = description
            else:
                visual_timeline.append(current_event)
                current_event = {
                    "representative_description": description,
                    "start_sec": timestamp,
                    "end_sec": timestamp,
                    "frame_ids": [frame_id]
                }

    if current_event:
        visual_timeline.append(current_event)

    return visual_timeline