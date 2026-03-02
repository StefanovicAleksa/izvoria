# Canvas: intelligence_bridge_timeline_merger_v2
# Path: app/features/intelligence_bridge/utils/timeline_merger.py

import uuid
from typing import List

from app.features.scene_detector.domain.entities import SceneMap
from app.features.intelligence_bridge.domain.entities import IntelligentSceneMap, EnrichedScene
from app.features.intelligence_bridge.utils.error_mapper import map_bridge_error


def _hex_to_int(hex_str: str) -> int:
    """Converts a hex pHash string to an integer for bitwise comparison."""
    return int(hex_str, 16)


def calculate_hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates the Hamming distance between two perceptual hashes."""
    if not hash1 or not hash2:
        return 64  # Max distance if either hash is missing

    try:
        x = _hex_to_int(hash1) ^ _hex_to_int(hash2)
        return bin(x).count('1')
    except (ValueError, TypeError):
        return 64


def merge_scene_maps(
        source_id: uuid.UUID,
        global_map: SceneMap,
        semantic_map: SceneMap,
        phash_threshold: int = 5
) -> IntelligentSceneMap:
    """
    Combines global and semantic scene maps into a single, deduplicated timeline.
    Favors semantic scenes over global scenes if they are visually identical.
    """
    try:
        raw_scenes: List[EnrichedScene] = []

        # 1. Tag the Global Pass frames
        for scene in global_map.scenes:
            raw_scenes.append(EnrichedScene(
                timestamp_sec=scene.timestamp_sec,
                extracted_file_id=scene.extracted_file_id,
                phash_value=scene.phash_value,
                extraction_method="global_pass",
                semantic_context=None
            ))

        # 2. Tag the Semantic Pass frames
        for scene in semantic_map.scenes:
            raw_scenes.append(EnrichedScene(
                timestamp_sec=scene.timestamp_sec,
                extracted_file_id=scene.extracted_file_id,
                phash_value=scene.phash_value,
                extraction_method="semantic_pass",
                semantic_context="Extracted via NLP Router Targeting"
            ))

        # 3. Sort chronologically
        raw_scenes.sort(key=lambda s: s.timestamp_sec)

        if not raw_scenes:
            return IntelligentSceneMap(source_id=source_id, scenes=[])

        # 4. Master Deduplication
        deduplicated: List[EnrichedScene] = [raw_scenes[0]]

        for current in raw_scenes[1:]:
            last_saved = deduplicated[-1]
            distance = calculate_hamming_distance(last_saved.phash_value, current.phash_value)

            if distance <= phash_threshold:
                # They are visually identical.
                # If current is semantic and last was global, REPLACE it to keep the semantic tag.
                if current.extraction_method == "semantic_pass" and last_saved.extraction_method == "global_pass":
                    deduplicated[-1] = current
                # Otherwise, drop the current frame.
            else:
                # Visually distinct, keep it!
                deduplicated.append(current)

        return IntelligentSceneMap(source_id=source_id, scenes=deduplicated)

    except Exception as e:
        raise map_bridge_error(e, context="Merging Scene Maps") from e