# Canvas: test_timeline_merger_v2
# Path: tests/features/intelligence_bridge/test_timeline_merger.py

import uuid

from app.legacy_features.scene_detector.domain.entities import SceneMap, DetectedScene
from app.legacy_features.intelligence_bridge.domain.entities import IntelligentSceneMap
from app.legacy_features.intelligence_bridge.utils.timeline_merger import (
    calculate_hamming_distance,
    merge_scene_maps
)


class TestTimelineMerger:
    def test_calculate_hamming_distance(self):
        # "f0f0" = 1111000011110000, "f000" = 1111000000000000 -> Distance is 4
        assert calculate_hamming_distance("f0f0", "f000") == 4
        assert calculate_hamming_distance("0000", "0000") == 0

        # Missing hashes should return max distance (64)
        # We pass an empty string to satisfy the type-checker instead of None
        assert calculate_hamming_distance("", "f0f0") == 64
        assert calculate_hamming_distance("invalid", "f0f0") == 64

    def test_merge_scene_maps_favors_semantic(self):
        """Verifies semantic scenes overwrite identical global scenes."""
        source_id = uuid.uuid4()

        # Identical phash values to trigger deduplication
        global_scene = DetectedScene(timestamp_sec=10.0, extracted_file_id=uuid.uuid4(), phash_value="abcde")
        semantic_scene = DetectedScene(timestamp_sec=10.5, extracted_file_id=uuid.uuid4(), phash_value="abcde")

        global_map = SceneMap(source_id=source_id, scenes=[global_scene])
        semantic_map = SceneMap(source_id=source_id, scenes=[semantic_scene])

        result = merge_scene_maps(source_id, global_map, semantic_map, phash_threshold=5)

        assert isinstance(result, IntelligentSceneMap)
        assert len(result.scenes) == 1
        assert result.scenes[0].extraction_method == "semantic_pass"
        assert result.scenes[0].extracted_file_id == semantic_scene.extracted_file_id

    def test_merge_scene_maps_keeps_distinct(self):
        """Verifies distinct frames are kept and sorted."""
        source_id = uuid.uuid4()

        # Using standard hex strings to avoid spell-check typos
        global_scene = DetectedScene(timestamp_sec=5.0, extracted_file_id=uuid.uuid4(), phash_value="11111")
        semantic_scene = DetectedScene(timestamp_sec=10.0, extracted_file_id=uuid.uuid4(), phash_value="aaaaa")

        global_map = SceneMap(source_id=source_id, scenes=[global_scene])
        semantic_map = SceneMap(source_id=source_id, scenes=[semantic_scene])

        result = merge_scene_maps(source_id, global_map, semantic_map, phash_threshold=5)

        assert len(result.scenes) == 2
        assert result.scenes[0].timestamp_sec == 5.0
        assert result.scenes[0].extraction_method == "global_pass"
        assert result.scenes[1].timestamp_sec == 10.0
        assert result.scenes[1].extraction_method == "semantic_pass"

    def test_merge_scene_maps_empty(self):
        source_id = uuid.uuid4()
        global_map = SceneMap(source_id=source_id, scenes=[])
        semantic_map = SceneMap(source_id=source_id, scenes=[])

        result = merge_scene_maps(source_id, global_map, semantic_map)
        assert len(result.scenes) == 0