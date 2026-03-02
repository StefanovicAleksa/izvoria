# Canvas: test_scene_api_v1.py
# Path: tests/features/scene_detector/test_scene_api.py

import pytest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.features.scene_detector.service.scene_api import SceneDetectorAPI
from app.features.scene_detector.domain.entities import SceneMap


class TestSceneDetectorAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "ffmpeg": MagicMock(),
            "storage": MagicMock(),
            "deduplicator": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return SceneDetectorAPI(
            repository=mocks["repo"],
            ffmpeg_adapter=mocks["ffmpeg"],
            storage_api=mocks["storage"],
            deduplicator=mocks["deduplicator"]
        )

    def test_process_video_idempotency(self, api, mocks):
        """Returns instantly if SceneMap already exists in the database."""
        source_id = uuid.uuid4()
        existing_map = SceneMap(source_id=source_id, scenes=[])

        mocks["repo"].get_map_by_source_id.return_value = existing_map

        result = api.process_video(source_id, "vid.mp4")

        assert result == existing_map
        mocks["ffmpeg"].extract_scenes.assert_not_called()

    @patch("app.features.scene_detector.service.scene_api.tempfile.TemporaryDirectory")
    def test_process_video_happy_path(self, mock_tempdir, api, mocks):
        """Tests the full orchestration flow."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None

        # Mock temp dir
        mock_tempdir.return_value.__enter__.return_value = "/tmp/onyx"

        # Mock FFmpeg returning raw scenes
        raw_scenes = [(12.0, Path("/tmp/onyx/frame.jpg"))]
        mocks["ffmpeg"].extract_scenes.return_value = raw_scenes

        # Mock Deduplicator surviving
        unique_scenes = [(12.0, Path("/tmp/onyx/frame.jpg"), "hash123")]
        mocks["deduplicator"].deduplicate.return_value = unique_scenes

        # Mock Storage
        vaulted_file = MagicMock()
        vaulted_file.id = uuid.uuid4()
        mocks["storage"].ingest_raw_file.return_value = vaulted_file

        # Mock Repo
        mocks["repo"].save_map.side_effect = lambda x: x

        result = api.process_video(source_id, "vid.mp4")

        # Assert flow
        assert result.source_id == source_id
        assert len(result.scenes) == 1
        assert result.scenes[0].extracted_file_id == vaulted_file.id

        mocks["ffmpeg"].extract_scenes.assert_called_once()
        mocks["deduplicator"].deduplicate.assert_called_once_with(raw_scenes)
        mocks["storage"].ingest_raw_file.assert_called_once()
        mocks["repo"].save_map.assert_called_once()