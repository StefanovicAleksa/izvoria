# Canvas: test_diarization_adapter_v4
# Path: tests/features/diarization/test_nemo_adapter.py

import uuid
import pytest
from unittest.mock import patch, MagicMock, mock_open

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter


class TestNeMoDiarizationAdapter:
    @pytest.fixture
    def adapter(self):
        return NeMoDiarizationAdapter()

    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.DIARIZATION,
            model_path="/fake/path/titanet.nemo",
            required_vram_mb=500.0
        )

    @patch("os.path.exists")
    def test_load_success(self, mock_exists, adapter, manifest):
        mock_exists.return_value = True
        loaded_model = adapter.load(manifest)
        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == "/fake/path/titanet.nemo"

    @patch("os.path.exists")
    def test_load_file_not_found(self, mock_exists, adapter, manifest):
        mock_exists.return_value = False
        with pytest.raises(Exception):
            adapter.load(manifest)

    def test_unload_success(self, adapter, manifest):
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        assert adapter.unload(loaded_model) is True

    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("app.features.diarization.data.nemo_adapter.Path.exists")
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_pure_mock(
            self, mock_temp_dir, mock_file_open, mock_path_exists, mock_get_diarizer, adapter, manifest
    ):
        """
        Tests the file I/O operations and RTTM parsing completely isolated
        from the NeMo hardware runtime.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=1.0, end_sec=5.0, confidence=0.99)])

        # 1. OS Mocks
        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = "/tmp/fake_onyx_dir"
        mock_temp_dir.return_value = mock_temp
        mock_path_exists.return_value = True

        # 2. File Handles (Fixing the exhaustion bug)
        fake_rttm_content = "SPEAKER fake_audio 1 1.5 2.5 <NA> <NA> Speaker_0 <NA> <NA>\n"
        handle_write_rttm = mock_open().return_value
        handle_write_manifest = mock_open().return_value
        handle_read_result = mock_open(read_data=fake_rttm_content).return_value

        # Sequentially assign handles to the 3 open() calls
        mock_file_open.side_effect = [
            handle_write_rttm,
            handle_write_manifest,
            handle_read_result
        ]

        # 3. NeMo Factory Mock (No NeMo imports occur!)
        mock_diarizer_instance = MagicMock()
        mock_get_diarizer.return_value = mock_diarizer_instance

        # Execute
        result = adapter.generate_diarization_map(loaded_model, "/fake/audio.wav", vad_map, source_id)

        # Assertions
        assert result.source_id == source_id
        assert len(result.segments) == 1
        assert result.segments[0].speaker_id == "Speaker_0"
        assert result.segments[0].start_sec == 1.5
        assert result.segments[0].end_sec == 4.0

        # Verify Execution Flow
        mock_get_diarizer.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once()
        assert mock_file_open.call_count == 3