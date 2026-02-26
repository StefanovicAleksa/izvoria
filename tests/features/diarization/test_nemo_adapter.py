# Canvas: test_diarization_adapter_v6
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

        # 2. File Handles
        fake_rttm_content = "SPEAKER fake_audio 1 1.5 2.5 <NA> <NA> Speaker_1 <NA> <NA>\n"
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
        assert result.segments[0].speaker_id == "Speaker_1"
        assert result.segments[0].start_sec == 1.5
        assert result.segments[0].end_sec == 4.0

        # Verify Execution Flow
        mock_get_diarizer.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once()
        assert mock_file_open.call_count == 3

    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_fallback_on_diarize_exception(
            self, mock_temp_dir, mock_file_open, mock_get_diarizer, adapter, manifest
    ):
        """
        Verifies that if NeMo's Clustering algorithm crashes (e.g., due to a monologue
        or LinAlgError), we safely fall back to marking all VAD speech as 'Speaker_0'.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=1.0, end_sec=5.0, confidence=0.99)])

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = "/tmp/fake_onyx_dir"
        mock_temp_dir.return_value = mock_temp

        mock_diarizer_instance = MagicMock()
        # Simulate the math crash
        mock_diarizer_instance.diarize.side_effect = ValueError("Singular matrix during Spectral Clustering")
        mock_get_diarizer.return_value = mock_diarizer_instance

        # Execute
        result = adapter.generate_diarization_map(loaded_model, "/fake/audio.wav", vad_map, source_id)

        # Assertions (Verifies the VAD map was copied perfectly and labeled Speaker_0)
        assert result.source_id == source_id
        assert len(result.segments) == 1
        assert result.segments[0].speaker_id == "Speaker_0"
        assert result.segments[0].start_sec == 1.0
        assert result.segments[0].end_sec == 5.0

        # Verify that the two setup files (rttm + manifest) were successfully written before the crash
        assert mock_file_open.call_count == 2

    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("app.features.diarization.data.nemo_adapter.Path.exists")
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_fallback_on_missing_rttm(
            self, mock_temp_dir, mock_file_open, mock_path_exists, mock_get_diarizer, adapter, manifest
    ):
        """
        Verifies that if NeMo successfully completes but fails to output the resulting .rttm file,
        we also fall back gracefully.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=2.0, end_sec=4.0, confidence=0.95)])

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = "/tmp/fake_onyx_dir"
        mock_temp_dir.return_value = mock_temp

        # Simulate missing RTTM output file
        mock_path_exists.return_value = False

        mock_diarizer_instance = MagicMock()
        mock_get_diarizer.return_value = mock_diarizer_instance

        # Execute
        result = adapter.generate_diarization_map(loaded_model, "/fake/audio.wav", vad_map, source_id)

        # Assertions
        assert result.source_id == source_id
        assert len(result.segments) == 1
        assert result.segments[0].speaker_id == "Speaker_0"
        assert result.segments[0].start_sec == 2.0
        assert result.segments[0].end_sec == 4.0

        # Verify that the two setup files (rttm + manifest) were successfully written before exiting early
        assert mock_file_open.call_count == 2