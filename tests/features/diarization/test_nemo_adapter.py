# Canvas: test_diarization_adapter_v8
# Path: tests/features/diarization/test_nemo_adapter.py

import uuid
import pytest
import torch
from unittest.mock import patch, MagicMock, mock_open

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.diarization.data.nemo_adapter import NeMoDiarizationAdapter


class TestNeMoDiarizationAdapter:
    @pytest.fixture
    def adapter(self) -> NeMoDiarizationAdapter:
        return NeMoDiarizationAdapter()

    @pytest.fixture
    def manifest(self) -> ModelManifest:
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

    @patch("torchaudio.load")
    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("app.features.diarization.data.nemo_adapter.Path.exists")
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_pure_mock(
            self, mock_temp_dir, mock_file_open, mock_path_exists, mock_get_diarizer, mock_ta_load, adapter, manifest
    ):
        """
        Tests the file I/O operations and RTTM parsing completely isolated
        from the NeMo hardware runtime.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=1.0, end_sec=5.0, confidence=0.99)])

        # Mock Torchaudio so it doesn't try to read the fake file (Simulates Mono 16kHz)
        mock_ta_load.return_value = (torch.zeros(1, 16000), 16000)

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

    @patch("torchaudio.save")
    @patch("torchaudio.load")
    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("app.features.diarization.data.nemo_adapter.Path.exists")
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_converts_stereo_to_mono(
            self, mock_temp_dir, mock_file_open, mock_path_exists, mock_get_diarizer, mock_ta_load, mock_ta_save,
            adapter, manifest
    ):
        """
        Verifies that if a Stereo or non-16kHz file is passed, it correctly
        resamples, downmixes to Mono, and resaves the file to prevent NeMo crashes.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=1.0, end_sec=5.0, confidence=0.99)])

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = "/tmp/fake_onyx_dir"
        mock_temp_dir.return_value = mock_temp
        mock_path_exists.return_value = True

        # Simulating Stereo, 48kHz Audio
        mock_ta_load.return_value = (torch.zeros(2, 48000), 48000)

        # File handles (Mock successful NeMo execution)
        fake_rttm_content = "SPEAKER audio_mono 1 1.5 2.5 <NA> <NA> Speaker_1 <NA> <NA>\n"
        mock_file_open.side_effect = [
            mock_open().return_value,
            mock_open().return_value,
            mock_open(read_data=fake_rttm_content).return_value
        ]

        mock_diarizer_instance = MagicMock()
        mock_get_diarizer.return_value = mock_diarizer_instance

        # Execute
        result = adapter.generate_diarization_map(loaded_model, "/fake/audio.wav", vad_map, source_id)

        # Verify downmixing occurred
        assert mock_ta_save.called
        saved_tensor = mock_ta_save.call_args[0][1]
        saved_sr = mock_ta_save.call_args[0][2]

        # Must be converted to 16000Hz Mono (1 Channel)
        assert saved_sr == 16000
        assert saved_tensor.shape[0] == 1

        assert len(result.segments) == 1
        assert result.segments[0].speaker_id == "Speaker_1"

    @patch("torchaudio.load")
    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_fallback_on_expected_clustering_error(
            self, mock_temp_dir, mock_file_open, mock_get_diarizer, mock_ta_load, adapter, manifest
    ):
        """
        Verifies that KNOWN math errors (ValueError, LinAlgError) gracefully fall back to 'Speaker_0'.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=1.0, end_sec=5.0, confidence=0.99)])

        # Mock standard Mono audio
        mock_ta_load.return_value = (torch.zeros(1, 16000), 16000)

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = "/tmp/fake_onyx_dir"
        mock_temp_dir.return_value = mock_temp

        mock_diarizer_instance = MagicMock()
        # Simulate a clustering math crash (the expected scenario for the fallback)
        mock_diarizer_instance.diarize.side_effect = ValueError("Singular matrix during Spectral Clustering")
        mock_get_diarizer.return_value = mock_diarizer_instance

        # Execute
        result = adapter.generate_diarization_map(loaded_model, "/fake/audio.wav", vad_map, source_id)

        # Assertions (Verifies it gracefully handled the clustering error)
        assert result.source_id == source_id
        assert len(result.segments) == 1
        assert result.segments[0].speaker_id == "Speaker_0"
        assert result.segments[0].start_sec == 1.0
        assert result.segments[0].end_sec == 5.0

        # Verify that the two setup files (rttm + manifest) were successfully written before the crash
        assert mock_file_open.call_count == 2

    @patch("torchaudio.load")
    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_bubbles_critical_system_errors(
            self, mock_temp_dir, mock_file_open, mock_get_diarizer, mock_ta_load, adapter, manifest
    ):
        """
        Verifies that CRITICAL system errors (like CUDA Out Of Memory) are NOT swallowed
        by the fallback and correctly bubble up to crash the pipeline.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=1.0, end_sec=5.0, confidence=0.99)])

        # Mock standard Mono audio
        mock_ta_load.return_value = (torch.zeros(1, 16000), 16000)

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = "/tmp/fake_onyx_dir"
        mock_temp_dir.return_value = mock_temp

        mock_diarizer_instance = MagicMock()
        # Simulate a hardware crash (should NOT trigger fallback)
        mock_diarizer_instance.diarize.side_effect = RuntimeError("CUDA out of memory")
        mock_get_diarizer.return_value = mock_diarizer_instance

        # Execute & Assert
        with pytest.raises(Exception) as exc_info:
            adapter.generate_diarization_map(loaded_model, "/fake/audio.wav", vad_map, source_id)

        # Prove the original error was bubbled up for the error mapper to handle
        assert "CUDA out of memory" in str(exc_info.value)

    @patch("torchaudio.load")
    @patch.object(NeMoDiarizationAdapter, '_get_diarizer')
    @patch("app.features.diarization.data.nemo_adapter.Path.exists")
    @patch("builtins.open")
    @patch("tempfile.TemporaryDirectory")
    def test_generate_diarization_map_fallback_on_missing_rttm(
            self, mock_temp_dir, mock_file_open, mock_path_exists, mock_get_diarizer, mock_ta_load, adapter, manifest
    ):
        """
        Verifies that if NeMo successfully completes but fails to output the resulting .rttm file,
        we also fall back gracefully.
        """
        source_id = uuid.uuid4()
        loaded_model = LoadedModel(manifest=manifest, model_instance="/fake/path/titanet.nemo")
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=2.0, end_sec=4.0, confidence=0.95)])

        # Mock standard Mono audio
        mock_ta_load.return_value = (torch.zeros(1, 16000), 16000)

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