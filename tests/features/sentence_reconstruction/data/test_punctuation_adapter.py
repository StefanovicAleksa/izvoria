# Canvas: test_punctuation_adapter_v3
# Path: tests/features/sentence_reconstruction/data/test_punctuation_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord
from app.features.sentence_reconstruction.data.punctuation_adapter import PunctuationAdapter
from app.features.sentence_reconstruction.exceptions.model import PunctuationModelLoadError


class TestPunctuationAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.SENTENCE_RECONSTRUCTION,
            model_path="/mock/path/fullstop",
            required_vram_mb=500.0
        )

    @patch("app.features.sentence_reconstruction.data.punctuation_adapter.pipeline")
    @patch("os.path.exists")
    def test_load_success(self, mock_exists, mock_pipeline, manifest):
        mock_exists.return_value = True
        mock_pipe_instance = MagicMock()
        mock_pipeline.return_value = mock_pipe_instance

        adapter = PunctuationAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == mock_pipe_instance

        # CRITICAL FIX: Assertions updated to match actual adapter changes
        mock_pipeline.assert_called_once_with(
            task="token-classification",
            model=manifest.model_path,
            tokenizer=manifest.model_path,
            device=0,
            aggregation_strategy="simple",
            model_kwargs={"local_files_only": True}
        )

    @patch("os.path.exists")
    def test_load_missing_file_raises_error(self, mock_exists, manifest):
        mock_exists.return_value = False
        adapter = PunctuationAdapter()

        with pytest.raises(PunctuationModelLoadError):
            adapter.load(manifest)

    def test_predict_punctuation_success(self, manifest):
        adapter = PunctuationAdapter()
        mock_pipe_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_pipe_instance)

        # 1. Input: Exact acoustic words
        input_words = [
            TranscriptWord(start_sec=0.0, end_sec=0.5, word="Hello", probability=0.99),
            TranscriptWord(start_sec=0.5, end_sec=1.0, word="world", probability=0.99)
        ]

        # 2. Mock Pipeline Output
        mock_pipe_instance.return_value = [
            {"entity_group": "0", "word": "Hello", "start": 0, "end": 5},
            {"entity_group": ".", "word": "world", "start": 6, "end": 11}
        ]

        # 3. Execute
        result = adapter.predict_punctuation(loaded_model, input_words)

        # 4. Assertions
        assert mock_pipe_instance.call_args[0][0] == "Hello world"

        assert len(result) == 2
        assert isinstance(result[0], PunctuatedWord)

        # Verify "0" is correctly sanitized to "NONE"
        assert result[0].acoustic_word.word == "Hello"
        assert result[0].punctuation_tag == "NONE"

        # Verify terminal punctuation is captured
        assert result[1].acoustic_word.word == "world"
        assert result[1].punctuation_tag == "."

    def test_predict_punctuation_handles_subword_overlaps(self, manifest):
        """Verifies our character-offset logic works even if Hugging Face returns weird subword bounds."""
        adapter = PunctuationAdapter()
        mock_pipe_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_pipe_instance)

        # Input: 1 long word
        input_words = [TranscriptWord(start_sec=0.0, end_sec=1.0, word="Echocardiogram", probability=0.99)]

        # Mock Pipeline Output: It predicts the tag only overlaps the last few characters
        mock_pipe_instance.return_value = [
            {"entity_group": "?", "word": "gram", "start": 10, "end": 14}
        ]

        result = adapter.predict_punctuation(loaded_model, input_words)

        assert len(result) == 1
        assert result[0].acoustic_word.word == "Echocardiogram"

        # Because chars 10-14 overlap with the acoustic word's 0-14 bounds, it should correctly map the "?"
        assert result[0].punctuation_tag == "?"