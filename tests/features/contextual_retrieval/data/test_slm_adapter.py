# Canvas: test_contextual_slm_adapter_v1
# Path: tests/features/contextual_retrieval/data/test_slm_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.contextual_retrieval.data.slm_adapter import SLMContextAdapter
from app.features.contextual_retrieval.exceptions.model import SLMModelLoadError


class TestSLMContextAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.DOMAIN_CLASSIFIER,
            model_path="/fake/llama.gguf",
            required_vram_mb=3500
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(SLMContextAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_llama, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_llama.return_value = mock_instance

        adapter = SLMContextAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_llama.assert_called_once_with(manifest.model_path)

    @patch("os.path.exists", return_value=False)
    def test_load_missing_file_raises(self, mock_exists, manifest):
        adapter = SLMContextAdapter()
        with pytest.raises(SLMModelLoadError):
            adapter.load(manifest)

    def test_generate_context_success_and_cleans_filler(self, manifest):
        adapter = SLMContextAdapter()
        mock_llm = MagicMock()

        # Simulate LLM returning the context but breaking the rule and adding conversational filler
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Here is the context: Dr. Smith is talking."}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.generate_context(loaded_model, "Full document text...", "Chunk text")

        # The adapter should strip the filler
        assert result == "Dr. Smith is talking."

        mock_llm.create_chat_completion.assert_called_once()
        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["temperature"] == 0.0

    def test_count_tokens_uses_model_tokenizer_without_bos(self, manifest):
        adapter = SLMContextAdapter()
        mock_llm = MagicMock()
        mock_llm.tokenize.return_value = [1, 2, 3, 4]
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert adapter.count_tokens(loaded_model, "Doktor Å") == 4
        mock_llm.tokenize.assert_called_once_with("Doktor Å".encode("utf-8"), add_bos=False)

    @pytest.mark.parametrize("raw_output, expected", [
        ("[Speaker_0 @ 00:00:00]: The patient reports chest pain.", "The patient reports chest pain."),
        ("[Speaker_0 @ 00:00:00]", ""),
        ("Speaker_1: Dr. Smith explains the MRI.", "Dr. Smith explains the MRI."),
        ("speaker_0: The Bund is testing support.", "The Bund is testing support."),
        ("[speaker_0 @ 00:32:43]: The range holds.", "The range holds."),
        ("Speaker_0 describes chest pain during intake.", "Speaker_0 describes chest pain during intake."),
    ])
    def test_generate_context_strips_echoed_speaker_labels(self, manifest, raw_output, expected):
        adapter = SLMContextAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": raw_output}}]}
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert adapter.generate_context(loaded_model, "Speaker_0: doc", "Speaker_0: chunk") == expected
