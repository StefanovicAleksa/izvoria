# Canvas: test_qwen_adapter_v6
# Path: tests/features/phonetic_correction/data/test_qwen_adapter.py

import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.phonetic_correction.data.qwen_adapter import QwenAdapter


class TestQwenAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.ASR_CORRECTION_PHASE_3,
            model_path="/fake/qwen.gguf",
            required_vram_mb=4000
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(QwenAdapter, "_get_qwen_model")
    def test_load_success(self, mock_get_qwen, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_qwen.return_value = mock_instance

        adapter = QwenAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    def test_generate_patches_success(self, manifest):
        adapter = QwenAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"corrections": [{"original_text": "bad", "corrected_text": "good"}]}'}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        assert len(patches) == 1
        assert patches[0].original == "bad"
        assert patches[0].correction == "good"

    def test_generate_patches_graceful_degradation(self, manifest):
        adapter = QwenAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{ bad json'}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        assert len(patches) == 0

    def test_generate_patches_drops_hallucinated_falsy_values_to_prevent_deletion(self, manifest):
        """
        Ensures that if Qwen tries to use integers or nulls to represent an empty string,
        we aggressively drop the patch entirely instead of resolving it to "" which would
        delete the word from the transcript.
        """
        adapter = QwenAdapter()
        mock_llm = MagicMock()

        bad_json_content = '''{
            "corrections": [
                {"original_text": "subscribe", "corrected_text": 0},
                {"original_text": "like", "corrected_text": null},
                {"original_text": "comment", "corrected_text": false},
                {"original_text": "valid", "corrected_text": "real word"}
            ]
        }'''
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": bad_json_content}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        # The first three hallucinated patches should be safely dropped, leaving only the valid one.
        assert len(patches) == 1
        assert patches[0].original == "valid"
        assert patches[0].correction == "real word"

    @patch("app.features.phonetic_correction.data.qwen_adapter.gc.collect")
    def test_unload_explicitly_closes_c_bindings(self, mock_gc, manifest):
        adapter = QwenAdapter()
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded_model)

        assert result is True
        mock_llm.close.assert_called_once()
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()