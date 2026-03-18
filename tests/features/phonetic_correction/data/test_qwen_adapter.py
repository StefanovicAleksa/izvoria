import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.phonetic_correction.data.qwen_adapter import QwenAdapter


class TestQwenAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(model_type=ModelType.ASR_CORRECTION_PHASE_3, model_path="/fake/qwen.gguf",
                             required_vram_mb=4000)

    @patch("os.path.exists", return_value=True)
    @patch("app.features.phonetic_correction.data.qwen_adapter.Llama")
    def test_load_success(self, mock_llama, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_llama.return_value = mock_instance

        adapter = QwenAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_llama.assert_called_once_with(model_path=manifest.model_path, n_gpu_layers=-1, n_ctx=8192, verbose=False)

    def test_generate_patches_success(self, manifest):
        adapter = QwenAdapter()
        mock_llm = MagicMock()

        # Simulate C++ JSON engine response
        mock_llm.create_chat_completion.return_value = {
            "choices": [
                {"message": {"content": '{"corrections": [{"original_text": "bad", "corrected_text": "good"}]}'}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        assert len(patches) == 1
        assert patches[0].original == "bad"
        assert patches[0].correction == "good"