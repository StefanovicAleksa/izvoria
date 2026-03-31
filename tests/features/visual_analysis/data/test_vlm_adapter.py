# Canvas: test_visual_analysis_vlm_adapter_v1
# Path: tests/features/visual_analysis/data/test_vlm_adapter.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_analysis.data.vlm_adapter import LlamaVLMAdapter
from app.features.visual_analysis.exceptions.model import ImageDecodeError
from app.features.visual_analysis.exceptions.base import VisualAnalysisError


class TestLlamaVLMAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.VISION,
            model_path="/mock/vault/models/llava.gguf",
            required_vram_mb=6500.0
        )

    @patch("app.features.visual_analysis.data.vlm_adapter.os.path.exists", return_value=True)
    @patch.object(LlamaVLMAdapter, "_get_llava_model")
    def test_load_success(self, mock_get_llava, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_llava.return_value = mock_instance

        adapter = LlamaVLMAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        # Verify it passed the base weights path to the factory
        mock_get_llava.assert_called_once_with(manifest.model_path, "models/llava-1.5-7b/mmproj-model-f16.gguf")

    @patch("app.features.visual_analysis.data.vlm_adapter.os.path.exists", return_value=False)
    def test_load_missing_file_raises(self, mock_exists, manifest):
        adapter = LlamaVLMAdapter()

        # Should raise our mapped domain error
        with pytest.raises(VisualAnalysisError):
            adapter.load(manifest)

    @patch("app.features.visual_analysis.data.vlm_adapter.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_jpeg_bytes")
    def test_analyze_frame_success(self, mock_file, mock_exists, manifest):
        adapter = LlamaVLMAdapter()
        mock_llm = MagicMock()

        # Simulate llama-cpp multimodal chat completion response
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": " The image shows a bar chart of Q3 revenue. "}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.analyze_frame(
            model=loaded_model,
            image_path=Path("/mock/vault/frame_1.jpg"),
            original_text="Look at Q3.",
            context_string="Financial review."
        )

        # 1. Assert standard string cleaning
        assert result == "The image shows a bar chart of Q3 revenue."

        # 2. Assert the payload formatting (OpenAI Vision standard)
        mock_llm.create_chat_completion.assert_called_once()
        call_kwargs = mock_llm.create_chat_completion.call_args[1]

        assert "messages" in call_kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Verify both text and base64 image were passed in the content array
        user_content = messages[1]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "text"
        assert "Look at Q3." in user_content[0]["text"]
        assert user_content[1]["type"] == "image_url"
        assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @patch("app.features.visual_analysis.data.vlm_adapter.Path.exists", return_value=False)
    def test_analyze_frame_missing_image_raises(self, mock_exists, manifest):
        adapter = LlamaVLMAdapter()
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        with pytest.raises(ImageDecodeError, match="missing from disk"):
            adapter.analyze_frame(
                model=loaded_model,
                image_path=Path("/mock/vault/ghost_frame.jpg"),
                original_text="text",
                context_string="context"
            )