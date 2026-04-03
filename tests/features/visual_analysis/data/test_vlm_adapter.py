# Canvas: test_visual_analysis_vlm_adapter_v3
# Path: tests/features/visual_analysis/data/test_vlm_adapter.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.exceptions.base import VisualAnalysisError


class TestQwenVLMAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.VISION,
            model_path="/mock/vault/models/qwen.gguf",
            required_vram_mb=7500.0
        )

    # FIXED: Replaced `_get_llava_model` with `_get_qwen_model`
    @patch("app.features.visual_analysis.data.vlm_adapter.os.path.exists", return_value=True)
    @patch.object(QwenVLMAdapter, "_get_qwen_model")
    def test_load_success(self, mock_get_qwen, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_qwen.return_value = mock_instance

        adapter = QwenVLMAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_qwen.assert_called_once_with(manifest.model_path, settings.VISION_MMPROJ_PATH)

    @patch("app.features.visual_analysis.data.vlm_adapter.os.path.exists", return_value=False)
    def test_load_missing_file_raises(self, mock_exists, manifest):
        adapter = QwenVLMAdapter()
        with pytest.raises(VisualAnalysisError):
            adapter.load(manifest)

    @patch("app.features.visual_analysis.data.vlm_adapter.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_jpeg_bytes")
    def test_analyze_frame_success(self, mock_file, mock_exists, manifest):
        adapter = QwenVLMAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "The image shows a bar chart of Q3 revenue."}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.analyze_frame(
            model=loaded_model,
            image_path=Path("/mock/vault/frame_1.jpg"),
            original_text="Look at Q3.",
            context_string="Financial review."
        )

        assert result == "The image shows a bar chart of Q3 revenue."
        mock_llm.create_chat_completion.assert_called_once()

    @patch("app.features.visual_analysis.data.vlm_adapter.Path.exists", return_value=False)
    def test_analyze_frame_missing_image_raises(self, mock_exists, manifest):
        adapter = QwenVLMAdapter()
        loaded_model = LoadedModel(manifest=manifest, model_instance=MagicMock())

        with pytest.raises(VisualAnalysisError):
            adapter.analyze_frame(loaded_model, Path("/fake/path.jpg"), "text", "context")