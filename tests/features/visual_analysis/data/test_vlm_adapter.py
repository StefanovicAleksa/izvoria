# Canvas: test_visual_analysis_vlm_adapter_v4
# Path: tests/features/visual_analysis/data/test_vlm_adapter.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.exceptions.base import VisualAnalysisError


class TestQwenVLMAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.VISION,
            model_path="/mock/vault/models/qwen2-vl-7b-hf",
            required_vram_mb=9000.0
        )

    @patch("app.features.visual_analysis.data.vlm_adapter.os.path.exists", return_value=True)
    @patch.object(QwenVLMAdapter, "_get_qwen_model")
    def test_load_success(self, mock_get_qwen, mock_exists, manifest):
        mock_payload = {"model": MagicMock(), "processor": MagicMock()}
        mock_get_qwen.return_value = mock_payload

        adapter = QwenVLMAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_payload
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    @patch("app.features.visual_analysis.data.vlm_adapter.os.path.exists", return_value=False)
    def test_load_missing_file_raises(self, mock_exists, manifest):
        adapter = QwenVLMAdapter()
        with pytest.raises(VisualAnalysisError):
            adapter.load(manifest)

    @patch("app.features.visual_analysis.data.vlm_adapter.Path.exists", return_value=True)
    @patch("app.features.visual_analysis.data.vlm_adapter.Image")
    @patch("app.features.visual_analysis.data.vlm_adapter.torch")
    def test_analyze_frame_success(self, mock_torch, mock_image_module, mock_exists, manifest):
        adapter = QwenVLMAdapter()

        mock_hf_model = MagicMock()
        mock_hf_model.device = "cuda"

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "<prompt>"

        mock_inputs = MagicMock()
        mock_inputs.input_ids = [[1, 2, 3]]
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_generated_ids = [[1, 2, 3, 4, 5]]
        mock_hf_model.generate.return_value = mock_generated_ids
        mock_processor.batch_decode.return_value = ["The image shows a bar chart of Q3 revenue."]

        # torch.inference_mode() is used as a context manager; mock it as a no-op.
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock(return_value=False)

        loaded_model = LoadedModel(
            manifest=manifest,
            model_instance={"model": mock_hf_model, "processor": mock_processor}
        )

        result = adapter.analyze_frame(
            model=loaded_model,
            image_path=Path("/mock/vault/frame_1.jpg"),
            original_text="Look at Q3.",
            context_string="Financial review."
        )

        assert result == "The image shows a bar chart of Q3 revenue."
        mock_hf_model.generate.assert_called_once()
        mock_processor.apply_chat_template.assert_called_once()

    @patch("app.features.visual_analysis.data.vlm_adapter.Path.exists", return_value=False)
    def test_analyze_frame_missing_image_raises(self, mock_exists, manifest):
        adapter = QwenVLMAdapter()
        loaded_model = LoadedModel(
            manifest=manifest,
            model_instance={"model": MagicMock(), "processor": MagicMock()}
        )

        with pytest.raises(VisualAnalysisError):
            adapter.analyze_frame(loaded_model, Path("/fake/path.jpg"), "text", "context")
