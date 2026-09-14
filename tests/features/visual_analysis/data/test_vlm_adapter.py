# Canvas: test_visual_analysis_vlm_adapter_v5
# Path: tests/features/visual_analysis/data/test_vlm_adapter.py

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.exceptions.base import VisualAnalysisError

MODULE = "app.features.visual_analysis.data.vlm_adapter"


class TestQwenVLMAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.VISION,
            model_path="/mock/vault/models/qwen2-vl-7b-hf",
            required_vram_mb=9000.0
        )

    @staticmethod
    def _loaded_model(manifest, decoded_text):
        mock_hf_model = MagicMock()
        mock_hf_model.device = "cuda"
        mock_hf_model.generate.return_value = [[1, 2, 3, 4, 5]]

        mock_inputs = MagicMock()
        mock_inputs.input_ids = [[1, 2, 3]]
        mock_inputs.to.return_value = mock_inputs

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "<prompt>"
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = [decoded_text]

        loaded_model = LoadedModel(manifest=manifest, model_instance={"model": mock_hf_model, "processor": mock_processor})
        return loaded_model, mock_hf_model, mock_processor

    @patch(f"{MODULE}.os.path.exists", return_value=True)
    @patch.object(QwenVLMAdapter, "_get_qwen_model")
    def test_load_success(self, mock_get_qwen, mock_exists, manifest):
        mock_payload = {"model": MagicMock(), "processor": MagicMock()}
        mock_get_qwen.return_value = mock_payload

        adapter = QwenVLMAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_payload
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    @patch(f"{MODULE}.os.path.exists", return_value=False)
    def test_load_missing_file_raises(self, mock_exists, manifest):
        adapter = QwenVLMAdapter()
        with pytest.raises(VisualAnalysisError):
            adapter.load(manifest)

    @patch(f"{MODULE}.Path.exists", return_value=True)
    @patch(f"{MODULE}.Image")
    @patch(f"{MODULE}.torch")
    def test_analyze_frames_single_frame(self, mock_torch, mock_image_module, mock_exists, manifest):
        loaded_model, mock_hf_model, mock_processor = self._loaded_model(manifest, "Bar chart titled 'Q3 Revenue'.")
        image = mock_image_module.open.return_value.__enter__.return_value.convert.return_value

        result = QwenVLMAdapter().analyze_frames(
            model=loaded_model,
            image_paths=[Path("/mock/vault/frame_1.jpg")],
            original_text="Look at Q3.",
            context_string="Financial review."
        )

        assert result == ["Bar chart titled 'Q3 Revenue'."]
        image.resize.assert_not_called()
        mock_processor.assert_called_once_with(text=["<prompt>"], images=[image], return_tensors="pt")
        mock_hf_model.generate.assert_called_once_with(
            max_new_tokens=settings.VISION_MAX_NEW_TOKENS_PER_FRAME, do_sample=False
        )

        messages = mock_processor.apply_chat_template.call_args[0][0]
        user_content = messages[1]["content"]
        assert user_content[0] == {"type": "image"}
        assert "Speaker's Exact Words: Look at Q3." in user_content[1]["text"]
        assert "Frame 1: <description>" in user_content[1]["text"]

    @patch(f"{MODULE}.Path.exists", return_value=True)
    @patch(f"{MODULE}.Image")
    @patch(f"{MODULE}.torch")
    def test_analyze_frames_describes_several_frames_in_one_call(self, mock_torch, mock_image_module, mock_exists,
                                                                 manifest):
        loaded_model, mock_hf_model, mock_processor = self._loaded_model(
            manifest,
            "Frame 1: Title slide reading 'Q3 Results'.\nFrame 2: Bar chart; the y-axis values are unreadable."
        )
        image = mock_image_module.open.return_value.__enter__.return_value.convert.return_value
        image.size = (1920, 1080)

        result = QwenVLMAdapter().analyze_frames(
            model=loaded_model,
            image_paths=[Path("/mock/vault/a.jpg"), Path("/mock/vault/b.jpg")],
            original_text="Revenue went up.",
            context_string="Quarterly review."
        )

        assert result == ["Title slide reading 'Q3 Results'.", "Bar chart; the y-axis values are unreadable."]

        # Each frame is downscaled to the shared multi-image pixel budget
        scale = (settings.VISION_MULTI_IMAGE_MAX_PIXELS / (1920 * 1080)) ** 0.5
        assert image.resize.call_count == 2
        image.resize.assert_called_with((int(1920 * scale), int(1080 * scale)), mock_image_module.Resampling.BICUBIC)
        mock_processor.assert_called_once_with(
            text=["<prompt>"], images=[image.resize.return_value, image.resize.return_value], return_tensors="pt"
        )

        user_content = mock_processor.apply_chat_template.call_args[0][0][1]["content"]
        assert user_content[:2] == [{"type": "image"}, {"type": "image"}]
        assert "You are given 2 frame(s)" in user_content[2]["text"]
        mock_hf_model.generate.assert_called_once_with(
            max_new_tokens=settings.VISION_MAX_NEW_TOKENS_PER_FRAME * 2, do_sample=False
        )

    def test_analyze_frames_with_no_frames_skips_the_model(self, manifest):
        loaded_model, mock_hf_model, mock_processor = self._loaded_model(manifest, "")

        assert QwenVLMAdapter().analyze_frames(loaded_model, [], "text", "context") == []
        mock_hf_model.generate.assert_not_called()

    @patch(f"{MODULE}.Path.exists", return_value=False)
    def test_analyze_frames_missing_image_raises(self, mock_exists, manifest):
        loaded_model, _, _ = self._loaded_model(manifest, "")

        with pytest.raises(VisualAnalysisError):
            QwenVLMAdapter().analyze_frames(loaded_model, [Path("/fake/path.jpg")], "text", "context")
