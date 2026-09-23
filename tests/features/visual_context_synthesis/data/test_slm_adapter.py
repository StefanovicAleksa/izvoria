# Canvas: test_visual_synthesis_slm_adapter_v1
# Path: tests/features/visual_context_synthesis/data/test_slm_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_context_synthesis.data.slm_adapter import VisualSLMAdapter
from app.features.visual_context_synthesis.data.prompts import VISUAL_SYNTHESIS_SYSTEM_PROMPT
from app.features.visual_context_synthesis.exceptions.model import VisualSLMLoadError


class TestVisualSLMAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.VISUAL_CONTEXT_SYNTHESIS,
            model_path="/fake/llama_synthesis.gguf",
            required_vram_mb=3500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(VisualSLMAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_llama, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_llama.return_value = mock_instance

        adapter = VisualSLMAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_llama.assert_called_once_with(manifest.model_path)

    @patch("os.path.exists", return_value=False)
    def test_load_missing_file_raises(self, mock_exists, manifest):
        adapter = VisualSLMAdapter()
        with pytest.raises(VisualSLMLoadError):
            adapter.load(manifest)

    def test_synthesize_context_formats_timeline_and_calls_llm(self, manifest):
        adapter = VisualSLMAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "The video shows a line chart with a spike."}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        visual_events = [
            {"start_sec": 10.0, "end_sec": 20.0, "representative_description": "A line chart.", "frame_ids": []}
        ]

        result = adapter.synthesize_context(loaded_model, visual_events)

        assert result == "The video shows a line chart with a spike."

        # Verify the LLM was called with the deterministic parameters
        mock_llm.create_chat_completion.assert_called_once()
        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.0

        # Verify the timeline formatting
        prompt_content = call_kwargs["messages"][1]["content"]
        assert "- [10.00s - 20.00s]: A line chart." in prompt_content

    @patch("app.features.visual_context_synthesis.data.slm_adapter.gc.collect")
    def test_unload_explicitly_closes_c_bindings(self, mock_gc, manifest):
        adapter = VisualSLMAdapter()
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded_model)

        assert result is True
        mock_llm.close.assert_called_once()
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()

class TestSynthesisLanguage:
    def test_slovenian_appliance_asks_for_slovenian_context(self, appliance_language):
        """The VLM describes frames in English; the context stored beside a Slovenian transcript is Slovenian."""
        appliance_language("sl")
        llm = MagicMock(**{"create_chat_completion.return_value": {"choices": [{"message": {"content": "Graf."}}]}})
        manifest = ModelManifest(model_type=ModelType.VISUAL_CONTEXT_SYNTHESIS, model_path="/m.gguf",
                                 required_vram_mb=1)

        VisualSLMAdapter().synthesize_context(
            LoadedModel(manifest=manifest, model_instance=llm),
            [{"start_sec": 0.0, "end_sec": 5.0, "representative_description": "A line chart.", "frame_ids": []}]
        )

        system_prompt = llm.create_chat_completion.call_args.kwargs["messages"][0]["content"]
        assert system_prompt.startswith(VISUAL_SYNTHESIS_SYSTEM_PROMPT)
        assert "in Slovenian" in system_prompt
