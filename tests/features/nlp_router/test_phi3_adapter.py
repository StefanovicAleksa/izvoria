# Canvas: test_nlp_phi3_adapter_v3
# Path: tests/features/nlp_router/test_phi3_adapter.py

import pytest
import json
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.nlp_router.data.phi3_adapter import Phi3RouterAdapter
from app.features.nlp_router.domain.entities import TranscriptChunk
from app.features.nlp_router.exceptions.model import NLPRouterModelLoadError, NLPRouterJSONParseError


class TestPhi3RouterAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path="dummy_phi_path.gguf",
            required_vram_mb=3500.0
        )

    @patch("app.features.nlp_router.data.phi3_adapter.os.path.exists", return_value=False)
    def test_load_fails_if_file_missing(self, mock_exists, manifest):
        adapter = Phi3RouterAdapter()
        with pytest.raises(NLPRouterModelLoadError):
            adapter.load(manifest)

        mock_exists.assert_called_once_with(manifest.model_path)

    # FIXED: Removed unused @patch decorators for os.path and Llama
    def test_extract_visual_cues_success(self, manifest):
        # 1. Setup Mock LLM returning valid JSON string
        mock_llm_instance = MagicMock()
        fake_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "reasoning": "Speaker pointed at chart.",
                        "windows": [{"start_sec": 10.5, "end_sec": 15.0}]
                    })
                }
            }]
        }
        mock_llm_instance.create_chat_completion.return_value = fake_response

        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm_instance)
        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Look at this.")

        # 2. Execute
        result = adapter.extract_visual_cues(loaded_model, chunk)

        # 3. Assert
        assert result.reasoning == "Speaker pointed at chart."
        assert len(result.windows) == 1
        assert result.windows[0].start_sec == 10.5

        # Verify strict typing enforcement was passed to Llama
        mock_llm_instance.create_chat_completion.assert_called_once()
        called_kwargs = mock_llm_instance.create_chat_completion.call_args[1]
        assert called_kwargs["response_format"]["type"] == "json_object"

    # FIXED: Removed unused patches
    def test_extract_visual_cues_catches_hallucinations(self, manifest):
        """Verifies Pydantic ValidationErrors are mapped if the LLM hallucination breaks schema."""
        mock_llm_instance = MagicMock()

        # FIXED: Pass a string to 'windows' instead of omitting it.
        # Omission triggers default_factory=list, but a string triggers a strict ValidationError.
        fake_response = {
            "choices": [{"message": {"content": json.dumps({
                "reasoning": "I am confused.",
                "windows": "I am a string, not a list of TimeWindows"
            })}}]
        }
        mock_llm_instance.create_chat_completion.return_value = fake_response

        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm_instance)
        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Text")

        with pytest.raises(NLPRouterJSONParseError):
            adapter.extract_visual_cues(loaded_model, chunk)