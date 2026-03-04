# Canvas: test_phi3_adapter_v2
# Path: tests/features/nlp_router/test_phi3_adapter.py

import json
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.nlp_router.domain.entities import TranscriptChunk
from app.features.nlp_router.exceptions.base import NLPRouterError
from app.features.nlp_router.data.phi3_adapter import Phi3RouterAdapter


class TestPhi3RouterAdapter:

    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path="dummy_hf_path",
            required_vram_mb=3500.0
        )

    def test_extract_visual_cues_success(self, manifest):
        """Verifies the adapter correctly parses the tensor outputs and decodes valid JSON."""
        # 1. Setup HF Mocks
        mock_model = MagicMock()
        mock_model.device = "cuda"

        mock_tokenizer = MagicMock()

        # Mock the input tensor shapes (Batch size 1, 2 input tokens)
        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 2)
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs

        # Mock the output tensor (Length 5, so generated tokens will be the last 3)
        mock_model.generate.return_value = [[0, 0, 1, 2, 3]]

        # Mock decode to return valid JSON with or without Markdown wrappers
        fake_json_string = json.dumps({
            "reasoning": "Speaker pointed at chart.",
            "windows": [{"start_sec": 10.5, "end_sec": 15.0}]
        })
        mock_tokenizer.decode.return_value = f"```json\n{fake_json_string}\n```"

        payload = {"model": mock_model, "tokenizer": mock_tokenizer}
        loaded_model = LoadedModel(manifest=manifest, model_instance=payload)

        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Look at this.")

        # 2. Execute
        result = adapter.extract_visual_cues(loaded_model, chunk)

        # 3. Assertions
        assert len(result.windows) == 1
        assert result.windows[0].start_sec == 10.5
        assert result.reasoning == "Speaker pointed at chart."

    def test_extract_visual_cues_catches_hallucinations(self, manifest):
        """Verifies Pydantic ValidationErrors are caught and mapped if the LLM hallucination breaks schema."""
        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 2)
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs
        mock_model.generate.return_value = [[0, 0, 1, 2, 3]]

        # Mock decode returning an invalid schema (string instead of list for windows)
        bad_json_string = json.dumps({
            "reasoning": "I am confused.",
            "windows": "I am a string, not a list of TimeWindows"
        })
        mock_tokenizer.decode.return_value = bad_json_string

        payload = {"model": mock_model, "tokenizer": mock_tokenizer}
        loaded_model = LoadedModel(manifest=manifest, model_instance=payload)

        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Text")

        with pytest.raises(NLPRouterError):
            adapter.extract_visual_cues(loaded_model, chunk)