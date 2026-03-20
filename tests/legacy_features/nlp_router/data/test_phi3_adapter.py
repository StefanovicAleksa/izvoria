# Canvas: test_phi3_adapter_v3
# Path: tests/features/nlp_router/test_phi3_adapter.py

import json
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.legacy_features.nlp_router.domain.entities import TranscriptChunk
from app.legacy_features.nlp_router.exceptions.base import NLPRouterError
from app.legacy_features.nlp_router.data.phi3_adapter import Phi3RouterAdapter


class TestPhi3RouterAdapter:

    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.LLM,
            model_path="dummy_hf_path",
            required_vram_mb=3500.0
        )

    def test_extract_visual_cues_success_first_try(self, manifest):
        """Verifies the adapter succeeds immediately when valid JSON is returned."""
        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 2)
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs
        mock_model.generate.return_value = [[0, 0, 1, 2, 3]]

        fake_json_string = json.dumps({
            "reasoning": "Speaker pointed at chart.",
            "windows": [{"start_sec": 10.5, "end_sec": 15.0}]
        })
        mock_tokenizer.decode.return_value = f"```json\n{fake_json_string}\n```"

        payload = {"model": mock_model, "tokenizer": mock_tokenizer}
        loaded_model = LoadedModel(manifest=manifest, model_instance=payload)

        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Look at this.")

        result = adapter.extract_visual_cues(loaded_model, chunk)

        assert len(result.windows) == 1
        assert mock_model.generate.call_count == 1

    def test_extract_visual_cues_retry_recovers_from_hallucination(self, manifest):
        """Verifies the adapter retries and recovers if the first attempt hallucinates bad JSON."""
        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 2)
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs
        mock_model.generate.return_value = [[0, 0, 1, 2, 3]]

        bad_json = "I am a conversational hallucination, here is your answer!"
        good_json = json.dumps({"reasoning": "Fixed.", "windows": []})

        # side_effect allows us to return different values on consecutive calls
        mock_tokenizer.decode.side_effect = [bad_json, good_json]

        payload = {"model": mock_model, "tokenizer": mock_tokenizer}
        loaded_model = LoadedModel(manifest=manifest, model_instance=payload)

        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Text")

        result = adapter.extract_visual_cues(loaded_model, chunk)

        assert len(result.windows) == 0
        assert mock_model.generate.call_count == 2  # It should have retried exactly once

    def test_extract_visual_cues_exhausts_retries(self, manifest):
        """Verifies the adapter correctly raises an exception if it exhausts all 3 retries."""
        mock_model = MagicMock()
        mock_model.device = "cuda"
        mock_tokenizer = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.shape = (1, 2)
        mock_tokenizer.apply_chat_template.return_value.to.return_value = mock_inputs
        mock_model.generate.return_value = [[0, 0, 1, 2, 3]]

        # Always return bad JSON
        mock_tokenizer.decode.return_value = "Forever hallucinating..."

        payload = {"model": mock_model, "tokenizer": mock_tokenizer}
        loaded_model = LoadedModel(manifest=manifest, model_instance=payload)

        adapter = Phi3RouterAdapter()
        chunk = TranscriptChunk(start_sec=10.0, end_sec=20.0, text_block="[10.0s] Text")

        with pytest.raises(NLPRouterError, match="Failed after 3 retries"):
            adapter.extract_visual_cues(loaded_model, chunk)

        assert mock_model.generate.call_count == 3  # Hit the absolute limit