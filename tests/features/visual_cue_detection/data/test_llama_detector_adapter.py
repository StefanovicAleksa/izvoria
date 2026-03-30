# Canvas: test_llama_detector_adapter_v1
# Path: tests/features/visual_cue_detection/data/test_llama_detector_adapter.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.visual_cue_detection.data.llama_adapter import LlamaDetectorAdapter
from app.features.visual_cue_detection.exceptions.detection import CueParseError


class TestLlamaDetectorAdapter:
    @pytest.fixture
    def adapter(self):
        return LlamaDetectorAdapter()

    @pytest.fixture
    def mock_model(self):
        # Mock the wrapper and the underlying Llama C++ instance
        model = MagicMock(spec=LoadedModel)
        llm_instance = MagicMock()
        model.model_instance = llm_instance
        return model, llm_instance

    @pytest.fixture
    def sample_chunk(self):
        return ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            original_text="Look at this chart.",
            context_string="A financial presentation.",
            needs_visual=False  # Starts false
        )

    def test_detect_cues_happy_path_true(self, adapter, mock_model, sample_chunk):
        model, llm_instance = mock_model

        # Simulate Llama-cpp returning valid JSON that flags the visual cue
        llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"needs_visual": true}'}}]
        }

        updated_chunk = adapter.detect_cues(model, sample_chunk)

        # Assertions
        assert updated_chunk.chunk_id == sample_chunk.chunk_id
        assert updated_chunk.needs_visual is True

        # Verify the schema enforcement was passed in the API call
        llm_instance.create_chat_completion.assert_called_once()
        call_kwargs = llm_instance.create_chat_completion.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"
        assert "schema" in call_kwargs["response_format"]

    def test_detect_cues_happy_path_false(self, adapter, mock_model, sample_chunk):
        model, llm_instance = mock_model

        # Simulate Llama-cpp returning valid JSON that denies the visual cue
        llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"needs_visual": false}'}}]
        }

        updated_chunk = adapter.detect_cues(model, sample_chunk)

        assert updated_chunk.needs_visual is False

    def test_detect_cues_maps_json_failure(self, adapter, mock_model, sample_chunk):
        model, llm_instance = mock_model

        # Simulate Llama-cpp ignoring the grammar and returning a broken string
        llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "I am an AI, I don't know."}}]
        }

        # The JSONDecodeError should be caught and mapped by our Error Mapper
        with pytest.raises(CueParseError):
            adapter.detect_cues(model, sample_chunk)