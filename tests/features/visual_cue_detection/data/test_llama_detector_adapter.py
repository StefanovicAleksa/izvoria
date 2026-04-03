# Canvas: test_llama_detector_adapter_v2
# Path: tests/features/visual_cue_detection/data/test_llama_detector_adapter.py

import uuid
import json
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.visual_cue_detection.data.qwen_detector_adapter import QwenDetectorAdapter
from app.features.visual_cue_detection.exceptions.detection import CueParseError


class TestQwenDetectorAdapter:
    @pytest.fixture
    def adapter(self):
        return QwenDetectorAdapter()

    @pytest.fixture
    def mock_model(self):
        model = MagicMock(spec=LoadedModel)
        llm_instance = MagicMock()
        model.model_instance = llm_instance
        return model, llm_instance

    @pytest.fixture
    def sample_chunk(self):
        return ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            original_text="As you can see on this chart, we've hit a peak of volume.",
            context_string="A technical presentation.",
            needs_visual=False  # Starts false
        )

    def test_detect_cues_happy_path_true(self, adapter, mock_model, sample_chunk):
        model, llm = mock_model

        # Mock the CoT JSON response exactly as the schema requires
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "reasoning": "The speaker references a chart and a spatial peak of volume, requiring visual context.",
                            "needs_visual": True
                        })
                    }
                }
            ]
        }
        llm.create_chat_completion.return_value = mock_response

        updated_chunk = adapter.detect_cues(model, sample_chunk)

        assert updated_chunk.needs_visual is True
        assert updated_chunk.chunk_id == sample_chunk.chunk_id

        # Verify it passed the required parameters
        call_kwargs = llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 250
        assert "response_format" in call_kwargs

    def test_detect_cues_happy_path_false(self, adapter, mock_model, sample_chunk):
        model, llm = mock_model

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "reasoning": "The speaker is just telling an anecdote without any visual evidence.",
                            "needs_visual": False
                        })
                    }
                }
            ]
        }
        llm.create_chat_completion.return_value = mock_response

        updated_chunk = adapter.detect_cues(model, sample_chunk)

        assert updated_chunk.needs_visual is False

    def test_detect_cues_invalid_json_throws_domain_error(self, adapter, mock_model, sample_chunk):
        model, llm = mock_model

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "This is just raw text, not JSON at all."
                    }
                }
            ]
        }
        llm.create_chat_completion.return_value = mock_response

        with pytest.raises(CueParseError) as exc_info:
            adapter.detect_cues(model, sample_chunk)

        assert "SLM failed to generate valid JSON" in str(exc_info.value)