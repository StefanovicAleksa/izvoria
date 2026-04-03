# Canvas: test_detection_api_unit_v1
# Path: tests/features/visual_cue_detection/service/test_detection_api_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI
from app.features.visual_cue_detection.exceptions.detection import LLMCueGenerationError
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.core.model_manager.domain.entities import LoadedModel


class TestVisualCueDetectionAPIUnit:
    @pytest.fixture
    def mock_orchestrator(self): return Mock()

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def mock_repo(self): return Mock()

    @pytest.fixture
    def api(self, mock_orchestrator, mock_adapter, mock_repo):
        return VisualCueDetectionAPI(
            orchestrator=mock_orchestrator,
            adapter=mock_adapter,
            repository=mock_repo
        )

    def test_process_transcript_bubbles_errors_strictly(self, api, mock_adapter):
        """Verifies that chunk-level SLM failures bubble up and fail the job, rather than silently falling back."""
        transcript = ContextualizedTranscript(
            id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            chunks=[
                ContextualizedChunk(
                    chunk_id=uuid.uuid4(),
                    original_text="test",
                    context_string="test",
                    needs_visual=False
                )
            ]
        )

        # Force the Llama adapter to simulate an out-of-memory or context length crash
        mock_adapter.detect_cues.side_effect = RuntimeError("llama_cpp: context window exceeded")

        mock_model = Mock(spec=LoadedModel)

        with pytest.raises(LLMCueGenerationError) as exc_info:
            api.process_transcript(transcript=transcript, model=mock_model, release_after=False)

        assert "context window exceeded" in str(exc_info.value).lower()