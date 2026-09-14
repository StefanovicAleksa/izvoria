# Canvas: test_detection_api_unit_v2
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

    def test_process_transcript_skips_a_fully_evaluated_transcript(self, api, mock_adapter, mock_orchestrator,
                                                                  mock_repo):
        transcript = ContextualizedTranscript(chunks=[
            ContextualizedChunk(original_text="a", needs_visual=True, cue_evaluated=True),
            ContextualizedChunk(original_text="b", needs_visual=False, cue_evaluated=True),
        ])

        result = api.process_transcript(transcript=transcript)

        assert result is transcript
        mock_orchestrator.acquire_model.assert_not_called()
        mock_adapter.detect_cues.assert_not_called()
        mock_repo.update.assert_not_called()

    def test_process_transcript_evaluates_only_pending_chunks(self, api, mock_adapter, mock_orchestrator, mock_repo):
        evaluated = ContextualizedChunk(original_text="a", needs_visual=True, cue_evaluated=True)
        pending = ContextualizedChunk(original_text="b")
        transcript = ContextualizedTranscript(chunks=[evaluated, pending])

        mock_adapter.detect_cues.side_effect = lambda model, chunk: chunk.model_copy(update={"needs_visual": True})
        mock_model = Mock(spec=LoadedModel)

        result = api.process_transcript(transcript=transcript, model=mock_model, release_after=False)

        mock_adapter.detect_cues.assert_called_once_with(mock_model, pending)
        assert result.chunks[0] is evaluated
        assert (result.chunks[1].chunk_id, result.chunks[1].needs_visual, result.chunks[1].cue_evaluated) == (
            pending.chunk_id, True, True
        )
        mock_repo.update.assert_called_once_with(transcript)
        mock_orchestrator.acquire_model.assert_not_called()
        mock_orchestrator.release_model.assert_not_called()
