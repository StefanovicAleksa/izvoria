import uuid
import pytest
from unittest.mock import MagicMock

from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI
from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationEvaluatedTranscript,
    VisualHallucinationAudit
)


class TestVisualHallucinationCorrectionAPI:
    @pytest.fixture
    def mock_repo(self): return MagicMock()

    @pytest.fixture
    def mock_adapter(self): return MagicMock()

    @pytest.fixture
    def mock_contextual_repo(self): return MagicMock()

    @pytest.fixture
    def mock_visual_repo(self): return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self): return MagicMock()

    @pytest.fixture
    def api(self, mock_repo, mock_adapter, mock_contextual_repo, mock_visual_repo, mock_orchestrator):
        return VisualHallucinationCorrectionAPI(
            repository=mock_repo, adapter=mock_adapter, contextual_repo=mock_contextual_repo,
            visual_analysis_repo=mock_visual_repo, orchestrator=mock_orchestrator
        )

    def test_execute_truncates_massive_visual_evidence(
            self, api, mock_repo, mock_adapter, mock_contextual_repo, mock_visual_repo
    ):
        """Verifies that heavily populated visual chunks are truncated for SLM safety."""
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        # Mock Transcript setup
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = chunk_id
        mock_chunk.needs_visual = True
        mock_chunk.original_text = "Short text."
        mock_transcript = MagicMock()
        mock_transcript.chunks = [mock_chunk]
        mock_contextual_repo.get_by_source_id.return_value = mock_transcript
        mock_repo.get_by_source_id.return_value = None

        # Simulate 15,000 characters of Visual Evidence
        massive_description = "A" * 15000
        mock_analysis = MagicMock()
        # Removed timestamp_sec to match entity
        mock_analysis.description = massive_description
        mock_visual_record = MagicMock()
        mock_visual_record.analyses = [mock_analysis]
        mock_visual_repo.get_by_chunk_id.return_value = mock_visual_record

        # Mock objects for the final result
        mock_audit = VisualHallucinationAudit(**{
            "chunk_id": chunk_id,
            "is_visual_accepted": False,
            "discard_reason": "Audit complete"
        })
        mock_adapter.evaluate_visual_match.return_value = mock_audit

        # Explicitly ensure mock save returns the entity to satisfy final check
        mock_repo.save.return_value = VisualHallucinationEvaluatedTranscript(source_id=source_id, audits=[mock_audit])

        # Execute
        result = api.execute(source_id)

        # Assertions
        assert isinstance(result, VisualHallucinationEvaluatedTranscript)
        adapter_call_args = mock_adapter.evaluate_visual_match.call_args[1]
        passed_visual_descriptions = adapter_call_args["visual_description"]

        assert len(passed_visual_descriptions) < 15000
        assert "[TRUNCATED TO PREVENT OOM]" in passed_visual_descriptions
    @staticmethod
    def _chunk(text: str, context: str = "") -> MagicMock:
        chunk = MagicMock()
        chunk.chunk_id = uuid.uuid4()
        chunk.needs_visual = True
        chunk.original_text = text
        chunk.context_string = context
        return chunk

    @staticmethod
    def _record(*descriptions: str) -> MagicMock:
        record = MagicMock()
        record.analyses = [MagicMock(description=d) for d in descriptions]
        return record

    def test_on_screen_text_sharing_the_topic_is_accepted_without_loading_the_model(
            self, api, mock_repo, mock_adapter, mock_contextual_repo, mock_visual_repo, mock_orchestrator
    ):
        source_id = uuid.uuid4()
        chunk = self._chunk("A 7-2 vote for a rate hike, then the reversal.", "Bank of England decision drill.")
        mock_contextual_repo.get_by_source_id.return_value = MagicMock(chunks=[chunk])
        mock_repo.get_by_source_id.return_value = None
        mock_visual_repo.get_by_chunk_id.return_value = self._record(
            'Slide with on-screen text "BoE REVERSAL 22/3/18"; "Result 7-2 for hike BUT".'
        )

        result = api.execute(source_id)

        assert result.audits == [VisualHallucinationAudit(chunk_id=chunk.chunk_id, is_visual_accepted=True)]
        mock_orchestrator.acquire_model.assert_not_called()
        mock_adapter.evaluate_visual_match.assert_not_called()
        mock_orchestrator.release_model.assert_not_called()
        mock_repo.save.assert_called_once()

    def test_unrelated_on_screen_text_is_checked_by_the_model_and_the_model_released(
            self, api, mock_repo, mock_adapter, mock_contextual_repo, mock_visual_repo, mock_orchestrator
    ):
        source_id = uuid.uuid4()
        related = self._chunk("A 7-2 vote for a rate hike, then the reversal.")
        unrelated = self._chunk("The shadow in the lower lobe of the lung.")
        mock_contextual_repo.get_by_source_id.return_value = MagicMock(chunks=[related, unrelated])
        mock_repo.get_by_source_id.return_value = None
        mock_visual_repo.get_by_chunk_id.side_effect = lambda chunk_id: {
            related.chunk_id: self._record('Slide with on-screen text "Result 7-2 for hike BUT".'),
            unrelated.chunk_id: self._record('Slide with on-screen text "TOMATO SOUP"; "Serves 4".'),
        }[chunk_id]
        loaded_model = MagicMock()
        mock_orchestrator.acquire_model.return_value = loaded_model
        rejection = VisualHallucinationAudit(chunk_id=unrelated.chunk_id, is_visual_accepted=False, discard_reason="x")
        mock_adapter.evaluate_visual_match.return_value = rejection

        result = api.execute(source_id, release_after=True)

        mock_orchestrator.acquire_model.assert_called_once()
        mock_adapter.evaluate_visual_match.assert_called_once_with(
            model=loaded_model, chunk_id=unrelated.chunk_id, text_chunk="The shadow in the lower lobe of the lung.",
            visual_description='- Frame 1: Slide with on-screen text "TOMATO SOUP"; "Serves 4".'
        )
        assert result.audits == [
            VisualHallucinationAudit(chunk_id=related.chunk_id, is_visual_accepted=True), rejection
        ]
        mock_orchestrator.release_model.assert_called_once_with(loaded_model, mock_adapter, flush=True)
