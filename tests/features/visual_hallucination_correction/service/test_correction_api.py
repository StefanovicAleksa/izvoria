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