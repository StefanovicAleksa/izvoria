# Canvas: test_hallucination_correction_api_v2
# Path: tests/features/visual_hallucination_correction/service/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI
from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationEvaluatedTranscript,
    VisualHallucinationAudit
)
from app.features.visual_hallucination_correction.exceptions.model import SLMInferenceError


class TestVisualHallucinationCorrectionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "adapter": MagicMock(),
            "contextual_repo": MagicMock(),
            "visual_analysis_repo": MagicMock(),
            "orchestrator": MagicMock(spec=ModelManagerOrchestrator)
        }

    @pytest.fixture
    def api(self, mocks):
        return VisualHallucinationCorrectionAPI(
            repository=mocks["repo"],
            adapter=mocks["adapter"],
            contextual_repo=mocks["contextual_repo"],
            visual_analysis_repo=mocks["visual_analysis_repo"],
            orchestrator=mocks["orchestrator"]
        )

    def test_execute_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        mock_existing = VisualHallucinationEvaluatedTranscript(source_id=source_id, audits=[])
        mocks["repo"].get_by_source_id.return_value = mock_existing

        result = api.execute(source_id)

        assert result == mock_existing
        mocks["repo"].get_by_source_id.assert_called_once_with(source_id)
        mocks["contextual_repo"].get_by_source_id.assert_not_called()

    def test_execute_passthrough_empty_transcript(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        mocks["contextual_repo"].get_by_source_id.return_value = None

        api.execute(source_id)

        mocks["repo"].save.assert_called_once()
        saved_arg = mocks["repo"].save.call_args[0][0]
        assert len(saved_arg.audits) == 0
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_execute_success_flow_and_vram_release(self, api, mocks):
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mocks["repo"].get_by_source_id.return_value = None

        # 1. Mock Contextual Transcript (needs_visual = True)
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = chunk_id
        mock_chunk.needs_visual = True
        mock_chunk.original_text = "Look at this scan."

        mock_transcript = MagicMock()
        mock_transcript.chunks = [mock_chunk]
        mocks["contextual_repo"].get_by_source_id.return_value = mock_transcript

        # 2. Mock Visual Analysis Result
        mock_analysis = MagicMock()
        mock_analysis.description = "A chest X-ray."
        mock_analysis_aggr = MagicMock()
        mock_analysis_aggr.analyses = [mock_analysis]
        mocks["visual_analysis_repo"].get_by_chunk_id.return_value = mock_analysis_aggr

        # 3. Mock SLM Evaluation
        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mock_audit = VisualHallucinationAudit(chunk_id=chunk_id, is_visual_accepted=True)
        mocks["adapter"].evaluate_visual_match.return_value = mock_audit

        mocks["repo"].save.side_effect = lambda x: x

        result = api.execute(source_id, release_after=True)

        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["adapter"].evaluate_visual_match.assert_called_once_with(
            model=mock_loaded_model,
            chunk_id=chunk_id,
            text_chunk="Look at this scan.",
            visual_description="A chest X-ray."
        )
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["adapter"], flush=True
        )

        assert len(result.audits) == 1
        assert result.audits[0].is_visual_accepted is True

    def test_execute_guarantees_vram_release_on_crash(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        mock_chunk = MagicMock()
        mock_chunk.needs_visual = True
        mock_transcript = MagicMock()
        mock_transcript.chunks = [mock_chunk]
        mocks["contextual_repo"].get_by_source_id.return_value = mock_transcript

        # FIX: Provide a concrete string representation to prevent string joining errors in API
        mock_analysis = MagicMock()
        mock_analysis.description = "A valid visual description string."
        mock_analysis_aggr = MagicMock()
        mock_analysis_aggr.analyses = [mock_analysis]
        mocks["visual_analysis_repo"].get_by_chunk_id.return_value = mock_analysis_aggr

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model
        mocks["adapter"].evaluate_visual_match.side_effect = SLMInferenceError("Out of Memory")

        with pytest.raises(SLMInferenceError):
            api.execute(source_id, release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["adapter"], flush=True
        )