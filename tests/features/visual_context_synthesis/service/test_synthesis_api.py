# Canvas: test_visual_synthesis_api_v1
# Path: tests/features/visual_context_synthesis/service/test_synthesis_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.visual_context_synthesis.service.synthesis_api import VisualSynthesisAPI
from app.features.visual_context_synthesis.domain.entities import VisualTimeline
from app.features.visual_context_synthesis.exceptions.model import VisualSLMInferenceError


class TestVisualSynthesisAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "orchestrator": MagicMock(spec=ModelManagerOrchestrator),
            "slm_adapter": MagicMock(),
            "visual_analysis_repo": MagicMock(),
            "hallucination_repo": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return VisualSynthesisAPI(
            repository=mocks["repo"],
            orchestrator=mocks["orchestrator"],
            slm_adapter=mocks["slm_adapter"],
            visual_analysis_repo=mocks["visual_analysis_repo"],
            hallucination_repo=mocks["hallucination_repo"]
        )

    def test_execute_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        mock_existing = VisualTimeline(source_id=source_id, visual_chunks=[])
        mocks["repo"].get_by_source_id.return_value = mock_existing

        dummy_transcript = ContextualizedTranscript(source_id=source_id, chunks=[])
        result = api.execute(dummy_transcript)

        assert result == mock_existing
        mocks["repo"].get_by_source_id.assert_called_once_with(source_id)
        mocks["orchestrator"].acquire_model.assert_not_called()

    @patch("app.features.visual_context_synthesis.service.synthesis_api.build_fuzzy_visual_timeline")
    def test_execute_success_flow_and_vram_release(self, mock_fuzzy, api, mocks):
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mocks["repo"].get_by_source_id.return_value = None

        # 1. Mock the input transcript
        mock_chunk = ContextualizedChunk(chunk_id=chunk_id)
        dummy_transcript = ContextualizedTranscript(source_id=source_id, chunks=[mock_chunk])

        # 2. Mock the Hallucination Ledger (Phase 13)
        mock_audit = MagicMock()
        mock_audit.chunk_id = chunk_id
        mock_audit.is_visual_accepted = True
        mock_ledger = MagicMock()
        mock_ledger.audits = [mock_audit]
        mocks["hallucination_repo"].get_by_source_id.return_value = mock_ledger

        # 3. Mock the Visual Analyses (Phase 12)
        mock_analysis_aggr = MagicMock()
        mock_analysis_aggr.analyses = [MagicMock()]
        mocks["visual_analysis_repo"].get_by_chunk_id.return_value = mock_analysis_aggr

        # 4. Mock the Fuzzy Grouper utility
        mock_fuzzy.return_value = [
            {"start_sec": 10.0, "end_sec": 10.0, "representative_description": "Chart.", "frame_ids": [uuid.uuid4()]}
        ]

        # 5. Mock the SLM
        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model
        mocks["slm_adapter"].synthesize_context.return_value = "A synthesized chart."
        mocks["repo"].save.side_effect = lambda x: x

        result = api.execute(dummy_transcript, release_after=True)

        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["slm_adapter"].synthesize_context.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["slm_adapter"], flush=True
        )

        assert len(result.visual_chunks) == 1
        assert result.visual_chunks[0].visual_context_string == "A synthesized chart."
        assert len(result.visual_chunks[0].visual_evidence) == 1

    def test_execute_guarantees_vram_release_on_crash(self, api, mocks):
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        mocks["repo"].get_by_source_id.return_value = None

        mock_chunk = ContextualizedChunk(chunk_id=chunk_id)
        dummy_transcript = ContextualizedTranscript(source_id=source_id, chunks=[mock_chunk])

        # Make the chunk accepted so it triggers inference
        mock_audit = MagicMock()
        mock_audit.chunk_id = chunk_id
        mock_audit.is_visual_accepted = True
        mock_ledger = MagicMock()
        mock_ledger.audits = [mock_audit]
        mocks["hallucination_repo"].get_by_source_id.return_value = mock_ledger

        mock_analysis_aggr = MagicMock()
        mock_analysis_aggr.analyses = [MagicMock()]
        mocks["visual_analysis_repo"].get_by_chunk_id.return_value = mock_analysis_aggr

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        # Force the SLM to crash
        mocks["slm_adapter"].synthesize_context.side_effect = VisualSLMInferenceError("Out of Memory")

        with pytest.raises(VisualSLMInferenceError):
            api.execute(dummy_transcript, release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["slm_adapter"], flush=True
        )