# Canvas: test_scrubber_api_v1
# Path: tests/features/hallucination_scrubber/service/test_scrubber_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.hallucination_scrubber.service.scrubber_api import HallucinationScrubberAPI
from app.features.hallucination_scrubber.domain.entities import HallucinationScrubbedTranscript


class TestHallucinationScrubberAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "transcript_repo": MagicMock(),
            "domain_repo": MagicMock(),
            "hallucination_repo": MagicMock(),
            "orchestrator": MagicMock(spec=ModelManagerOrchestrator),
            "adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return HallucinationScrubberAPI(
            transcript_repo=mocks["transcript_repo"],
            domain_repo=mocks["domain_repo"],
            hallucination_repo=mocks["hallucination_repo"],
            orchestrator=mocks["orchestrator"],
            adapter=mocks["adapter"]
        )

    def test_execute_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        existing_record = HallucinationScrubbedTranscript(source_id=source_id, segments=[], audit_trail=[])
        mocks["hallucination_repo"].get_by_source_id.return_value = existing_record

        result = api.execute(source_id)

        assert result == existing_record
        mocks["transcript_repo"].get_by_source_id.assert_not_called()
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_execute_fast_path_clean_transcript(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["hallucination_repo"].get_by_source_id.return_value = None

        clean_segment = TranscriptSegment(start_sec=0.0, end_sec=5.0, text="Valid trading discussion.", speaker_id="S0",
                                          words=[])
        dirty_transcript = Transcript(source_id=source_id, segments=[clean_segment])
        mocks["transcript_repo"].get_by_source_id.return_value = dirty_transcript

        glossary = DomainGlossary(source_id=source_id, industry="FINANCE", sub_domains=["TRADING"], hotword_string="")
        mocks["domain_repo"].get_glossary_by_source_id.return_value = glossary

        # Adapter returns scores showing it's completely safe (<0.85)
        mocks["adapter"].classify.return_value = {"youtube outro, subscription request, or video formatting text": 0.05}

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mocks["hallucination_repo"].save.side_effect = lambda x: x

        result = api.execute(source_id, release_after=True)

        # Fast path means it never needed to batch-classify sub-clauses
        mocks["adapter"].classify_batch.assert_not_called()
        assert len(result.segments) == 1
        assert len(result.audit_trail) == 0  # No patches applied
        mocks["orchestrator"].release_model.assert_called_once_with(mock_loaded_model, mocks["adapter"], flush=True)

    def test_execute_slow_path_salvages_valid_clauses(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["hallucination_repo"].get_by_source_id.return_value = None

        dirty_segment = TranscriptSegment(
            start_sec=0.0, end_sec=5.0,
            text="Valid trading discussion. Thank you for watching.",
            speaker_id="S0", words=[]
        )
        mocks["transcript_repo"].get_by_source_id.return_value = Transcript(source_id=source_id,
                                                                            segments=[dirty_segment])
        mocks["domain_repo"].get_glossary_by_source_id.return_value = None

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        # 1. Fast path catches the hallucination
        mocks["adapter"].classify.return_value = {"youtube outro, subscription request, or video formatting text": 0.98}

        # 2. Slow path evaluates sub-clauses. Clause 1 is safe, Clause 2 is dirty.
        mocks["adapter"].classify_batch.return_value = [
            {"youtube outro, subscription request, or video formatting text": 0.02},  # "Valid trading discussion."
            {"youtube outro, subscription request, or video formatting text": 0.99}  # "Thank you for watching."
        ]

        mocks["hallucination_repo"].save.side_effect = lambda x: x

        result = api.execute(source_id)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Valid trading discussion."

        # Ensure the immutable audit trail captured the exact deletion
        assert len(result.audit_trail) == 1
        assert result.audit_trail[0].original_text == "Valid trading discussion. Thank you for watching."
        assert result.audit_trail[0].scrubbed_text == "Valid trading discussion."
        assert len(result.audit_trail[0].applied_patches) == 1
        assert result.audit_trail[0].applied_patches[0].removed_text == "Thank you for watching."