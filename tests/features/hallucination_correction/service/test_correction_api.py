# Canvas: test_correction_api_v2
# Path: tests/features/hallucination_correction/service/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.hallucination_correction.service.correction_api import CorrectionAPI
from app.features.hallucination_correction.exceptions.validation import MissingPrerequisiteError
from app.features.hallucination_correction.exceptions.database import CorrectionDatabaseError


class TestCorrectionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "transcript_repo": MagicMock(),
            "vad_repo": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return CorrectionAPI(
            transcript_repository=mocks["transcript_repo"],
            vad_repository=mocks["vad_repo"]
        )

    def test_process_transcript_missing_transcript(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["transcript_repo"].get_by_source_id.return_value = None

        with pytest.raises(MissingPrerequisiteError):
            api.process_transcript(source_id)

    def test_process_transcript_missing_vad(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["transcript_repo"].get_by_source_id.return_value = Transcript(source_id=source_id, segments=[])
        mocks["vad_repo"].get_map_by_source_id.return_value = None

        with pytest.raises(MissingPrerequisiteError):
            api.process_transcript(source_id)

    def test_process_transcript_empty_transcript_returns_early(self, api, mocks):
        source_id = uuid.uuid4()
        empty_transcript = Transcript(source_id=source_id, segments=[])

        mocks["transcript_repo"].get_by_source_id.return_value = empty_transcript
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(source_id=source_id, segments=[])

        result_transcript, telemetry = api.process_transcript(source_id)

        assert result_transcript == empty_transcript
        assert telemetry.total_original_segments == 0
        mocks["transcript_repo"].update.assert_not_called()

    def test_process_transcript_success_flow(self, api, mocks):
        source_id = uuid.uuid4()

        # Valid Segment (Should survive)
        seg_good = TranscriptSegment(start_sec=1.0, end_sec=2.0, text="Valid speech", no_speech_prob=0.1)
        # Invalid Segment (Should be regex scrubbed)
        seg_bad = TranscriptSegment(start_sec=3.0, end_sec=4.0, text="Please subscribe", no_speech_prob=0.1)

        input_transcript = Transcript(source_id=source_id, segments=[seg_good, seg_bad])
        vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0.0, end_sec=10.0, confidence=0.9)])

        mocks["transcript_repo"].get_by_source_id.return_value = input_transcript
        mocks["vad_repo"].get_map_by_source_id.return_value = vad_map
        mocks["transcript_repo"].update.return_value = input_transcript  # Mutated in place

        result_transcript, telemetry = api.process_transcript(source_id)

        # Assert Telemetry
        assert telemetry.total_original_segments == 2
        assert telemetry.removed_by_regex == 1
        assert telemetry.total_surviving_segments == 1

        # Assert Persistence
        mocks["transcript_repo"].update.assert_called_once_with(input_transcript)
        assert len(result_transcript.segments) == 1
        assert result_transcript.segments[0].text == "Valid speech"

    def test_process_transcript_db_failure_maps_safely(self, api, mocks):
        source_id = uuid.uuid4()
        seg_good = TranscriptSegment(start_sec=1.0, end_sec=2.0, text="Valid speech")

        mocks["transcript_repo"].get_by_source_id.return_value = Transcript(source_id=source_id, segments=[seg_good])
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(source_id=source_id, segments=[
            VadSegment(start_sec=0.0, end_sec=10.0, confidence=0.9)])

        # FIX: Raise a true SQLAlchemyError so the error_mapper correctly buckets it as a Database Error
        mocks["transcript_repo"].update.side_effect = SQLAlchemyError("SQLAlchemy Commit Failed")

        with pytest.raises(CorrectionDatabaseError):
            api.process_transcript(source_id)