# Canvas: test_detection_api_v2
# Path: tests/features/visual_cue_detection/service/test_detection_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI


class TestVisualCueDetectionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "adapter": MagicMock(),
            "contextual_api": MagicMock()  # Swapped repo for the contextual_api mock
        }

    @pytest.fixture
    def api(self, mocks):
        return VisualCueDetectionAPI(
            detector_adapter=mocks["adapter"],
            contextualizer_api=mocks["contextual_api"]
        )

    def test_process_transcript_updates_all_chunks(self, api, mocks):
        chunk1 = ContextualizedChunk(chunk_id=uuid.uuid4(), needs_visual=False)
        chunk2 = ContextualizedChunk(chunk_id=uuid.uuid4(), needs_visual=False)
        input_transcript = ContextualizedTranscript(
            id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            chunks=[chunk1, chunk2]
        )

        mock_model = MagicMock()

        updated_chunk1 = chunk1.model_copy(update={"needs_visual": True})
        updated_chunk2 = chunk2.model_copy(update={"needs_visual": True})

        mocks["adapter"].detect_cues.side_effect = [updated_chunk1, updated_chunk2]

        result_transcript = api.process_transcript(input_transcript, mock_model)

        assert mocks["adapter"].detect_cues.call_count == 2
        assert len(result_transcript.chunks) == 2
        assert result_transcript.chunks[0].needs_visual is True

        # Verify it now calls update_transcript on the ContextualizerAPI
        mocks["contextual_api"].update_transcript.assert_called_once()
        saved_transcript = mocks["contextual_api"].update_transcript.call_args[0][0]
        assert saved_transcript.id == input_transcript.id