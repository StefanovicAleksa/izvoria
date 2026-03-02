# Canvas: test_nlp_chunker_v2
# Path: tests/features/nlp_router/test_chunker.py

import uuid
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.nlp_router.utils.chunker import chunk_transcript


class TestTranscriptChunker:
    def test_chunk_transcript_under_limit(self):
        """Verifies a short transcript stays as a single chunk."""
        transcript = Transcript(
            source_id=uuid.uuid4(),  # FIXED: Replaced None with valid UUID
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=5.0, text="Hello", speaker_id="Speaker_0"),
                TranscriptSegment(start_sec=5.0, end_sec=10.0, text="World", speaker_id="Speaker_1"),
            ]
        )

        chunks = chunk_transcript(transcript, max_duration_sec=180.0)

        assert len(chunks) == 1
        assert chunks[0].start_sec == 0.0
        assert chunks[0].end_sec == 10.0
        assert "[0.0s - 5.0s] Speaker_0: Hello" in chunks[0].text_block
        assert "[5.0s - 10.0s] Speaker_1: World" in chunks[0].text_block

    def test_chunk_transcript_exceeds_limit(self):
        """Verifies a long transcript is split strictly at the duration boundary."""
        transcript = Transcript(
            source_id=uuid.uuid4(),  # FIXED: Replaced None with valid UUID
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=100.0, text="Part 1", speaker_id="Speaker"),
                TranscriptSegment(start_sec=100.0, end_sec=200.0, text="Part 2", speaker_id="Speaker"),
                TranscriptSegment(start_sec=200.0, end_sec=210.0, text="Part 3", speaker_id="Speaker"),
            ]
        )

        chunks = chunk_transcript(transcript, max_duration_sec=180.0)

        # It should split after Part 2 because 200.0 - 0.0 > 180.0
        assert len(chunks) == 2

        assert chunks[0].start_sec == 0.0
        assert chunks[0].end_sec == 200.0
        assert "Part 1" in chunks[0].text_block
        assert "Part 2" in chunks[0].text_block

        assert chunks[1].start_sec == 200.0
        assert chunks[1].end_sec == 210.0
        assert "Part 3" in chunks[1].text_block