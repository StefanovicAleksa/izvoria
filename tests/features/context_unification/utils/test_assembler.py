# Canvas: test_context_unification_assembler_v1
# Path: tests/features/context_unification/utils/test_assembler.py

import uuid
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.visual_context_synthesis.domain.entities import VisualTimeline, VisualTimelineChunk, \
    VisualContextFrame
from app.features.context_unification.utils.assembler import assemble_unified_chunks


class TestContextUnificationAssembler:
    def test_assemble_unified_chunks_with_visual_match(self):
        chunk_id = uuid.uuid4()

        audio_transcript = ContextualizedTranscript(
            source_id=uuid.uuid4(),
            chunks=[ContextualizedChunk(
                chunk_id=chunk_id,
                start_sec=0.0,
                end_sec=5.0,
                original_text="Look at this chart.",
                context_string="Speaker introduces Q3 financials."
            )]
        )

        frame = VisualContextFrame(frame_id=uuid.uuid4(), timestamp_sec=2.5)
        visual_timeline = VisualTimeline(
            source_id=audio_transcript.source_id,
            visual_chunks=[VisualTimelineChunk(
                chunk_id=chunk_id,
                visual_context_string="A bar chart showing Q3 growth.",
                visual_evidence=[frame]
            )]
        )

        result = assemble_unified_chunks(audio_transcript, visual_timeline)

        assert len(result) == 1
        unified_chunk = result[0]

        assert unified_chunk.chunk_id == chunk_id
        assert "Audio Context: Speaker introduces Q3 financials." in unified_chunk.text_payload
        assert "Visual Context: A bar chart showing Q3 growth." in unified_chunk.text_payload
        assert "Look at this chart." in unified_chunk.text_payload
        assert len(unified_chunk.visual_evidence) == 1
        assert unified_chunk.visual_evidence[0] == frame

    def test_assemble_unified_chunks_without_visual_match(self):
        chunk_id = uuid.uuid4()

        audio_transcript = ContextualizedTranscript(
            source_id=uuid.uuid4(),
            chunks=[ContextualizedChunk(
                chunk_id=chunk_id,
                start_sec=0.0,
                end_sec=5.0,
                original_text="Just talking to the camera.",
                context_string="General introduction."
            )]
        )

        result = assemble_unified_chunks(audio_transcript, None)

        assert len(result) == 1
        unified_chunk = result[0]

        assert "Audio Context: General introduction." in unified_chunk.text_payload
        assert "Visual Context:" not in unified_chunk.text_payload
        assert "Just talking to the camera." in unified_chunk.text_payload
        assert len(unified_chunk.visual_evidence) == 0