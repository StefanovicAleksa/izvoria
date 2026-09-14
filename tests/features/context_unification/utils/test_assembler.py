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

    def test_assemble_keeps_raw_transcript_and_generated_context_separate(self):
        import uuid as _uuid
        from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence

        chunk_id = _uuid.uuid4()
        sentences = [
            ReconstructedSentence(text="Did you see him?", speaker_id="Speaker_1", start_sec=0, end_sec=1),
            ReconstructedSentence(text="Yes.", speaker_id="Speaker_0", start_sec=1, end_sec=2),
            ReconstructedSentence(text="Where?", speaker_id="Speaker_1", start_sec=2, end_sec=3),
        ]
        raw = "[Speaker_1 @ 00:00:00]: Did you see him?\n[Speaker_0 @ 00:00:01]: Yes.\n[Speaker_1 @ 00:00:02]: Where?"
        audio_transcript = ContextualizedTranscript(
            source_id=_uuid.uuid4(),
            chunks=[ContextualizedChunk(chunk_id=chunk_id, start_sec=0.0, end_sec=3.0, original_text=raw,
                                        context_string="Cross-examination of the witness.", sentences=sentences)]
        )
        visual_timeline = VisualTimeline(
            source_id=audio_transcript.source_id,
            visual_chunks=[VisualTimelineChunk(chunk_id=chunk_id, visual_context_string="Exhibit 4 on screen.")]
        )

        unified_chunk = assemble_unified_chunks(audio_transcript, visual_timeline)[0]

        assert unified_chunk.raw_text == raw
        assert unified_chunk.context_text == (
            "[Audio Context: Cross-examination of the witness.]\n[Visual Context: Exhibit 4 on screen.]"
        )
        assert unified_chunk.text_payload == unified_chunk.context_text + "\n\n" + raw
        assert unified_chunk.speaker_labels == ["Speaker_1", "Speaker_0"]
