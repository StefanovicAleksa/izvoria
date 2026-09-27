# Canvas: context_unification_assembler_v2
# Path: app/features/context_unification/utils/assembler.py

import uuid
from typing import List, Dict, Optional

from app.core.anchors.domain.enums import AnchorKind
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.visual_context_synthesis.domain.entities import VisualTimeline, VisualTimelineChunk
from app.features.context_unification.domain.entities import UnifiedChunk


def assemble_unified_chunks(
        audio_transcript: ContextualizedTranscript,
        visual_timeline: Optional[VisualTimeline]
) -> List[UnifiedChunk]:
    """
    Deterministically stitches Phase 9 Audio Context and Phase 14 Visual Context
    into a single, RAG-optimized text payload.

    The contextualized `text_payload` is what gets embedded and keyword-indexed (Anthropic's
    Contextual Retrieval). The verbatim transcript and the generated context are also kept
    apart in `raw_text` / `context_text`, so the answer model is only ever shown source text.

    Args:
        audio_transcript: The Phase 9 output containing situational context.
        visual_timeline: The Phase 14 output containing narrative visual descriptions (can be None).

    Returns:
        List of finalized UnifiedChunks ready for the vector database.
    """
    # 1. Create O(1) Lookup Dictionary for Visual Chunks
    visual_dict: Dict[uuid.UUID, VisualTimelineChunk] = {}
    if visual_timeline:
        visual_dict = {vc.chunk_id: vc for vc in visual_timeline.visual_chunks}

    unified_chunks: List[UnifiedChunk] = []

    # 2. Iterate and Stitch
    for audio_chunk in audio_transcript.chunks:

        # Clean text inputs
        audio_context = audio_chunk.context_string.strip()
        original_text = audio_chunk.original_text.strip()

        # Look for a matching visual chunk
        visual_match = visual_dict.get(audio_chunk.chunk_id)

        # 3. Apply the formatting template
        is_document = audio_chunk.anchor_kind is not AnchorKind.TIME
        if audio_chunk.anchor_kind is AnchorKind.IMAGE:
            # A standalone image: its context is its file's metadata, and its "text" is already the description
            context_lines = [f"[Image Context: {audio_context}]"]
        elif is_document:
            # A document chunk has no audio and no frames: its context is the SLM's situating sentence and the
            # headings it sits under, which also give keyword search the section's vocabulary
            context_lines = [f"[Document Context: {audio_context}]"]
            if audio_chunk.section_path:
                context_lines.append(f"[Section: {audio_chunk.section_path}]")
        else:
            context_lines = [f"[Audio Context: {audio_context}]"]
        if visual_match and visual_match.visual_context_string.strip():
            context_lines.append(f"[Visual Context: {visual_match.visual_context_string.strip()}]")
            evidence = visual_match.visual_evidence
        else:
            # Fallback if there was no screen activity during this chunk
            evidence = []

        context_text = "\n".join(context_lines)
        payload = f"{context_text}\n\n{original_text}"

        # 4. Construct Final Domain Entity
        unified_chunks.append(UnifiedChunk(
            chunk_id=audio_chunk.chunk_id,
            start_sec=audio_chunk.start_sec,
            end_sec=audio_chunk.end_sec,
            anchor_kind=audio_chunk.anchor_kind,
            page_start=audio_chunk.page_start,
            page_end=audio_chunk.page_end,
            section_path=audio_chunk.section_path,
            text_payload=payload,
            raw_text=original_text,
            context_text=context_text,
            speaker_labels=list(dict.fromkeys(s.speaker_id for s in audio_chunk.sentences)),
            visual_evidence=evidence
        ))

    return unified_chunks
