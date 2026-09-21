# Path: app/engine/utils/conversation_history.py

import uuid
from typing import Callable, Dict, List

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import ConversationHistory, ConversationTurn, NumberedChunk
from app.features.session_memory.domain.entities import SessionTurn


def build_conversation_history(
        turns: List[SessionTurn],
        get_chunks: Callable[[List[uuid.UUID]], List[RetrievedChunk]]
) -> ConversationHistory:
    """
    Stored turns (oldest first) with their evidence under the conversation-wide [Doc N] numbers they were given.
    Chunks deleted since (e.g. a re-ingested source) are skipped without renumbering the rest.
    """
    all_ids = [chunk_id for turn in turns for chunk_id in turn.evidence_chunk_ids]
    chunks: Dict[uuid.UUID, RetrievedChunk] = {c.chunk_id: c for c in get_chunks(all_ids)} if all_ids else {}

    history_turns = [
        ConversationTurn(
            question=turn.question,
            answer=turn.answer_text,
            evidence=[
                NumberedChunk(doc_number=turn.first_doc_number + offset, chunk=chunks[chunk_id])
                for offset, chunk_id in enumerate(turn.evidence_chunk_ids)
                if chunk_id in chunks
            ]
        )
        for turn in turns
    ]
    next_doc_number = turns[-1].next_doc_number if turns else 1
    return ConversationHistory(turns=history_turns, next_doc_number=next_doc_number)


def history_chunks_newest_first(history: ConversationHistory) -> List[RetrievedChunk]:
    """Every chunk the conversation has shown, most recent turn first, each once."""
    seen = set()
    ordered: List[RetrievedChunk] = []
    for turn in reversed(history.turns):
        for numbered in turn.evidence:
            if numbered.chunk.chunk_id not in seen:
                seen.add(numbered.chunk.chunk_id)
                ordered.append(numbered.chunk)
    return ordered
