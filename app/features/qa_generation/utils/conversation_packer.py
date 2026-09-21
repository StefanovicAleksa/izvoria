# Path: app/features/qa_generation/utils/conversation_packer.py

import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import ConversationHistory, NumberedChunk
from app.features.qa_generation.utils.citation_parser import number_for_reading
from app.features.qa_generation.utils.context_packer import DOC_HEADER_TOKENS, chunk_source_text

# Chat-template role markers around one message
MESSAGE_OVERHEAD_TOKENS = 8


@dataclass
class PackedTurn:
    question: str
    answer: str
    evidence: List[NumberedChunk] = field(default_factory=list)


@dataclass
class PackedConversation:
    """What the answer model is shown: retained earlier turns (oldest first) and the current turn's new evidence."""
    turns: List[PackedTurn]
    new_evidence: List[NumberedChunk]

    @property
    def shown_evidence(self) -> List[NumberedChunk]:
        return [n for turn in self.turns for n in turn.evidence] + self.new_evidence

    @property
    def doc_mapping(self) -> Dict[str, uuid.UUID]:
        return {f"[Doc {n.doc_number}]": n.chunk.chunk_id for n in self.shown_evidence}


def pack_conversation(
        history: ConversationHistory,
        current_question: str,
        current_chunks: List[RetrievedChunk],
        token_budget: int,
        turn_template_tokens: int,
        count_tokens: Callable[[str], int]
) -> PackedConversation:
    """
    Fits a conversation into token_budget, in priority order:

    1. The current question, then the current turn's evidence (best first). A chunk an earlier turn already
       showed is not numbered again: that turn's copy is kept instead, so it stays citable under its old tag.
    2. The text of earlier turns, newest first. Once one doesn't fit, it and every older turn are dropped.
    3. The evidence of the retained turns, newest turn first, so the oldest turns lose their evidence first.

    Numbers are never reused: new evidence continues from history.next_doc_number, whatever was dropped.
    turn_template_tokens is the cost of an empty user message (the template around evidence and question).
    """
    def chunk_cost(chunk: RetrievedChunk) -> int:
        return count_tokens(chunk_source_text(chunk)) + DOC_HEADER_TOKENS

    turns = history.turns
    owner: Dict[uuid.UUID, int] = {}
    for index, turn in enumerate(turns):
        for numbered in turn.evidence:
            owner[numbered.chunk.chunk_id] = index

    used = count_tokens(current_question) + turn_template_tokens + MESSAGE_OVERHEAD_TOKENS

    # 1. Current evidence
    new_chunks: List[RetrievedChunk] = []
    pinned: Set[uuid.UUID] = set()
    seen: Set[uuid.UUID] = set()
    for chunk in current_chunks:
        if chunk.chunk_id in seen:
            continue
        cost = chunk_cost(chunk)
        if used + cost > token_budget:
            continue
        seen.add(chunk.chunk_id)
        used += cost
        if chunk.chunk_id in owner:
            pinned.add(chunk.chunk_id)
        else:
            new_chunks.append(chunk)

    # 2. Earlier turns' text, newest first
    first_retained = len(turns)
    for index in range(len(turns) - 1, -1, -1):
        turn = turns[index]
        cost = (count_tokens(turn.question) + count_tokens(turn.answer)
                + turn_template_tokens + 2 * MESSAGE_OVERHEAD_TOKENS)
        if used + cost > token_budget:
            break
        used += cost
        first_retained = index

    # A pinned chunk whose turn was dropped is shown with the new evidence instead
    for chunk in current_chunks:
        if chunk.chunk_id in pinned and owner[chunk.chunk_id] < first_retained:
            pinned.discard(chunk.chunk_id)
            new_chunks.append(chunk)

    # 3. Retained turns' evidence, newest turn first
    kept: Dict[int, Set[uuid.UUID]] = {}
    for index in range(len(turns) - 1, first_retained - 1, -1):
        keep = kept.setdefault(index, set())
        for numbered in turns[index].evidence:
            chunk_id = numbered.chunk.chunk_id
            if owner[chunk_id] != index or chunk_id in keep:
                continue
            if chunk_id in pinned:
                keep.add(chunk_id)
                continue
            cost = chunk_cost(numbered.chunk)
            if used + cost > token_budget:
                continue
            used += cost
            keep.add(chunk_id)

    packed_turns = [
        PackedTurn(
            question=turns[index].question,
            answer=turns[index].answer,
            evidence=[n for n in turns[index].evidence if n.chunk.chunk_id in kept.get(index, set())]
        )
        for index in range(first_retained, len(turns))
    ]
    new_evidence = [
        NumberedChunk(doc_number=number, chunk=chunk)
        for number, chunk in number_for_reading(new_chunks, history.next_doc_number)
    ]
    return PackedConversation(turns=packed_turns, new_evidence=new_evidence)
