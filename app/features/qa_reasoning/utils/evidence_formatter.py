# Path: app/features/qa_reasoning/utils/evidence_formatter.py

import logging
import uuid
from typing import Callable, Dict, List, Optional

from app.core.anchors.domain.enums import AnchorKind
from app.core.anchors.utils.formatting import format_location
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_reasoning.domain.entities import ReasonerDecision, EvidenceBlock

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_CHUNK = 3000
_SEPARATOR_TOKENS = 2
_EMPTY_POOL_TEXT = "(No evidence gathered yet.)"


def _location(chunk: RetrievedChunk) -> str:
    """A transcript chunk keeps its "12.0s-40.0s" range; a document chunk shows its page and section."""
    if chunk.anchor_kind is AnchorKind.TIME:
        return f"{chunk.start_sec:.1f}s-{chunk.end_sec:.1f}s"
    return format_location(chunk.anchor_kind, page_start=chunk.page_start, page_end=chunk.page_end,
                           section_path=chunk.section_path)


def build_evidence_block(
        evidence_pool: List[RetrievedChunk],
        token_budget: Optional[int] = None,
        count_tokens: Optional[Callable[[str], int]] = None
) -> EvidenceBlock:
    """
    Formats the evidence pool for the Reasoner's prompt, tagging chunks [E1]..[En]. A 7B model copies a short
    tag reliably where it garbled 36-character UUIDs; the returned mapping turns tags back into chunk ids.

    The pool arrives best-first. With a token budget, chunks that don't fit are left out (a later, shorter
    chunk may still fit), so a growing evidence pool can never overflow the context window.
    """
    if not evidence_pool:
        return EvidenceBlock(formatted_text=_EMPTY_POOL_TEXT, tag_mapping={})

    parts = []
    tag_mapping: Dict[str, uuid.UUID] = {}
    used = 0
    for chunk in evidence_pool:
        source_text = chunk.raw_text or chunk.text_payload
        tag = f"E{len(parts) + 1}"
        part = f"[{tag}] ({_location(chunk)}):\n{source_text[:_MAX_CHARS_PER_CHUNK]}"

        if token_budget is not None and count_tokens is not None:
            cost = count_tokens(part) + _SEPARATOR_TOKENS
            if used + cost > token_budget:
                continue
            used += cost

        parts.append(part)
        tag_mapping[tag] = chunk.chunk_id

    if not parts:
        return EvidenceBlock(formatted_text=_EMPTY_POOL_TEXT, tag_mapping={})
    if len(parts) < len(evidence_pool):
        logger.info(f"Reasoner evidence budget of {token_budget} tokens fits {len(parts)} of {len(evidence_pool)} chunks.")

    return EvidenceBlock(formatted_text="\n\n".join(parts), tag_mapping=tag_mapping)


def build_decision_history_block(decision_history: List[ReasonerDecision]) -> str:
    """One line per past reasoning turn, so the Reasoner doesn't repeat a query it already tried."""
    if not decision_history:
        return "(No prior reasoning turns.)"

    lines = []
    for idx, decision in enumerate(decision_history, start=1):
        detail = decision.refined_query or decision.clarification_question or ""
        suffix = f" — {detail}" if detail else ""
        lines.append(f"Iteration {idx}: {decision.decision.value}{suffix}")

    return "\n".join(lines)
