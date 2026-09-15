# Path: app/features/qa_reasoning/utils/evidence_formatter.py

from typing import List

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_reasoning.domain.entities import ReasonerDecision

_MAX_CHARS_PER_CHUNK = 3000


def build_evidence_block(evidence_pool: List[RetrievedChunk]) -> str:
    """
    Formats every surviving chunk for the Reasoner's prompt, tagged by its literal chunk_id
    (not a synthetic [Doc N] tag) so ReasonerDecision.evidence_links can reference it directly
    for traceability across iterations.
    """
    if not evidence_pool:
        return "(No evidence gathered yet.)"

    parts = []
    for chunk in evidence_pool:
        source_text = chunk.raw_text or chunk.text_payload
        safe_text = source_text[:_MAX_CHARS_PER_CHUNK]
        parts.append(f"[{chunk.chunk_id}] ({chunk.start_sec:.1f}s-{chunk.end_sec:.1f}s):\n{safe_text}")

    return "\n\n".join(parts)


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
