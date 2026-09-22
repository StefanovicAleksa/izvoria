# Path: app/features/deep_research_report/utils/evidence_bank.py

"""
The evidence memory bank: turning a sweep's raw hits into a stably numbered, deduplicated set of
`EvidenceEntry` rows, and slicing that set back out per section at writing time.

Numbering happens exactly once, over the whole bank, before any section is written. That is what makes a
`[Doc N]` mean the same thing in section 2, in section 7, in the summary, in the references table, in the
desktop's click-to-verify panel and in a resumed run three days later.
"""

import uuid
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from app.features.deep_research_report.domain.entities import EvidenceEntry
from app.features.deep_research_report.domain.enums import EvidenceOrigin
from app.features.qa_generation.utils.context_packer import chunk_source_text, order_for_reading
from app.features.rag_retrieval.domain.entities import RetrievedChunk

# How much of a chunk's transcript the planner sees per bank entry. Long enough to judge relevance, short
# enough that a 200-entry bank still fits one prompt alongside the outline.
DIGEST_CHARS = 200

_UNKNOWN_SOURCE = "Unknown source"


def build_digest(chunk: RetrievedChunk) -> str:
    """
    The head of a chunk's VERBATIM transcript, for the planner's bank index.

    `raw_text`, never `text_payload`: the generated situating context inside `text_payload` is there to make
    retrieval work, and is explicitly not citable source material. A digest built from it would let the
    planner assign — and ultimately let the report cite — a sentence no human ever said.
    """
    text = " ".join(chunk_source_text(chunk).split())
    if len(text) <= DIGEST_CHARS:
        return text
    return text[:DIGEST_CHARS].rstrip() + "…"


def dedupe_hits(
        hits: Iterable[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]]
) -> List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]]:
    """
    Collapses the fan-out's overlap. A chunk surfaced by three sub-questions is one piece of evidence, not
    three: it gets one doc number, and (if scrubbing is on) is judged once instead of three times.

    The best-scoring occurrence wins, so a chunk keeps the score of the query it matched best. `found_by`
    and `origin` come from that same occurrence, which is why a chunk that both a sub-question and the
    sweep found is attributed to whichever ranked it higher rather than to whichever ran first.
    """
    best: Dict[uuid.UUID, Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]] = {}
    for chunk, origin, found_by in hits:
        existing = best.get(chunk.chunk_id)
        if existing is None or _score(chunk) > _score(existing[0]):
            best[chunk.chunk_id] = (chunk, origin, found_by)
    return list(best.values())


def _score(chunk: RetrievedChunk) -> float:
    """Rerank score when the cross-encoder ran, RRF score otherwise."""
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.relevance_score


def number_bank(
        hits: List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]],
        source_names: Dict[uuid.UUID, str],
        scrubbed_ids: Optional[Iterable[uuid.UUID]] = None,
) -> List[EvidenceEntry]:
    """
    Assigns `[Doc 1..N]` across the whole bank in reading order and returns the numbered entries.

    Reading order (grouped by source, chronological within a source) rather than rank order, for the same
    reason the chat path uses it: a reader following citations through the references table should move
    through each recording as a timeline, not bounce around by relevance score.
    """
    scrubbed = set(scrubbed_ids or ())
    by_id = {chunk.chunk_id: (chunk, origin, found_by) for chunk, origin, found_by in hits}

    ordered_chunks = order_for_reading([chunk for chunk, _, _ in hits])

    entries: List[EvidenceEntry] = []
    for number, chunk in enumerate(ordered_chunks, start=1):
        _, origin, found_by = by_id[chunk.chunk_id]
        entries.append(EvidenceEntry(
            doc_number=number,
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_name=source_names.get(chunk.source_id, _UNKNOWN_SOURCE),
            start_sec=chunk.start_sec,
            end_sec=chunk.end_sec,
            speaker_labels=list(chunk.speaker_labels),
            digest=build_digest(chunk),
            origin=origin,
            found_by=found_by,
            rerank_score=chunk.rerank_score,
            scrubbed=chunk.chunk_id in scrubbed,
        ))
    return entries


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_bank_index(entries: List[EvidenceEntry]) -> str:
    """
    The whole bank as one line per entry, for the outline-optimisation call.

    This is the trick that keeps planning and writing inside a single model acquisition: the planner never
    reads full transcripts, only this index, so even a 200-entry bank costs ~12K tokens and leaves the
    128K window free for the section writes that follow on the same loaded model.
    """
    lines = []
    for entry in entries:
        speakers = ", ".join(entry.speaker_labels) if entry.speaker_labels else "unlabelled"
        lines.append(
            f"[Doc {entry.doc_number}] {entry.source_name} "
            f"({format_timestamp(entry.start_sec)}–{format_timestamp(entry.end_sec)}, {speakers}): {entry.digest}"
        )
    return "\n".join(lines)


def build_section_digest(sections: Iterable[Tuple[str, str]], max_chars: int = 600) -> str:
    """
    Written sections as (title, body) pairs, trimmed, for the executive summary call.

    The summary is composed from section prose rather than from evidence on purpose: it is the one part of
    a report that synthesises across sections, so it is the easiest place to invent a connection no source
    supports. Giving it the sections' own words -- already cited and already validated -- means the worst
    it can do is restate them.
    """
    parts = []
    for title, body in sections:
        text = " ".join(body.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        parts.append(f"## {title}\n{text}")
    return "\n\n".join(parts)


def chunks_for_section(
        doc_numbers: Iterable[int],
        entries_by_number: Dict[int, EvidenceEntry],
        chunk_lookup: Dict[uuid.UUID, RetrievedChunk],
) -> List[Tuple[int, RetrievedChunk]]:
    """
    This section's slice of the bank, as (doc_number, chunk) pairs in ascending doc order.

    Numbers whose chunk is missing -- deleted by a re-ingest between gathering and writing -- are dropped
    rather than rendered as an empty document, so the writer is never shown a `[Doc N]` header with no
    text under it and invited to cite it anyway.
    """
    pairs: List[Tuple[int, RetrievedChunk]] = []
    for number in sorted(set(doc_numbers)):
        entry = entries_by_number.get(number)
        if entry is None:
            continue
        chunk = chunk_lookup.get(entry.chunk_id)
        if chunk is None:
            continue
        pairs.append((number, chunk))
    return pairs


def pack_section_evidence(
        pairs: List[Tuple[int, RetrievedChunk]],
        token_budget: int,
        count_tokens: Callable[[str], int],
) -> List[Tuple[int, RetrievedChunk]]:
    """
    Trims a section's evidence to its token budget, keeping the best-numbered-first order stable.

    Mirrors `select_within_budget`: a chunk that doesn't fit is skipped rather than ending the selection,
    so a shorter one further down can still use the remaining room. Kept separate because the report packs
    (number, chunk) pairs whose numbers are already fixed bank-wide, while the chat path numbers as it packs.
    """
    from app.features.qa_generation.utils.context_packer import DOC_HEADER_TOKENS

    selected: List[Tuple[int, RetrievedChunk]] = []
    used = 0
    for number, chunk in pairs:
        cost = count_tokens(chunk_source_text(chunk)) + DOC_HEADER_TOKENS
        if used + cost > token_budget:
            continue
        selected.append((number, chunk))
        used += cost
    return selected
