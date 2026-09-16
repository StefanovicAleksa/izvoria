# Path: app/features/qa_generation/utils/context_packer.py

from typing import Callable, Dict, List, Optional

from app.features.rag_retrieval.domain.entities import RetrievedChunk

# Tokens reserved per document for its "[Doc N] (hh:mm:ss–hh:mm:ss)" header and separators
DOC_HEADER_TOKENS = 24
# A shared prefix shorter than this is a coincidence (e.g. "So"), not the chunker's repeated overlap sentence
_MIN_OVERLAP_CHARS = 20


def chunk_source_text(chunk: RetrievedChunk) -> str:
    """The verbatim transcript an answer model may read; the payload only for chunks without raw text."""
    return chunk.raw_text or chunk.text_payload


def select_within_budget(
        chunks: List[RetrievedChunk],
        token_budget: int,
        count_tokens: Callable[[str], int]
) -> List[RetrievedChunk]:
    """
    Keeps chunks in priority order (best first) while they fit the token budget. A chunk that doesn't fit is
    skipped rather than ending the selection, so a shorter lower-ranked chunk can still use the remaining room.
    Duplicate chunk ids are kept once.
    """
    selected: List[RetrievedChunk] = []
    seen = set()
    used = 0
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        cost = count_tokens(chunk_source_text(chunk)) + DOC_HEADER_TOKENS
        if used + cost > token_budget:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        used += cost
    return selected


def order_for_reading(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Groups chunks by source (sources ordered by their best-ranked chunk) and orders each source's chunks by time,
    so the model reads each recording as a continuous timeline instead of score-ordered fragments.
    """
    best_rank: Dict = {}
    for rank, chunk in enumerate(chunks):
        best_rank.setdefault(chunk.source_id, rank)
    return sorted(chunks, key=lambda c: (best_rank[c.source_id], c.start_sec, c.end_sec))


def _overlap_length(previous_text: str, body: str) -> int:
    """Length of the longest suffix of previous_text that starts body, ending on a word boundary."""
    limit = min(len(previous_text), len(body))
    for length in range(limit, _MIN_OVERLAP_CHARS - 1, -1):
        if previous_text.endswith(body[:length]) and (length == len(body) or not body[length].isalnum()):
            return length
    return 0


def strip_repeated_overlap(previous: Optional[RetrievedChunk], chunk: RetrievedChunk) -> str:
    """
    The chunker repeats the last sentence of a chunk at the start of the next one. When both chunks are shown
    back to back, that sentence is dropped from the later chunk so the model doesn't read (and cite) it twice.
    Only the first speaker line is inspected; its "[speaker @ time]: " label is kept.
    """
    text = chunk_source_text(chunk)
    if (previous is None or previous.source_id != chunk.source_id
            or chunk.start_sec > previous.end_sec):
        return text

    first_line, newline, rest = text.partition("\n")
    label, separator, body = first_line.partition("]: ")
    if not separator:
        label, separator, body = "", "", first_line

    previous_text = chunk_source_text(previous).rstrip()
    overlap = _overlap_length(previous_text, body)
    if not overlap:
        return text

    remainder = body[overlap:].lstrip()
    if not remainder:
        return rest if newline else text
    return f"{label}{separator}{remainder}{newline}{rest}"
