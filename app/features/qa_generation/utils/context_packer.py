# Path: app/features/qa_generation/utils/context_packer.py

from typing import Callable, Dict, List, Optional

from app.core.anchors.domain.enums import AnchorKind
from app.features.rag_retrieval.domain.entities import RetrievedChunk

# Tokens reserved per document for its "[Doc N] (hh:mm:ss–hh:mm:ss)" header and separators
DOC_HEADER_TOKENS = 24
# A document excerpt's "[Doc N] (p. 7 · Article 7 Termination › 7.2)" header also carries its section path
DOCUMENT_HEADER_TOKENS = 48
# A header's source title is cut to this many words, and each word is reserved this many tokens on top of the above
MAX_TITLE_WORDS = 12
TITLE_TOKENS_PER_WORD = 2
# A shared prefix shorter than this is a coincidence (e.g. "So"), not the chunker's repeated overlap sentence
_MIN_OVERLAP_CHARS = 20


def chunk_source_text(chunk: RetrievedChunk) -> str:
    """The verbatim transcript or document text an answer model may read; the payload only without raw text."""
    return chunk.raw_text or chunk.text_payload


def source_label(chunk: RetrievedChunk) -> str:
    """The chunk's source title as a header shows it: one line, at most MAX_TITLE_WORDS words, no brackets."""
    words = (chunk.source_title or "").replace("[", "(").replace("]", ")").split()
    if len(words) > MAX_TITLE_WORDS:
        return " ".join(words[:MAX_TITLE_WORDS]) + "…"
    return " ".join(words)


def has_document_chunks(chunks) -> bool:
    """True when any chunk is a document excerpt; a context of transcripts alone gets the prompts it always had."""
    return any(chunk.anchor_kind is not AnchorKind.TIME for chunk in chunks)


def header_tokens(chunk: RetrievedChunk) -> int:
    """Tokens a chunk's "[Doc N] <source title> (location)" header costs on top of its text."""
    base = DOC_HEADER_TOKENS if chunk.anchor_kind is AnchorKind.TIME else DOCUMENT_HEADER_TOKENS
    return base + TITLE_TOKENS_PER_WORD * len(source_label(chunk).split())


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
        cost = count_tokens(chunk_source_text(chunk)) + header_tokens(chunk)
        if used + cost > token_budget:
            continue
        selected.append(chunk)
        seen.add(chunk.chunk_id)
        used += cost
    return selected


def order_for_reading(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Groups chunks by source (sources ordered by their best-ranked chunk) and orders each source's chunks by their
    position in it, so the model reads each recording as a continuous timeline and each document in page order
    instead of score-ordered fragments. Chunks built without a position (cached eval candidates) fall back to
    time order.
    """
    best_rank: Dict = {}
    for rank, chunk in enumerate(chunks):
        best_rank.setdefault(chunk.source_id, rank)
    return sorted(chunks, key=lambda c: (best_rank[c.source_id], c.position, c.start_sec, c.end_sec))


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
    Only the first speaker line is inspected; its "[speaker @ time]: " label is kept. Document chunks never
    overlap (they split on the document's own sections), so they are returned untouched.
    """
    text = chunk_source_text(chunk)
    if (previous is None or previous.source_id != chunk.source_id
            or chunk.anchor_kind is not AnchorKind.TIME or previous.anchor_kind is not AnchorKind.TIME
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
