# Canvas: citation_parser_v3
# Path: app/features/qa_generation/utils/citation_parser.py

import re
import uuid
from functools import lru_cache
from typing import Collection, Dict, List, Optional, Tuple

from app.core.anchors.utils.formatting import format_location
from app.core.language.service.language_api import get_language_pack
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import RAGContextBlock, QACitation
from app.features.qa_generation.utils.context_packer import order_for_reading, strip_repeated_overlap


def format_chunk_location(chunk: RetrievedChunk) -> str:
    """Where a chunk sits in its source, as a citation header shows it: a time range, or a page and section."""
    return format_location(chunk.anchor_kind, chunk.start_sec, chunk.end_sec,
                           chunk.page_start, chunk.page_end, chunk.section_path)


def build_numbered_context_block(numbered_chunks: List[Tuple[int, RetrievedChunk]]) -> RAGContextBlock:
    """
    Renders chunks that already carry their document numbers, in the given order. A multi-turn conversation
    numbers documents once for the whole conversation, so a turn's block can start past [Doc 1] or have gaps.

    Only the verbatim transcript (`raw_text`) is shown to the model. The model-generated situating
    context embedded in `text_payload` improves retrieval, but it is not source material: if it
    were in the prompt, an answer grounded only in that generated sentence would still receive a
    valid citation. `text_payload` is used only for chunks that carry no separate raw text.

    The sentence the chunker repeats between neighbouring chunks is shown only once.
    """
    formatted_parts = []
    doc_mapping: Dict[str, uuid.UUID] = {}
    previous = None

    for number, chunk in numbered_chunks:
        doc_id = f"[Doc {number}]"
        doc_mapping[doc_id] = chunk.chunk_id

        source_text = strip_repeated_overlap(previous, chunk)
        previous = chunk
        location = format_chunk_location(chunk)
        header = f"{doc_id} ({location})" if location else doc_id

        # Format the text injected into the LLM context window
        formatted_parts.append(f"{header}\n{source_text}\n")

    return RAGContextBlock(
        formatted_text="\n".join(formatted_parts),
        doc_mapping=doc_mapping
    )


def number_for_reading(chunks: List[RetrievedChunk], start_number: int = 1) -> List[Tuple[int, RetrievedChunk]]:
    """Numbers chunks in reading order (grouped by source, chronological within a source)."""
    return list(enumerate(order_for_reading(chunks), start=start_number))


def build_context_block(chunks: List[RetrievedChunk], start_number: int = 1) -> RAGContextBlock:
    """
    Transforms retrieved chunks into a strict, cited text block for the LLM.
    Maintains a mapping of [Doc X] to the original chunk UUID.

    Chunks are numbered in reading order from start_number (see build_numbered_context_block).
    """
    return build_numbered_context_block(number_for_reading(chunks, start_number))


def doc_numbers(doc_mapping: Dict[str, uuid.UUID]) -> set:
    """The document numbers a doc_mapping's "[Doc N]" keys refer to."""
    return {int(m.group()) for m in (re.search(r"\d+", key) for key in doc_mapping) if m}


# Any bracket that starts with "Doc"/"Docs", e.g. [Doc 1], [doc 2], [Doc 1, Doc 3], [Doc 1, 3], [Docs 2-4], [Doc 1 and 2]
_CITATION_GROUP = re.compile(r"\[\s*Docs?\b[^\[\]]*\]", re.IGNORECASE)
_CITATION_NUMBER = re.compile(r"(\d+)\s*[-–—]\s*(\d+)|(\d+)")


@lru_cache(maxsize=8)
def _citation_filler(joiners: Tuple[str, ...]) -> re.Pattern:
    """
    What a citation group may contain besides numbers; anything else means the bracket is not a citation. The
    joining words are the language pack's ("and", "i", "in") plus English, the language the prompts are written in.
    """
    words = "|".join(re.escape(word) for word in sorted({"and", *joiners}))
    return re.compile(rf"docs?|\b(?:{words})\b|\d+|[\s,;&.\-–—]", re.IGNORECASE)
_MAX_RANGE_SPAN = 20


def normalize_citation_tags(
        generated_text: str,
        doc_count: int,
        valid_numbers: Optional[Collection[int]] = None
) -> str:
    """
    Rewrites every citation group into canonical adjacent tags ("[Doc 1, 3]" -> "[Doc 1][Doc 3]") so the answer
    text and the parsed citations always agree. Numbers outside 1..doc_count (or, when given, not in
    valid_numbers) point at no evidence and are dropped, since a tag that resolves to nothing would look like a
    citation without being one.
    """
    filler = _citation_filler(tuple(get_language_pack().text.citation_joiners))

    def _rewrite(match: re.Match) -> str:
        body = match.group(0)[1:-1]
        if filler.sub("", body):
            return match.group(0)

        numbers: List[int] = []
        for range_start, range_end, single in _CITATION_NUMBER.findall(body):
            if single:
                numbers.append(int(single))
                continue
            low, high = int(range_start), int(range_end)
            if low <= high and high - low <= _MAX_RANGE_SPAN:
                numbers.extend(range(low, high + 1))
            else:
                numbers.extend([low, high])

        valid = [n for n in dict.fromkeys(numbers)
                 if (n in valid_numbers if valid_numbers is not None else 1 <= n <= doc_count)]
        return "".join(f"[Doc {n}]" for n in valid)

    return _CITATION_GROUP.sub(_rewrite, generated_text)


def parse_citations(
        generated_text: str,
        chunks: List[RetrievedChunk],
        doc_mapping: Dict[str, uuid.UUID]
) -> List[QACitation]:
    """
    Scans the LLM's generated response for citation tags and maps them
    back to the physical video clips for the UI to render as clickable links.
    """
    citations: List[QACitation] = []

    # Grouped forms ([Doc 1, 3], [Docs 1-2]) are expanded first so every cited document is found
    normalized_text = normalize_citation_tags(generated_text, len(doc_mapping), doc_numbers(doc_mapping))
    found_tags = set(re.findall(r'\[Doc \d+]', normalized_text))

    # O(1) lookup map for chunks
    chunk_lookup = {c.chunk_id: c for c in chunks}

    for tag in found_tags:
        chunk_id = doc_mapping.get(tag)
        if chunk_id and chunk_id in chunk_lookup:
            chunk = chunk_lookup[chunk_id]

            # The engine resolves the clip path from these storage links after the pipeline returns. A
            # placeholder string here used to count as "already resolved", so no citation ever got a path.
            citations.append(QACitation(
                doc_id=tag,
                chunk_id=chunk_id,
                source_id=chunk.source_id,
                extracted_file_id=chunk.extracted_file_id,
                physical_path=getattr(chunk, 'physical_path', None),
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                anchor_kind=chunk.anchor_kind,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path
            ))

    # Ensure deterministic ordering for the UI based on Document Number
    def extract_doc_num(doc_str: str) -> int:
        match = re.search(r'\d+', doc_str)
        return int(match.group()) if match else 0

    citations.sort(key=lambda c: extract_doc_num(c.doc_id))

    return citations
