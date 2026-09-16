# Canvas: citation_parser_v3
# Path: app/features/qa_generation/utils/citation_parser.py

import re
import uuid
from typing import List, Dict

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import RAGContextBlock, QACitation
from app.features.qa_generation.utils.context_packer import order_for_reading, strip_repeated_overlap


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def build_context_block(chunks: List[RetrievedChunk]) -> RAGContextBlock:
    """
    Transforms retrieved chunks into a strict, cited text block for the LLM.
    Maintains a mapping of [Doc X] to the original chunk UUID.

    Only the verbatim transcript (`raw_text`) is shown to the model. The model-generated situating
    context embedded in `text_payload` improves retrieval, but it is not source material: if it
    were in the prompt, an answer grounded only in that generated sentence would still receive a
    valid citation. `text_payload` is used only for chunks that carry no separate raw text.

    Chunks are numbered in reading order (grouped by source, chronological within a source), and the sentence
    the chunker repeats between neighbouring chunks is shown only once.
    """
    formatted_parts = []
    doc_mapping: Dict[str, uuid.UUID] = {}
    previous = None

    for idx, chunk in enumerate(order_for_reading(chunks), start=1):
        doc_id = f"[Doc {idx}]"
        doc_mapping[doc_id] = chunk.chunk_id

        source_text = strip_repeated_overlap(previous, chunk)
        previous = chunk
        header = f"{doc_id} ({_format_timestamp(chunk.start_sec)}–{_format_timestamp(chunk.end_sec)})"

        # Format the text injected into the LLM context window
        formatted_parts.append(f"{header}\n{source_text}\n")

    return RAGContextBlock(
        formatted_text="\n".join(formatted_parts),
        doc_mapping=doc_mapping
    )


# Any bracket that starts with "Doc"/"Docs", e.g. [Doc 1], [doc 2], [Doc 1, Doc 3], [Doc 1, 3], [Docs 2-4], [Doc 1 and 2]
_CITATION_GROUP = re.compile(r"\[\s*Docs?\b[^\[\]]*\]", re.IGNORECASE)
_CITATION_NUMBER = re.compile(r"(\d+)\s*[-–—]\s*(\d+)|(\d+)")
# What a citation group may contain besides numbers; anything else means the bracket is not a citation
_CITATION_FILLER = re.compile(r"docs?|and|\d+|[\s,;&.\-–—]", re.IGNORECASE)
_MAX_RANGE_SPAN = 20


def normalize_citation_tags(generated_text: str, doc_count: int) -> str:
    """
    Rewrites every citation group into canonical adjacent tags ("[Doc 1, 3]" -> "[Doc 1][Doc 3]") so the answer
    text and the parsed citations always agree. Numbers outside 1..doc_count point at no evidence and are
    dropped, since a tag that resolves to nothing would look like a citation without being one.
    """
    def _rewrite(match: re.Match) -> str:
        body = match.group(0)[1:-1]
        if _CITATION_FILLER.sub("", body):
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

        valid = [n for n in dict.fromkeys(numbers) if 1 <= n <= doc_count]
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
    normalized_text = normalize_citation_tags(generated_text, len(doc_mapping))
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
                end_sec=chunk.end_sec
            ))

    # Ensure deterministic ordering for the UI based on Document Number
    def extract_doc_num(doc_str: str) -> int:
        match = re.search(r'\d+', doc_str)
        return int(match.group()) if match else 0

    citations.sort(key=lambda c: extract_doc_num(c.doc_id))

    return citations
