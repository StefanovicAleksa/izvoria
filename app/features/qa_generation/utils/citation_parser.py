# Canvas: citation_parser_v3
# Path: app/features/qa_generation/utils/citation_parser.py

import re
import uuid
from typing import List, Dict

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import RAGContextBlock, QACitation


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
    """
    formatted_parts = []
    doc_mapping: Dict[str, uuid.UUID] = {}

    for idx, chunk in enumerate(chunks, start=1):
        doc_id = f"[Doc {idx}]"
        doc_mapping[doc_id] = chunk.chunk_id

        source_text = chunk.raw_text or chunk.text_payload
        header = f"{doc_id} ({_format_timestamp(chunk.start_sec)}–{_format_timestamp(chunk.end_sec)})"

        # Format the text injected into the LLM context window
        formatted_parts.append(f"{header}\n{source_text}\n")

    return RAGContextBlock(
        formatted_text="\n".join(formatted_parts),
        doc_mapping=doc_mapping
    )


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

    # Extract all unique citation tags matching the pattern [Doc 1], [Doc 2], etc.
    found_tags = set(re.findall(r'\[Doc \d+]', generated_text))

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
