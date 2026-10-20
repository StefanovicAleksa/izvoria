# Canvas: rag_storage_payload_formatter_v1
# Path: app/features/rag_storage/utils/payload_formatter.py

import re
import unicodedata

def normalize_payload_text(text: str) -> str:
    """
    Cleans and normalizes the unified text payload before indexing.
    Ensures that invisible unicode characters, zero-width spaces,
    and excessive whitespaces do not skew BM25 tokenization or dense embeddings.
    """
    if not text:
        return ""

    # 1. Normalize unicode (NFKC combines characters properly, e.g., fi ligatures to f i)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove zero-width spaces and other invisible formatting characters
    text = text.replace('\u200b', '').replace('\ufeff', '')

    # 3. Collapse multiple spaces, newlines, and tabs into a single space
    text = re.sub(r'\s+', ' ', text)

    # 4. Strip leading/trailing whitespace
    return text.strip()

# The first line of a chunk's indexed context names the source it is from, so a question that names the source
# ("lecture 4 of Stat 110") finds its chunks. Context only: it is never part of the citable raw text.
SOURCE_LINE_PREFIX = "[Source:"


def _unbracketed(text: str) -> str:
    # The line must end at its own "]": RetrievedChunk.content_payload strips it up to the first one
    return (text or "").strip().replace("[", "(").replace("]", ")")


def source_line(title: str, topic: str = "") -> str:
    """`[Source: Statistics 110, Lecture 4 — Conditional probability and Bayes' rule.]`"""
    title, topic = _unbracketed(title), _unbracketed(topic)
    return f"{SOURCE_LINE_PREFIX} {title} — {topic}]" if topic else f"{SOURCE_LINE_PREFIX} {title}]"


def with_source_line(context_text: str, line: str) -> str:
    """A chunk's context with `line` as its source line, replacing any it had."""
    kept = [entry for entry in (context_text or "").split("\n") if not entry.startswith(SOURCE_LINE_PREFIX)]
    return "\n".join([line, *kept]).strip("\n")


def build_payload(context_text: str, raw_text: str) -> str:
    """What is embedded and keyword-indexed: the generated context, then the source text (as assembled at ingestion)."""
    return f"{context_text}\n\n{raw_text}" if context_text else raw_text
