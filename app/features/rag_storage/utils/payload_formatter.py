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