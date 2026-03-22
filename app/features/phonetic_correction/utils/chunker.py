# Canvas: chunker_v6
# Path: app/features/phonetic_correction/utils/chunker.py

from typing import List, TypeVar

T = TypeVar('T')


def sliding_window_chunker(items: List[T], chunk_size: int = 40, overlap: int = 5) -> List[List[T]]:
    """
    Slices a list into overlapping windows to prevent LLM context loss at chunk boundaries.

    Example (Size 40, Overlap 5):
      Chunk 1: Items 0 to 40
      Chunk 2: Items 35 to 75
      Chunk 3: Items 70 to 110
    """
    if not items:
        return []

    chunks = []
    # Ensure step is at least 1 to prevent infinite loops
    step = max(1, chunk_size - overlap)

    for i in range(0, len(items), step):
        chunk = items[i: i + chunk_size]
        chunks.append(chunk)

        # If this chunk reached the end of the list, stop to prevent trailing duplicate micro-chunks
        if i + chunk_size >= len(items):
            break

    return chunks