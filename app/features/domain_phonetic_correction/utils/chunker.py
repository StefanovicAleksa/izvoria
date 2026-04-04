# Canvas: chunker_v10
# Path: app/features/phonetic_correction/utils/chunker.py

from typing import List, TypeVar, Callable

T = TypeVar('T')


def sliding_window_chunker(
        items: List[T],
        text_extractor: Callable[[T], str],
        max_chars: int = 12000,
        overlap_items: int = 2
) -> List[List[T]]:
    """
    Creates a running window of chunks based on character length.
    Expects input items to already be pre-split if they individually exceed max_chars.
    """
    if not items:
        return []

    chunks = []
    i = 0
    n = len(items)

    while i < n:
        current_chunk = []
        current_chars = 0
        idx = i

        while idx < n:
            item_text = text_extractor(items[idx])
            item_len = len(item_text)

            # If adding this item exceeds max_chars AND we already have at least one item
            if current_chars + item_len > max_chars and current_chunk:
                break

            current_chunk.append(items[idx])
            current_chars += item_len
            idx += 1

        chunks.append(current_chunk)

        # Reached the end of the array
        if idx == n:
            break

        # Slide the window back by `overlap_items`, but ensure we always advance by at least 1
        next_start = idx - overlap_items
        i = max(i + 1, next_start)

    return chunks