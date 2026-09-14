# Path: app/features/contextual_retrieval/utils/document_window.py

from typing import List, Tuple


def build_document_sections(token_counts: List[int], budget_tokens: int) -> List[Tuple[int, int]]:
    """
    Splits a transcript that is too long for the context model into overlapping sections.

    Args:
        token_counts: token cost of each sentence, in document order.
        budget_tokens: maximum tokens a section may hold.

    Returns:
        `[(start, end), ...]` half-open sentence index ranges. Each section is filled up to the
        budget (always at least one sentence), and the next one starts roughly half a budget
        later, so every chunk sits comfortably inside at least one section. Consecutive chunks
        share a section — and therefore an identical prompt prefix the KV cache can reuse —
        until the section changes.
    """
    n = len(token_counts)
    if n == 0:
        return []
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive.")

    sections: List[Tuple[int, int]] = []
    start = 0
    while True:
        end, used = start, 0
        while end < n and (end == start or used + token_counts[end] <= budget_tokens):
            used += token_counts[end]
            end += 1
        sections.append((start, end))

        if end >= n:
            break

        next_start, advanced = start, 0
        while next_start < end and advanced < budget_tokens // 2:
            advanced += token_counts[next_start]
            next_start += 1
        start = max(start + 1, next_start)

    return sections


def assign_section(sections: List[Tuple[int, int]], first_idx: int, last_idx: int) -> int:
    """
    Picks the section a chunk (sentences first_idx..last_idx inclusive) should be situated in:
    the section that fully contains it with the most margin on both sides; if none contains it,
    the section with the largest overlap.
    """
    if not sections:
        raise ValueError("sections must not be empty.")

    best_idx = 0
    best_key: Tuple[int, int] = (-1, -1)
    for idx, (start, end) in enumerate(sections):
        if start <= first_idx and last_idx < end:
            key = (1, min(first_idx - start, end - 1 - last_idx))
        else:
            key = (0, max(0, min(end, last_idx + 1) - max(start, first_idx)))
        if key > best_key:
            best_idx, best_key = idx, key
    return best_idx
