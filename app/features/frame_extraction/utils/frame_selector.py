# Path: app/features/frame_extraction/utils/frame_selector.py

from pathlib import Path
from typing import List, Tuple

# (timestamp_sec, path, quality_score, phash_hex)
ScoredFrame = Tuple[float, Path, float, str]


def phash_distance(hash_a: str, hash_b: str) -> int:
    """Hamming distance between two hex-encoded perceptual hashes."""
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def select_diverse_frames(frames: List[ScoredFrame], max_frames: int) -> Tuple[List[ScoredFrame], List[ScoredFrame]]:
    """
    Caps a chunk's frames at `max_frames`, keeping the most visually distinct ones.

    Greedy farthest-point selection on pHash distance: start from the sharpest frame, then repeatedly add
    the frame farthest from everything already chosen (ties go to the sharper frame). A chunk that cuts
    between a slide, a close-up and a chart keeps all three rather than consecutive seconds of one slide.
    Returns (selected in chronological order, dropped).
    """
    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames), []

    candidates = list(range(len(frames)))
    first = max(candidates, key=lambda i: frames[i][2])
    chosen = [first]
    candidates.remove(first)

    # Distance from each candidate to its nearest chosen frame
    nearest = {i: phash_distance(frames[i][3], frames[first][3]) for i in candidates}

    while len(chosen) < max_frames and candidates:
        best = max(candidates, key=lambda i: (nearest[i], frames[i][2]))
        chosen.append(best)
        candidates.remove(best)
        for i in candidates:
            nearest[i] = min(nearest[i], phash_distance(frames[i][3], frames[best][3]))

    chosen_set = set(chosen)
    selected = sorted((frames[i] for i in chosen), key=lambda frame: frame[0])
    dropped = [frames[i] for i in range(len(frames)) if i not in chosen_set]
    return selected, dropped
