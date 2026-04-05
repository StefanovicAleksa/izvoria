# Canvas: text_segmenter_v2
# Path: app/features/hallucination_scrubber/utils/text_segmenter.py

import re
from typing import List, Optional

from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord


class TextSegmenter:
    """
    Utility for safely splitting transcript segments and filtering their underlying
    word timestamp arrays to maintain precise temporal alignment for Phase 7.
    """

    @staticmethod
    def split_into_clauses(text: str) -> List[str]:
        """
        Splits a text block into evaluatable clauses based on punctuation,
        ensuring we don't send massive paragraphs to the Zero-Shot classifier.
        """
        if not text:
            return []

        # Split by sentence-ending punctuation followed by whitespace
        clauses = re.split(r'(?<=[.!?])\s+', text.strip())
        return [c.strip() for c in clauses if c.strip()]

    @staticmethod
    def reconstruct_segment(
            original_segment: TranscriptSegment,
            surviving_clauses: List[str],
            deleted_clauses: List[str]
    ) -> Optional[TranscriptSegment]:
        """
        Rebuilds the segment and carefully filters out the Timestamped Words
        that belonged to the deleted hallucinations.
        """
        if not surviving_clauses:
            return None

        new_text = " ".join(surviving_clauses)

        # Build a fast lookup string of the deleted text to filter the word array
        # We strip punctuation to ensure a clean match
        deleted_text_blob = " ".join(deleted_clauses).lower()
        deleted_text_blob = re.sub(r'[^\w\s]', '', deleted_text_blob)

        surviving_words: List[TranscriptWord] = []

        for w in original_segment.words:
            # Clean the individual word to check if it was part of the deleted blob
            clean_word = re.sub(r'[^\w\s]', '', w.word.lower())

            # If the word is empty after stripping (e.g., it was just a comma), keep it safely
            if not clean_word:
                surviving_words.append(w)
                continue

            # If the word does NOT appear in the deleted hallucination blob, it survives
            if clean_word not in deleted_text_blob:
                surviving_words.append(w)

        return TranscriptSegment(
            start_sec=original_segment.start_sec,
            end_sec=original_segment.end_sec,
            text=new_text,
            words=surviving_words,
            speaker_id=original_segment.speaker_id
        )