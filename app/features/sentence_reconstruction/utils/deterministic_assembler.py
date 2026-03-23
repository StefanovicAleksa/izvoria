# Canvas: deterministic_assembler_v2
# Path: app/features/sentence_reconstruction/utils/deterministic_assembler.py

import uuid
from typing import List

from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord, ReconstructedSentence


def assemble_sentences(speaker_id: str, punctuated_words: List[PunctuatedWord]) -> List[ReconstructedSentence]:
    """
    Deterministically groups a stream of tagged words into grammatically complete sentences.
    Closes a sentence precisely when a terminal punctuation mark is encountered.
    """
    if not punctuated_words:
        return []

    sentences: List[ReconstructedSentence] = []
    current_words: List[TranscriptWord] = []
    current_text_parts: List[str] = []

    # Standard terminal punctuation marks that indicate a sentence boundary
    TERMINAL_PUNCTUATION = {".", "?", "!"}

    # Punctuation that attaches directly to the preceding word without a space
    ATTACHED_PUNCTUATION = {".", "?", "!", ",", ":", ";", "-"}
    ATTACHED_CHARS = "".join(ATTACHED_PUNCTUATION)

    for p_word in punctuated_words:
        word_obj = p_word.acoustic_word
        tag = p_word.punctuation_tag

        # Always retain the unaltered acoustic word in the audit trail
        current_words.append(word_obj)

        # 1. Clean the acoustic string
        # Strip trailing/leading spaces (fixes double-spacing)
        # Strip existing punctuation (fixes "passage??" duplicates)
        clean_word = word_obj.word.strip().rstrip(ATTACHED_CHARS)

        if not clean_word:
            # Edge case: Whisper emitted a word that was literally just a space or punctuation
            # We still need to process its tag to close sentences, but we skip appending text.
            if tag in TERMINAL_PUNCTUATION and current_text_parts:
                current_text_parts[-1] = current_text_parts[-1].rstrip() + tag
                sentences.append(
                    ReconstructedSentence(
                        sentence_id=uuid.uuid4(),
                        text="".join(current_text_parts).strip(),
                        speaker_id=speaker_id,
                        start_sec=current_words[0].start_sec,
                        end_sec=current_words[-1].end_sec,
                        words=list(current_words)
                    )
                )
                current_words.clear()
                current_text_parts.clear()
            continue

        # 2. Append the clean word
        if not current_text_parts:
            # Capitalize the first word of a new sentence
            current_text_parts.append(clean_word.capitalize())
        else:
            current_text_parts.append(f" {clean_word}")

        # 3. Apply the Neural Punctuation Tag
        if tag != "NONE":
            if tag in ATTACHED_PUNCTUATION:
                current_text_parts[-1] = current_text_parts[-1].rstrip() + tag
            else:
                current_text_parts.append(tag)

        # 4. If we hit a terminal tag, seal the sentence instantly
        if tag in TERMINAL_PUNCTUATION:
            sentences.append(
                ReconstructedSentence(
                    sentence_id=uuid.uuid4(),
                    text="".join(current_text_parts).strip(),
                    speaker_id=speaker_id,
                    start_sec=current_words[0].start_sec,
                    end_sec=current_words[-1].end_sec,
                    words=list(current_words)
                )
            )
            # Reset the accumulators for the next sentence
            current_words.clear()
            current_text_parts.clear()

    # 5. Flush any remaining words that didn't get a terminal punctuation mark
    if current_words and current_text_parts:
        # Force a period at the end of the final orphaned fragment
        if not "".join(current_text_parts).endswith(tuple(TERMINAL_PUNCTUATION)):
            current_text_parts[-1] = current_text_parts[-1].rstrip() + "."

        sentences.append(ReconstructedSentence(
                sentence_id=uuid.uuid4(),
                text="".join(current_text_parts).strip(),
                speaker_id=speaker_id,
                start_sec=current_words[0].start_sec,
                end_sec=current_words[-1].end_sec,
                words=list(current_words)
            )
        )

    return sentences