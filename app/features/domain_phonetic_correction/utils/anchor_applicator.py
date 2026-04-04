# Canvas: anchor_applicator_v4
# Path: app/features/domain_phonetic_correction/utils/anchor_applicator.py

import re
import logging

logger = logging.getLogger(__name__)


class AnchorPatchApplicator:
    """
    A specialized utility that applies text patches using anchor words.
    It isolates the exact changed substring between an original and a correction,
    allowing it to safely replace the error while preserving the exact case,
    spacing, and punctuation of the surrounding anchor words in the source text.
    """

    @staticmethod
    def apply(text: str, original: str, correction: str) -> str:
        if not original or not correction or original == correction:
            return text

        orig_words = original.split()
        corr_words = correction.split()

        if not orig_words or not corr_words:
            return text

        # 1. Determine common prefix word count (FIXED: Case-Sensitive)
        # Removing .lower() ensures that words with fixed capitalization (e.g., ebitda -> EBITDA)
        # are treated as part of the core correction, rather than being preserved as unchanged anchors.
        prefix_count = 0
        while (prefix_count < len(orig_words) and
               prefix_count < len(corr_words) and
               orig_words[prefix_count] == corr_words[prefix_count]):
            prefix_count += 1

        # 2. Determine common suffix word count (FIXED: Case-Sensitive)
        suffix_count = 0
        while (suffix_count < len(orig_words) - prefix_count and
               suffix_count < len(corr_words) - prefix_count and
               orig_words[-(suffix_count + 1)] == corr_words[-(suffix_count + 1)]):
            suffix_count += 1

        # 3. Extract the core correction words
        if suffix_count == 0:
            core_corr_words = corr_words[prefix_count:]
        else:
            core_corr_words = corr_words[prefix_count:-suffix_count]

        core_corr_str = " ".join(core_corr_words)

        # 4. Create a regex pattern to find the exact original phrase in the text
        # We escape the words and join them with \s+ to handle variable spacing
        escaped_words = [re.escape(w) for w in orig_words]
        pattern_str = r'\s+'.join(escaped_words)
        pattern = re.compile(rf'(?<!\w){pattern_str}(?!\w)', re.IGNORECASE)

        def replacer(match: re.Match) -> str:
            matched_str = match.group(0)

            # Split the matched string while keeping whitespaces to rebuild it precisely
            matched_parts = re.split(r'(\s+)', matched_str)
            expected_parts_len = len(orig_words) * 2 - 1

            # Fallback if complex regex whitespace causes an unexpected split array length
            if len(matched_parts) != expected_parts_len:
                return correction

            # Reconstruct prefix preserving original case and spacing
            prefix_str = ""
            for i in range(prefix_count):
                prefix_str += matched_parts[i * 2]  # The word
                # Safe boundary check
                if i * 2 + 1 < len(matched_parts):
                    prefix_str += matched_parts[i * 2 + 1]  # The trailing space

            # Reconstruct suffix preserving original case and spacing
            suffix_str = ""
            for i in range(suffix_count):
                word_idx = -(1 + i * 2)
                space_idx = -(2 + i * 2)

                suffix_part = matched_parts[word_idx]
                # Safe boundary check
                if abs(space_idx) <= len(matched_parts):
                    suffix_part = matched_parts[space_idx] + suffix_part

                # Prepend to suffix string to maintain order
                suffix_str = suffix_part + suffix_str

            return prefix_str + core_corr_str + suffix_str

        new_text = pattern.sub(replacer, text)

        if new_text != text:
            logger.info(f"Anchor Patch Applied: '{original}' -> '{correction}'")

        return new_text