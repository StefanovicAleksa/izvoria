# Canvas: phonetic_text_sanitizer_v3
# Path: app/features/domain_phonetic_correction/utils/text_sanitizer.py

import re
import logging

logger = logging.getLogger(__name__)

class PhoneticTextSanitizer:
    @staticmethod
    def sanitize(text: str) -> str:
        """
        Strips rogue '0' tokens hallucinated by upstream ASR before they
        can confuse the Phonetic Correction LLM.
        """
        if not text:
            return text

        original_text = text

        # 1. Remove '0' attached to the END of an alphabetical word
        # FIX: We use ([a-zA-Z]{2,}) to capture words with AT LEAST 2 letters.
        # This fixes "drill0" and "So0" but safely ignores "S0".
        cleaned_text = re.sub(r'([a-zA-Z]{2,})0(?=\s|$|[,.!?])', r'\1', original_text)

        # 2. Remove '0' attached to the START of an alphabetical word
        # FIX: Again, we enforce the 2-letter minimum to avoid destroying things like "0A"
        cleaned_text = re.sub(r'(^|\s)0([a-zA-Z]{2,})', r'\1\2', cleaned_text)

        # 3. Clean up any accidental double spaces created
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        if cleaned_text != original_text:
            logger.debug("Sanitized rogue '0' tokens from transcript chunk.")

        return cleaned_text