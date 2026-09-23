# Canvas: phonetic_text_sanitizer_v3
# Path: app/features/domain_phonetic_correction/utils/text_sanitizer.py

import re
import logging

from app.core.language.service.language_api import get_language_pack

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
        # Letters of the appliance's language ("a-z" for English), matched in either case
        letters = f"[{get_language_pack().phonetic.alphabet}]"

        # 1. Remove '0' attached to the END of an alphabetical word
        # FIX: We capture words with AT LEAST 2 letters.
        # This fixes "drill0" and "So0" but safely ignores "S0".
        cleaned_text = re.sub(rf'({letters}{{2,}})0(?=\s|$|[,.!?])', r'\1', original_text, flags=re.IGNORECASE)

        # 2. Remove '0' attached to the START of an alphabetical word
        # FIX: Again, we enforce the 2-letter minimum to avoid destroying things like "0A"
        cleaned_text = re.sub(rf'(^|\s)0({letters}{{2,}})', r'\1\2', cleaned_text, flags=re.IGNORECASE)

        # 3. Clean up any accidental double spaces created
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        if cleaned_text != original_text:
            logger.debug("Sanitized rogue '0' tokens from transcript chunk.")

        return cleaned_text