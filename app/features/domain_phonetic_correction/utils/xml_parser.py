# Canvas: xml_parser_v4
# Path: app/features/domain_phonetic_correction/utils/xml_parser.py

import re
from typing import List
import logging

from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticPatch

logger = logging.getLogger(__name__)


def parse_xml_patches(raw_llm_output: str) -> List[DomainPhoneticPatch]:
    """
    Aggressively parses XML-style <patch> tags using a strict Regex Firewall.
    Tracks identical phantom patches to prevent false-positive warning logs.
    """
    # The Regex Firewall: [^<]*? ensures nested or corrupted tags are strictly rejected
    # Added [^>]* to the tags to gracefully allow LLM-hallucinated attributes (e.g. <patch id="1">)
    pattern = re.compile(
        r'<patch[^>]*>\s*<orig[^>]*>([^<]*?)</orig>\s*<corr[^>]*>([^<]*?)</corr>\s*</patch>',
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.finditer(raw_llm_output)
    patches: List[DomainPhoneticPatch] = []
    identical_skipped = 0

    for match in matches:
        orig_text = match.group(1).strip().replace('\x00', '')
        corr_text = match.group(2).strip().replace('\x00', '')

        if orig_text and corr_text:
            if orig_text != corr_text:
                patches.append(DomainPhoneticPatch(original=orig_text, correction=corr_text))
            else:
                identical_skipped += 1

    # Intelligent Logging
    if not patches and "<patch>" in raw_llm_output.lower():
        if identical_skipped > 0:
            logger.debug(f"Ignored {identical_skipped} LLM patches because original and correction were identical.")
        else:
            logger.warning("Found <patch> tags but failed to parse valid orig/corr pairs. LLM output was malformed.")

    return patches