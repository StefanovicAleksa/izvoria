# Canvas: xml_parser_v3
# Path: app/features/domain_phonetic_correction/utils/xml_parser.py

import re
from typing import List
import logging

from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticPatch

logger = logging.getLogger(__name__)


def parse_xml_patches(raw_llm_output: str) -> List[DomainPhoneticPatch]:
    """
    Aggressively parses XML-style <patch> tags using Regex.
    Extracts the <orig> and <corr> strings to feed the AnchorApplicator.
    """
    # THE FIREWALL: By using [^<]*? instead of .*?, we guarantee that nested
    # or hallucinated XML tags CANNOT be captured inside the text.
    pattern = re.compile(
        r'<patch>\s*<orig>([^<]*?)</orig>\s*<corr>([^<]*?)</corr>\s*</patch>',
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.finditer(raw_llm_output)
    patches: List[DomainPhoneticPatch] = []

    for match in matches:
        orig_text = match.group(1).strip()
        corr_text = match.group(2).strip()

        # Strip null terminators just in case llama.cpp leaks them across the C++ boundary
        orig_text = orig_text.replace('\x00', '')
        corr_text = corr_text.replace('\x00', '')

        if orig_text and corr_text and orig_text != corr_text:
            patches.append(DomainPhoneticPatch(original=orig_text, correction=corr_text))

    if not patches and "<patch>" in raw_llm_output.lower():
        logger.warning(f"Found <patch> tags but failed to parse valid orig/corr pairs. LLM output was malformed.")

    return patches