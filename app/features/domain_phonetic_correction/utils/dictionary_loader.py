# File: app/features/domain_phonetic_correction/utils/dictionary_loader.py
# Canvas: dictionary_loader_v5
# Path: app/features/domain_phonetic_correction/utils/dictionary_loader.py

import json
import logging
from pathlib import Path  # CRITICAL: Keep this import for unit test patching
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

class DictionaryLoader:
    """
    Utility for loading the master industry JSON glossaries.
    Extracts the base terms and sub-domain jargon to build a static context window for the LLM.
    """

    @staticmethod
    def build_prompt_dictionary(industry: str, sub_domains: List[str]) -> str:
        """
        Reads the master glossary and returns a bulleted string of valid terminology.
        Aggregates terms from multiple sub-domains if provided.
        """
        # Centralized path resolution
        glossary_path = settings.GLOSSARY_DIR / f"{industry.lower()}.json"

        if not glossary_path.exists():
            logger.warning(f"Master domain glossary not found at {glossary_path}.")
            return ""

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)

            # 1. Grab the base industry terms
            terms = manifest_data.get("base_terms", [])

            # 2. Aggregate sub-domain terms
            manifest_sub_domains = manifest_data.get("sub_domains", {})

            for sub_domain in sub_domains:
                if sub_domain in manifest_sub_domains:
                    data = manifest_sub_domains[sub_domain]
                    if isinstance(data, dict):
                        terms.extend(data.get("anchor_words", []))
                        terms.extend(data.get("terms", []))
                    else:
                        terms.extend(data)

            if not terms:
                return ""

            # Deduplicate and sort
            unique_terms = sorted(list(set(terms)))
            return "\n".join([f"- {term.strip()}" for term in unique_terms])

        except Exception as e:
            logger.error(f"Failed to parse JSON dictionary at {glossary_path}: {e}")
            return ""