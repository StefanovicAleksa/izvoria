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
    Extracts the base terms and the chosen sub-domains' anchor words and terms.
    """

    @staticmethod
    def load_terms(industry: str, sub_domains: List[str]) -> List[str]:
        """
        Reads the master glossary and returns its base terms plus the anchor words and terms of every listed
        sub-domain, stripped, de-duplicated and sorted. Returns an empty list if the glossary is missing or unreadable.
        """
        # Centralized path resolution
        glossary_path = settings.GLOSSARY_DIR / f"{industry.lower()}.json"

        if not glossary_path.exists():
            logger.warning(f"Master domain glossary not found at {glossary_path}.")
            return []

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)

            # 1. Grab the base industry terms
            terms = list(manifest_data.get("base_terms", []))

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

            return sorted({term.strip() for term in terms if isinstance(term, str) and term.strip()})

        except Exception as e:
            logger.error(f"Failed to parse JSON dictionary at {glossary_path}: {e}")
            return []
