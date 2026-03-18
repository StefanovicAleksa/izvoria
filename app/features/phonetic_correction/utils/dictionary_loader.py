# Canvas: dictionary_loader_v1
# Path: app/features/phonetic_correction/utils/dictionary_loader.py

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DictionaryLoader:
    """
    Utility for loading the master industry JSON glossaries.
    Extracts the base terms and sub-domain jargon to build a static context window for the LLM.
    """

    @staticmethod
    def build_prompt_dictionary(industry: str, sub_domain: str) -> str:
        """
        Reads the master glossary and returns a bulleted string of valid terminology.
        """
        # Match the path logic used in Phase 5 Domain Adaptation
        root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        glossary_path = root_dir / "data" / "glossaries" / f"{industry.lower()}.json"

        if not glossary_path.exists():
            logger.warning(f"Master domain glossary not found at {glossary_path}.")
            return ""

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)

            # 1. Grab the base industry terms
            terms = manifest_data.get("base_terms", [])

            # 2. Add the specific sub-domain terms and anchors if they exist
            sub_domains = manifest_data.get("sub_domains", {})
            if sub_domain in sub_domains:
                terms.extend(sub_domains[sub_domain].get("anchor_words", []))
                terms.extend(sub_domains[sub_domain].get("terms", []))

            if not terms:
                return ""

            # Deduplicate, sort, and format into a clean string list for the LLM
            unique_terms = sorted(list(set(terms)))
            return "\n".join([f"- {term.strip()}" for term in unique_terms])

        except Exception as e:
            logger.error(f"Failed to parse JSON dictionary at {glossary_path}: {e}")
            return ""