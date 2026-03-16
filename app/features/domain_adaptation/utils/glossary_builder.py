# Canvas: glossary_builder_v1
# Path: app/features/domain_adaptation/utils/glossary_builder.py

import json
from pathlib import Path

from app.features.domain_adaptation.exceptions.process import GlossaryManifestError


class GlossaryBuilder:
    """
    Handles File I/O for the domain manifests and executes the
    Anchor Validation business logic.
    """
    HOTWORD_WEIGHT = 15.0

    @classmethod
    def validate_and_build(cls, industry: str, guessed_subdomain: str, scouted_text: str) -> tuple[str, str]:
        """
        Validates the AI's guess against the transcript and builds the Whisper hotword string.
        Returns: (final_subdomain, hotword_string)
        """
        manifest_data = cls._load_master_glossary(industry)
        base_terms = manifest_data.get("base_terms", [])
        subdomains = manifest_data.get("sub_domains", {})

        # Edge Case: The API passes "GENERAL" if the Scout heard nothing
        if guessed_subdomain == "GENERAL" or guessed_subdomain not in subdomains:
            return "GENERAL", cls._format_hotwords(base_terms)

        anchor_words = subdomains[guessed_subdomain].get("anchor_words", [])
        domain_terms = subdomains[guessed_subdomain].get("terms", [])

        text_lower = scouted_text.lower()
        is_valid = any(anchor.lower() in text_lower for anchor in anchor_words)

        if not is_valid:
            print(f"[DOMAIN ADAPTATION] Anchor validation failed for {guessed_subdomain}. Falling back to Base.")
            return "BASE_FALLBACK", cls._format_hotwords(base_terms)

        # Merge base terms with specific subdomain jargon and the anchors themselves
        final_terms = base_terms + domain_terms + anchor_words
        return guessed_subdomain, cls._format_hotwords(final_terms)

    @classmethod
    def get_candidate_labels(cls, industry: str) -> list[str]:
        """Extracts just the keys (e.g., ['UK_MARKETS', 'OPTIONS_TRADING']) for the Classifier."""
        manifest_data = cls._load_master_glossary(industry)
        return list(manifest_data.get("sub_domains", {}).keys())

    @classmethod
    def _format_hotwords(cls, terms: list[str]) -> str:
        unique_terms = list(set(terms))
        return ", ".join([f"{term.strip()}:{cls.HOTWORD_WEIGHT}" for term in unique_terms])

    @staticmethod
    def _load_master_glossary(industry: str) -> dict:
        file_name = f"{industry.lower()}.json"
        root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        glossary_path = root_dir / "data" / "glossaries" / file_name

        if not glossary_path.exists():
            raise GlossaryManifestError(f"Master domain glossary not found at {glossary_path}.")

        with open(glossary_path, 'r', encoding='utf-8') as f:
            return json.load(f)