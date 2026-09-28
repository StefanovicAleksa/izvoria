# File: app/features/domain_adaptation/utils/glossary_builder.py
# Canvas: glossary_builder_v4

import json
import logging
from typing import Dict, List, Tuple

from app.core.config import settings
from app.features.domain_adaptation.exceptions.process import GlossaryManifestError

logger = logging.getLogger(__name__)

class GlossaryBuilder:
    """
    Handles JSON glossary parsing and formats token-safe injection strings
    for the Whisper Large model to prevent CTranslate2 OOM crashes.
    """
    MAX_ANCHOR_WORDS = 40
    MAX_DESCRIPTION_TERMS = 10

    @classmethod
    def build_injection_payloads(cls, industry: str, predicted_domains: List[str]) -> Tuple[str, str]:
        """
        Returns: (hotword_string, initial_prompt_string)

        The initial prompt is always empty. faster-whisper feeds hotwords and the initial prompt to the model as
        "previous transcript", and the sentence prompt ("This recording discusses ..., including terms such as ...")
        was copied into transcripts as if it had been spoken. The hotword list alone carries the vocabulary bias.
        """
        manifest_data = cls._load_master_glossary(industry)
        base_terms = manifest_data.get("base_terms", [])
        sub_domains_dict = manifest_data.get("sub_domains", {})

        consolidated_anchors = []

        # 1. Gather all high-value anchors from the detected sub-domains
        for domain in predicted_domains:
            if domain in sub_domains_dict:
                domain_anchors = sub_domains_dict[domain].get("anchor_words", [])
                consolidated_anchors.extend(domain_anchors)

        if not consolidated_anchors:
            consolidated_anchors = base_terms

        # 2. Deduplicate and apply token safety limit
        unique_anchors = list(dict.fromkeys(consolidated_anchors))
        safe_anchors = unique_anchors[:cls.MAX_ANCHOR_WORDS]

        # 3. Hotwords (the only text injected into Whisper's prompt)
        return cls._format_hotwords(safe_anchors), ""

    @classmethod
    def get_candidate_labels(cls, industry: str) -> List[str]:
        manifest_data = cls._load_master_glossary(industry)
        return list(manifest_data.get("sub_domains", {}).keys())

    @classmethod
    def get_candidate_descriptions(cls, industry: str) -> Dict[str, str]:
        """Sub-domain label -> short description (its first anchor words and terms), used by the router to score topics."""
        descriptions: Dict[str, str] = {}
        for label, entry in cls._load_master_glossary(industry).get("sub_domains", {}).items():
            vocabulary = entry.get("anchor_words", []) + entry.get("terms", []) if isinstance(entry, dict) else list(entry)
            descriptions[label] = ", ".join(
                term.strip() for term in vocabulary[:cls.MAX_DESCRIPTION_TERMS] if term.strip()
            )
        return descriptions

    @classmethod
    def _format_hotwords(cls, terms: List[str]) -> str:
        # faster-whisper feeds `hotwords` into the decoder prompt as plain text; there is no weighting
        # syntax, so "term:15.0" put literal ":15.0" noise into every window's prompt.
        return ", ".join([term.strip() for term in terms if term.strip()])

    @staticmethod
    def _load_master_glossary(industry: str) -> dict:
        file_name = f"{industry.lower()}.json"
        # FIX: Use centralized settings instead of brittle parent crawling
        glossary_path = settings.GLOSSARY_DIR / file_name

        if not glossary_path.exists():
            if industry.upper() == "GENERAL":
                logger.warning(f"Master glossary missing at {glossary_path}. Using empty fallback.")
                return {"base_terms": [], "sub_domains": {}}
            raise GlossaryManifestError(
                f"Master domain glossary not found at {glossary_path}. Glossaries are per language: build one for "
                f"'{settings.APPLIANCE_LANGUAGE}' there (scripts/run_miner.py) or set APPLIANCE_INDUSTRY=GENERAL."
            )

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise GlossaryManifestError(f"Corrupted JSON in {file_name}: {str(e)}")
