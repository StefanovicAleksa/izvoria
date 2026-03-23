# Canvas: domain_adaptation_glossary_builder_v2
# Path: app/features/domain_adaptation/utils/glossary_builder.py

import json
from pathlib import Path
from typing import List, Tuple

from app.features.domain_adaptation.exceptions.process import GlossaryManifestError
from app.features.domain_adaptation.data.prompts import INITIAL_PROMPT_TEMPLATE


class GlossaryBuilder:
    """
    Handles JSON glossary parsing and formats token-safe injection strings
    for the Whisper Large model to prevent CTranslate2 OOM crashes.
    """
    HOTWORD_WEIGHT = 15.0
    MAX_ANCHOR_WORDS = 40

    @classmethod
    def build_injection_payloads(cls, industry: str, predicted_domains: List[str]) -> Tuple[str, str]:
        """
        Returns: (hotword_string, initial_prompt_string)
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

        # 3. Build Payload A: Hotwords (Beam Search Bias)
        hotword_string = cls._format_hotwords(safe_anchors)

        # 4. Build Payload B: Initial Prompt (from imported template)
        # Fallback to "General" if the array is empty to prevent weird grammar
        domain_names = ", ".join([d.replace("_", " ").title() for d in predicted_domains]) or "General"

        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(
            domain_names=domain_names
        )

        return hotword_string, initial_prompt

    @classmethod
    def get_candidate_labels(cls, industry: str) -> List[str]:
        manifest_data = cls._load_master_glossary(industry)
        return list(manifest_data.get("sub_domains", {}).keys())

    @classmethod
    def _format_hotwords(cls, terms: List[str]) -> str:
        return ", ".join([f"{term.strip()}:{cls.HOTWORD_WEIGHT}" for term in terms])

    @staticmethod
    def _load_master_glossary(industry: str) -> dict:
        file_name = f"{industry.lower()}.json"
        root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        glossary_path = root_dir / "data" / "glossaries" / file_name

        if not glossary_path.exists():
            raise GlossaryManifestError(f"Master domain glossary not found at {glossary_path}.")

        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise GlossaryManifestError(f"Corrupted JSON in {file_name}: {str(e)}")