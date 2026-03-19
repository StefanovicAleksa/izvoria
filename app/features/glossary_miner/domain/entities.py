# Canvas: glossary_miner_entities_v4
# Path: app/features/glossary_miner/domain/entities.py

from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict

# ==========================================
# WIKIDATA EXTRACTION ENTITIES
# ==========================================

class WikidataItem(BaseModel):
    """
    Represents a raw entity extracted from Wikidata's graph.
    Aliases are natively grouped by their ISO language code directly from the SPARQL query.
    """
    item_id: str = Field(description="The Q-identifier from Wikidata (e.g., Q32098)")
    label: str = Field(description="The primary English name of the term.")
    aliases_by_lang: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="A dictionary mapping language codes to aliases (e.g., {'en': ['Heart Attack'], 'la': ['Infarctus myocardii']})."
    )

    model_config = ConfigDict(frozen=True)


# ==========================================
# ONYX APPLIANCE GLOSSARY MANIFEST ENTITIES
# ==========================================

class SubDomainGlossary(BaseModel):
    """
    Represents the specialized terminology and anchor words for a specific sub-domain.
    Formatted strictly for Whisper's initial_prompt requirements.
    """
    anchor_words: List[str] = Field(
        default_factory=list,
        description="Highly unique words that deterministically prove this sub-domain is being discussed."
    )
    terms: List[str] = Field(
        default_factory=list,
        description="The broader list of jargon, acronyms, and proper nouns for Whisper phonetic matching."
    )

class OnyxGlossaryManifest(BaseModel):
    """
    The ultimate Aggregate Root for the Onyx Appliance's Domain Adaptation feature.
    This schema perfectly mirrors what the Phase 8 `GlossaryBuilder` expects to load.
    """
    base_terms: List[str] = Field(
        default_factory=list,
        description="General industry terminology applicable across all sub-domains."
    )
    sub_domains: Dict[str, SubDomainGlossary] = Field(
        default_factory=dict,
        description="A mapping of specific sub-domain names to their dedicated glossaries."
    )