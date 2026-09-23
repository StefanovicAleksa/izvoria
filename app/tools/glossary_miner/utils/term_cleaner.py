# Canvas: glossary_miner_term_cleaner_v1
# Path: app/features/glossary_miner/utils/term_cleaner.py

import re
from typing import List, Optional, Set

from app.core.language.service.language_api import get_language_pack
from app.core.language.utils.script import normalize_script
from app.tools.glossary_miner.domain.entities import WikidataItem


def clean_and_extract_terms(raw_items: List[WikidataItem]) -> List[str]:
    """
    Processes raw Wikidata entities into a flat, deduplicated, Whisper-safe list of strings.
    Applies strict regex and length heuristics.
    """
    valid_terms: Set[str] = set()
    # Terms are written the way the transcript is (Serbian Wikidata labels are mostly Cyrillic; transcripts Latin)
    script = get_language_pack().transcription.script

    for item in raw_items:
        # 1. Add the primary label, in the appliance's language
        _add_if_valid(item.label, valid_terms, script)

        # 2. Iterate through every alias language the miner collected
        for aliases in item.aliases_by_lang.values():
            for alias in aliases:
                _add_if_valid(alias, valid_terms, script)

    # Return a sorted list for deterministic JSON output
    return sorted(list(valid_terms))


def _add_if_valid(term: str, collection: Set[str], script: Optional[str] = None) -> None:
    """Internal heuristic engine to validate and clean a single term."""
    if not term:
        return
    term = normalize_script(term, script)

    # 1. Length Constraint: Whisper hotwords degrade if they are long phrases.
    # Count words by splitting on spaces.
    words = term.split()
    if len(words) > 3:
        return

    # 2. Character Constraint: Remove brackets, parentheses, and their contents.
    # e.g., "Acute Infarction (disorder)" -> "Acute Infarction"
    cleaned_term = re.sub(r'\(.*?\)|\[.*?]', '', term)

    # 3. Strip commas and colons, as they break Whisper's comma-separated initial_prompt
    cleaned_term = cleaned_term.replace(',', '').replace(':', '')

    # 4. Normalize whitespace
    cleaned_term = " ".join(cleaned_term.split())

    # 5. Ascii/Alphabet Check: Ensure no weird symbols or lingering foreign characters
    # Only allow letters, numbers, hyphens, and spaces.
    if not re.match(r'^[\w\s\-]+$', cleaned_term):
        return

    # 6. Final Deduplication (Lowercase for uniqueness check, but keep original case for display)
    # Whisper prefers exact casing if possible, but for the set we just want uniqueness.
    if cleaned_term and len(cleaned_term) > 1:
        collection.add(cleaned_term)