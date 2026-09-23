# Canvas: test_glossary_miner_term_cleaner_v2
# Path: tests/features/glossary_miner/utils/test_term_cleaner.py

from app.tools.glossary_miner.domain.entities import WikidataItem
from app.tools.glossary_miner.utils.term_cleaner import clean_and_extract_terms


class TestTermCleaner:

    def test_clean_standard_terms(self):
        item = WikidataItem(
            item_id="Q1",
            label="Heart Attack",
            aliases_by_lang={
                "en": ["Myocardial Infarction", "MI"],
                "la": ["Infarctus"]
            }
        )
        results = clean_and_extract_terms([item])

        assert len(results) == 4
        assert "Heart Attack" in results
        assert "Myocardial Infarction" in results

    def test_drop_long_phrases(self):
        item = WikidataItem(
            item_id="Q2",
            label="Valid",
            aliases_by_lang={"en": ["This is way too long"]}
        )
        results = clean_and_extract_terms([item])
        assert results == ["Valid"]

    def test_strip_brackets_and_punctuation(self):
        item = WikidataItem(
            item_id="Q3",
            label="Infarction (disorder)",
            aliases_by_lang={"en": ["Earnings, Before Interest:"]}
        )
        results = clean_and_extract_terms([item])

        assert "Infarction" in results
        assert "Earnings Before Interest" in results

    def test_deduplication_and_sorting(self):
        item1 = WikidataItem(item_id="Q1", label="Zebra", aliases_by_lang={"en": ["Apple"]})
        item2 = WikidataItem(item_id="Q2", label="Apple", aliases_by_lang={"la": ["Zebra"]})

        results = clean_and_extract_terms([item1, item2])
        assert results == ["Apple", "Zebra"]

class TestTermCleanerInSerbian:
    def test_cyrillic_labels_become_latin_like_the_transcripts(self, appliance_language):
        appliance_language("sr")
        item = WikidataItem(item_id="Q1", label="Инфаркт миокарда", aliases_by_lang={"sr-el": ["srčani udar"]})

        assert clean_and_extract_terms([item]) == ["Infarkt miokarda", "srčani udar"]
