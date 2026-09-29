# Path: tests/features/rag_retrieval/utils/test_sparse_query.py

from unittest.mock import patch

import pytest

from app.core.config import Settings
from app.features.rag_retrieval.utils.sparse_query import sparse_query_texts


@pytest.fixture
def with_unstemmed_config():
    """An appliance whose index also holds chunks under `simple`, which neither stems nor drops stop words."""
    with patch("app.features.rag_retrieval.utils.sparse_query.settings",
               Settings(APPLIANCE_LANGUAGE="en", FTS_LANGUAGE_CONFIG="simple")):
        yield


class TestSparseQueryTexts:
    def test_one_text_per_configuration_in_the_index(self):
        configs, texts = sparse_query_texts("contract termination")

        assert configs == ["english", "serbian"]
        assert texts == ["contract termination"] * 2

    def test_unstemmed_configuration_gets_the_question_without_any_languages_stop_words(self, with_unstemmed_config):
        # "the"/"of" are English, "je"/"u" Serbian: each would match nearly every chunk `simple` indexed
        configs, texts = sparse_query_texts("What is the date of the pogodba in je u ugovoru")

        assert configs == ["english", "serbian", "simple"]
        assert texts[configs.index("simple")] == "date pogodba ugovoru"

    def test_stemmed_configurations_keep_the_question_as_asked(self, with_unstemmed_config):
        configs, texts = sparse_query_texts("What is the date of the pogodba")

        assert texts[configs.index("english")] == texts[configs.index("serbian")] == "What is the date of the pogodba"

    def test_a_question_of_only_stop_words_leaves_nothing_for_simple(self, with_unstemmed_config):
        configs, texts = sparse_query_texts("what is the")

        assert texts[configs.index("simple")] == ""

    def test_follows_an_explicit_fts_config(self):
        with patch("app.features.rag_retrieval.utils.sparse_query.settings",
                   Settings(APPLIANCE_LANGUAGE="en", FTS_LANGUAGE_CONFIG="german")):
            configs, texts = sparse_query_texts("Vertrag")

        assert configs == ["english", "serbian", "german"]
        assert texts == ["Vertrag"] * 3
