# Path: integration_tests/features/rag_retrieval/test_index_language_integration.py

from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from app.core.config import settings
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_retrieval.exceptions import IndexLanguageMismatchError
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI


@pytest.mark.integration
class TestIndexLanguageIntegration:
    """Real Postgres (db_test): the text search configurations the language packs rely on, and the index guard."""

    def test_the_test_index_reports_the_configuration_it_was_built_with(self, db_manager):
        assert PostgresSearchRepository().get_index_fts_config() == settings.FTS_LANGUAGE_CONFIG == "english"

    def test_guard_passes_for_the_language_the_index_was_built_for(self, db_manager):
        RetrievalAPI(MagicMock(), MagicMock(), PostgresSearchRepository()).verify_index_language()

    def test_guard_refuses_an_index_built_for_another_language(self, db_manager, monkeypatch):
        monkeypatch.setattr(settings, "FTS_LANGUAGE_CONFIG", "serbian")
        monkeypatch.setattr(settings, "APPLIANCE_LANGUAGE", "sr")

        with pytest.raises(IndexLanguageMismatchError, match="'english'.*'serbian'"):
            RetrievalAPI(MagicMock(), MagicMock(), PostgresSearchRepository()).verify_index_language()

    @pytest.mark.parametrize("config", ["english", "serbian", "simple"])
    def test_every_packs_configuration_exists(self, db_manager, config):
        with db_manager.session() as session:
            assert session.execute(text("SELECT count(*) FROM pg_ts_config WHERE cfgname = :c"), {"c": config}).scalar() == 1

    def test_serbian_stemmer_meets_cyrillic_latin_and_both_variants(self, db_manager):
        """Ijekavian Latin, ekavian Cyrillic and an inflected form all reach one stem."""
        with db_manager.session() as session:
            matches = session.execute(text(
                "SELECT to_tsvector('serbian', :doc) @@ plainto_tsquery('serbian', :query)"
            ), {"doc": "Ljekari su pregledali pacijente.", "query": "лекар"}).scalar()

        assert matches is True

    def test_slovenian_simple_configuration_matches_exact_forms_only(self, db_manager):
        with db_manager.session() as session:
            exact, inflected = session.execute(text(
                "SELECT to_tsvector('simple', :doc) @@ plainto_tsquery('simple', 'zdravniki'), "
                "       to_tsvector('simple', :doc) @@ plainto_tsquery('simple', 'zdravnik')"
            ), {"doc": "Zdravniki so pregledali bolnike."}).one()

        assert exact is True
        assert inflected is False     # no Slovenian stemmer in Postgres: the dense leg has to carry inflection
