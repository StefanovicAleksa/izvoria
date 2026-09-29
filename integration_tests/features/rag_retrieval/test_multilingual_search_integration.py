# Path: integration_tests/features/rag_retrieval/test_multilingual_search_integration.py

import os
import uuid

import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_retrieval.domain.entities import SearchQuery
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.storage.data.models import FileModel, SourceModel

pytestmark = pytest.mark.integration

# A search filtered to one source ranks its single chunk first on the dense leg: 1/(k+1). A chunk the keyword
# leg also matched scores that again on top, so the score says whether the keyword query found it.
DENSE_ONLY = 1.0 / (settings.RAG_RRF_K_CONSTANT + 1)

TEXTS = {
    "en": "The contracts were terminated in March.",
    "sr": "Ugovori su raskinuti u martu.",
}


def _index(db_manager, monkeypatch, language: str, text: str, embedding) -> uuid.UUID:
    """Ingests one chunk the way a batch in `language` does: through the rag_storage repository."""
    with db_manager.session() as session:
        file_id, source_id = uuid.uuid4(), uuid.uuid4()
        session.add(FileModel(id=file_id, file_path=f"/vault/{language}.wav", file_name=f"{language}.wav",
                              file_size_bytes=1))
        session.flush()
        session.add(SourceModel(id=source_id, name=f"{language}.wav", source_type="audio",
                                content_hash=str(uuid.uuid4()), language=language, file_id=file_id))
        session.flush()
    monkeypatch.setattr(settings, "FTS_LANGUAGE_CONFIG", {"en": "english", "sr": "serbian"}[language])
    chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0.0, end_sec=5.0, text_payload=text, raw_text=text)
    PostgresRagRepository().replace_rag_chunks(source_id, [chunk], [embedding])
    monkeypatch.setattr(settings, "FTS_LANGUAGE_CONFIG", "english")
    return source_id


class TestKeywordSearchAcrossLanguages:
    @pytest.fixture
    def sources(self, db_manager, monkeypatch):
        return {code: _index(db_manager, monkeypatch, code, text, [0.01] * 1024) for code, text in TEXTS.items()}

    def _keyword_hit(self, source_id: uuid.UUID, query: str) -> bool:
        [chunk] = PostgresSearchRepository().execute_hybrid_search([0.02] * 1024, query, limit=5,
                                                                   source_id_filter=source_id)
        return chunk.relevance_score > DENSE_ONLY + 1e-9

    def test_each_chunk_is_stamped_with_its_batch_languages_configuration(self, db_manager, sources):
        with db_manager.session() as session:
            stamped = dict(session.execute(sa_text(
                "SELECT s.language, v.fts_config::text FROM vector_chunks v JOIN sources s ON s.id = v.source_id "
                "WHERE v.source_id = ANY(CAST(:ids AS uuid[]))"), {"ids": [str(i) for i in sources.values()]}).all())

        assert stamped == {"en": "english", "sr": "serbian"}

    def test_an_inflected_serbian_word_finds_its_serbian_chunk(self, sources):
        assert self._keyword_hit(sources["sr"], "raskid ugovora")

    def test_an_english_plural_finds_its_english_chunk(self, sources):
        assert self._keyword_hit(sources["en"], "contract termination")

    def test_a_chunk_is_not_matched_by_another_languages_stems(self, sources):
        assert not self._keyword_hit(sources["sr"], "contract")
        assert not self._keyword_hit(sources["en"], "ugovora")


class TestDenseSearchAcrossLanguages:
    @pytest.fixture(autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH):
            pytest.skip(f"Needs the embedding model at {settings.RAG_EMBEDDING_MODEL_PATH}")

    def test_an_english_question_finds_the_serbian_chunk(self, db_manager, monkeypatch):
        adapter = HuggingFacePromptAdapter()
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        api = RetrievalAPI(orchestrator, adapter, PostgresSearchRepository())
        loaded = orchestrator.acquire_model(api._embedding_manifest(), adapter)
        try:
            serbian, unrelated = adapter.embed_prompts(loaded, [
                "Svedok je rekao da je ugovor o zakupu raskinut zbog neplaćene kirije.",
                "The weather in the mountains was cold and windy all week.",
            ])
        finally:
            orchestrator.release_model(loaded, adapter)
        serbian_id = _index(db_manager, monkeypatch, "sr", "Svedok je rekao da je ugovor o zakupu raskinut.", serbian)
        _index(db_manager, monkeypatch, "en", "The weather in the mountains was cold.", unrelated)

        result = api.search(SearchQuery(prompt="Why was the lease terminated?", limit=2))

        assert result.results[0].source_id == serbian_id
