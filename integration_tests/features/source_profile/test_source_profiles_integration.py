# Path: integration_tests/features/source_profile/test_source_profiles_integration.py

"""
Source profiles in the real database: the `source_profiles` table round-trips and cascades with its source, chunks
read back carry their source's title, and a source is re-indexed in place under a new `[Source: …]` line with the
same chunk ids and its own stemmer.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text as sa_text

from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.service.storage_api import RAGStorageService
from app.features.source_profile.data.llm_profile_adapter import LlmProfileAdapter
from app.features.source_profile.data.postgres_profile_repo import PostgresSourceProfileRepository
from app.features.source_profile.domain.entities import SourceProfile
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.service.source_profile_api import SourceProfileAPI
from app.features.storage.data.models import FileModel, SourceModel

pytestmark = pytest.mark.integration


def _source(session, name: str) -> uuid.UUID:
    file_id, source_id = uuid.uuid4(), uuid.uuid4()
    session.add(FileModel(id=file_id, file_path=f"/vault/{name}", file_name=name, file_size_bytes=1,
                          mime_type="video/mp4"))
    session.flush()
    session.add(SourceModel(id=source_id, name=name, source_type="video", content_hash=str(uuid.uuid4()),
                            file_id=file_id, language="en"))
    session.flush()
    return source_id


def _chunk(start: float, text: str) -> UnifiedChunk:
    return UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=start, end_sec=start + 30,
                        text_payload=f"[Audio Context: a lecture]\n\n{text}", raw_text=text,
                        context_text="[Audio Context: a lecture]")


class TestSourceProfilesIntegration:
    @pytest.fixture
    def lecture(self, db_manager):
        with db_manager.session() as session:
            source_id = _source(session, "module_01.mp4")
        chunks = [_chunk(0, "Today we cover conditional probability."), _chunk(30, "Bayes rule follows.")]
        PostgresRagRepository().replace_rag_chunks(source_id, chunks, [[0.01] * 1024 for _ in chunks])
        return source_id, chunks

    @pytest.fixture
    def profile_api(self, db_manager):
        return SourceProfileAPI(PostgresSourceProfileRepository(db_manager), LlmProfileAdapter())

    def test_a_profile_round_trips_and_is_listed_with_its_source(self, lecture, profile_api):
        source_id, _ = lecture
        assert source_id in {s.source_id for s in profile_api.list_unprofiled()}

        profile_api.edit(source_id, "Statistics 110, Lecture 4", "Conditional probability.")

        assert profile_api.get(source_id) == SourceProfile(
            source_id=source_id, title="Statistics 110, Lecture 4", topic="Conditional probability.",
            origin=ProfileOrigin.EDITED, language="en")
        [listed] = [s for s in profile_api.list_indexed() if s.source_id == source_id]
        assert (listed.identity.file_name, listed.display_title) == ("module_01.mp4", "Statistics 110, Lecture 4")
        assert source_id not in {s.source_id for s in profile_api.list_unprofiled()}

    def test_saving_twice_replaces_the_profile(self, lecture, profile_api):
        source_id, _ = lecture
        profile_api.edit(source_id, "First")
        profile_api.edit(source_id, "Second")

        assert profile_api.get(source_id).title == "Second"

    def test_deleting_the_source_deletes_its_profile(self, db_manager, lecture, profile_api):
        source_id, _ = lecture
        profile_api.edit(source_id, "Lecture 4")

        with db_manager.session() as session:
            session.execute(sa_text("DELETE FROM sources WHERE id = :id"), {"id": source_id})

        assert profile_api.get(source_id) is None

    def test_chunks_read_back_carry_the_title_or_the_name(self, lecture, profile_api):
        source_id, chunks = lecture
        repo = PostgresSearchRepository()

        assert {c.source_title for c in repo.fetch_chunks([c.chunk_id for c in chunks])} == {"module_01.mp4"}

        profile_api.edit(source_id, "Statistics 110, Lecture 4")
        whole = repo.fetch_source_chunks(source_id)
        assert [c.chunk_id for c in whole] == [c.chunk_id for c in chunks]
        assert {c.source_title for c in whole} == {"Statistics 110, Lecture 4"}
        assert {c.source_title for c in repo.fetch_neighbors([chunks[0].chunk_id])} == {"Statistics 110, Lecture 4"}
        [indexed] = [s for s in repo.list_indexed_sources() if s.source_id == source_id]
        assert indexed.name == "Statistics 110, Lecture 4"

    def test_refresh_re_indexes_in_place_and_the_title_becomes_searchable(self, db_manager, lecture, profile_api):
        source_id, chunks = lecture
        profile_api.edit(source_id, "Statistics 110 Lecture Four", "Conditional probability.")
        embedder, orchestrator = MagicMock(), MagicMock()
        embedder.embed_batch.side_effect = lambda model, texts: [[0.02] * 1024 for _ in texts]
        service = RAGStorageService(PostgresRagRepository(), orchestrator, embedder, profile_api=profile_api)

        assert service.refresh_source_line(source_id) == 2

        with db_manager.session() as session:
            rows = session.execute(sa_text(
                "SELECT chunk_id, context_text, text_payload, raw_text, fts_config::text AS cfg, "
                "fts_document @@ plainto_tsquery('english', 'lecture four') AS found "
                "FROM vector_chunks WHERE source_id = :id ORDER BY position"), {"id": source_id}).mappings().all()
        assert [r["chunk_id"] for r in rows] == [c.chunk_id for c in chunks]
        line = "[Source: Statistics 110 Lecture Four — Conditional probability.]"
        assert rows[0]["context_text"] == f"{line}\n[Audio Context: a lecture]"
        assert rows[0]["text_payload"] == f"{line} [Audio Context: a lecture] Today we cover conditional probability."
        assert rows[0]["raw_text"] == "Today we cover conditional probability."
        assert all(r["found"] and r["cfg"] == "english" for r in rows)

        # A second refresh replaces the line rather than stacking another
        profile_api.edit(source_id, "Renamed")
        service.refresh_source_line(source_id)
        with db_manager.session() as session:
            context = session.execute(sa_text(
                "SELECT context_text FROM vector_chunks WHERE chunk_id = :id"), {"id": chunks[0].chunk_id}).scalar()
        assert context == "[Source: Renamed]\n[Audio Context: a lecture]"
