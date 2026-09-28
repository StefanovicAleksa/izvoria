# Path: integration_tests/features/rag_retrieval/test_document_anchors_integration.py

"""
Page-anchored document chunks in the real `vector_chunks` table: written by the RAG repository, read back with
their pages, neighbours found by reading position (every document chunk has start_sec 0.0), and the startup
upgrade that adds these columns to a table from an earlier build.
"""

import uuid

import pytest
from sqlalchemy import text as sa_text

from app.core.anchors.domain.enums import AnchorKind
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.core.database.data.migrations.pre_alembic import VECTOR_CHUNKS_CATCH_UP
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.storage.data.models import FileModel, SourceModel

pytestmark = pytest.mark.integration


def _source(session, name: str, source_type: str) -> uuid.UUID:
    file_id, source_id = uuid.uuid4(), uuid.uuid4()
    session.add(FileModel(id=file_id, file_path=f"/vault/{name}", file_name=name, file_size_bytes=1,
                          mime_type="application/pdf"))
    session.flush()
    session.add(SourceModel(id=source_id, name=name, source_type=source_type, content_hash=str(uuid.uuid4()),
                            file_id=file_id))
    session.flush()
    return source_id


def _page_chunk(page: int, text: str) -> UnifiedChunk:
    return UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0.0, end_sec=0.0, anchor_kind=AnchorKind.PAGE,
                        page_start=page, page_end=page, section_path=f"Article {page}",
                        text_payload=f"[Document Context: lease]\n\n{text}", raw_text=text)


class TestDocumentAnchorsIntegration:
    @pytest.fixture
    def lease(self, db_manager):
        with db_manager.session() as session:
            source_id = _source(session, "lease.pdf", "document")
        chunks = [_page_chunk(page, f"Clause text on page {page}.") for page in (1, 2, 3, 4)]
        PostgresRagRepository().replace_rag_chunks(source_id, chunks, [[0.01] * 1024 for _ in chunks])
        return source_id, chunks

    def test_page_anchor_and_position_round_trip(self, lease):
        source_id, chunks = lease

        fetched = PostgresSearchRepository().fetch_chunks([c.chunk_id for c in chunks])

        assert [c.position for c in fetched] == [0, 1, 2, 3]
        assert all(c.anchor_kind is AnchorKind.PAGE for c in fetched)
        assert [(c.page_start, c.page_end, c.section_path) for c in fetched] == [
            (p, p, f"Article {p}") for p in (1, 2, 3, 4)]

    def test_neighbours_follow_page_order_although_every_start_sec_is_zero(self, lease):
        _, chunks = lease

        neighbours = PostgresSearchRepository().fetch_neighbors([chunks[2].chunk_id])

        assert sorted(n.page_start for n in neighbours) == [2, 4]

    def test_indexed_source_reports_its_kind_and_page_count(self, lease):
        source_id, _ = lease

        [source] = [s for s in PostgresSearchRepository().list_indexed_sources() if s.source_id == source_id]

        assert (source.anchor_kind, source.page_count, source.chunk_count) == (AnchorKind.PAGE, 4, 4)

    def test_upgrade_backfills_transcript_positions_in_time_order_and_is_idempotent(self, db_manager):
        with db_manager.session() as session:
            source_id = _source(session, "interview.wav", "audio")
            # Rows as an earlier build wrote them: no position
            for start in (60.0, 0.0, 30.0):
                session.execute(sa_text(
                    "INSERT INTO vector_chunks (id, source_id, chunk_id, start_sec, end_sec, text_payload, raw_text, "
                    "speaker_labels, embedding) VALUES (:id, :sid, :cid, :start, :end, 't', 't', '[]', :emb)"),
                    {"id": str(uuid.uuid4()), "sid": str(source_id), "cid": str(uuid.uuid4()), "start": start,
                     "end": start + 30, "emb": str([0.01] * 1024)})

            for _ in range(2):
                for statement in VECTOR_CHUNKS_CATCH_UP:
                    session.execute(sa_text(statement))

            rows = session.execute(sa_text(
                "SELECT start_sec, position, anchor_kind FROM vector_chunks WHERE source_id = :sid ORDER BY start_sec"),
                {"sid": str(source_id)}).all()

        assert [(r.start_sec, r.position, r.anchor_kind) for r in rows] == [
            (0.0, 0, "time"), (30.0, 1, "time"), (60.0, 2, "time")]
