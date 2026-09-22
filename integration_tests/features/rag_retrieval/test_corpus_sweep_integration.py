# Path: integration_tests/features/rag_retrieval/test_corpus_sweep_integration.py

import uuid

import pytest
from sqlalchemy import text as sa_text

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_storage.data.models import VectorChunkModel
from app.features.storage.data.models import FileModel, SourceModel

pytestmark = pytest.mark.integration

CHUNKS_PER_SOURCE = 5


def _add_file(session, file_name: str) -> uuid.UUID:
    """`sources.file_id` is NOT NULL, so every seeded source needs a real file row behind it."""
    file_id = uuid.uuid4()
    session.add(FileModel(
        id=file_id,
        file_path=f"/vault/{file_name}",
        file_name=file_name,
        file_size_bytes=1024,
        mime_type="audio/wav",
    ))
    session.flush()
    return file_id


def _vector(seed: float) -> list:
    """A 1024-dim vector whose first component carries the seed, so distance ordering is predictable."""
    vector = [0.01] * 1024
    vector[0] = seed
    return vector


class TestCorpusSweepIntegration:
    """
    Real Postgres + pgvector. Covers the two reads that let report mode reason about corpus coverage:
    what is actually indexed, and which sources a per-document sweep should look inside.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_schema(self):
        Base.metadata.create_all(bind=db_manager._engine_wrapper._engine)  # noqa
        yield

    @pytest.fixture
    def repository(self, db_manager):
        return PostgresSearchRepository()

    @pytest.fixture
    def seeded_corpus(self, db_manager):
        """
        Three indexed sources, plus one ingested source with NO chunks -- the case that distinguishes
        "ingested" from "indexed".
        """
        sources = []
        with db_manager.session() as session:
            session.execute(
                sa_text("ALTER TABLE vector_chunks DROP CONSTRAINT IF EXISTS vector_chunks_source_id_fkey;")
            )

            file_ids = []
            for index in range(3):
                source_id = uuid.uuid4()
                file_id = _add_file(session, f"recording_{index}.wav")
                file_ids.append(file_id)
                session.add(SourceModel(
                    id=source_id,
                    name=f"Recording {index}",
                    source_type="audio",
                    content_hash=str(uuid.uuid4()),
                    file_id=file_id,
                ))
                for chunk_index in range(CHUNKS_PER_SOURCE):
                    session.add(VectorChunkModel(
                        source_id=source_id,
                        chunk_id=uuid.uuid4(),
                        start_sec=float(chunk_index * 30),
                        end_sec=float(chunk_index * 30 + 30),
                        text_payload=f"context\n\n[SPEAKER_0 @ 00:00:00]: evidence handling {index}",
                        raw_text=f"[SPEAKER_0 @ 00:00:00]: evidence handling {index}",
                        context_text="context",
                        speaker_labels=["SPEAKER_0"],
                        # Source 0 is nearest the probe vector, source 2 furthest.
                        embedding=_vector(0.9 - index * 0.3 - chunk_index * 0.01),
                    ))
                sources.append(source_id)

            unindexed_id = uuid.uuid4()
            unindexed_file_id = _add_file(session, "never_indexed.wav")
            file_ids.append(unindexed_file_id)
            session.add(SourceModel(
                id=unindexed_id, name="Ingested but never indexed",
                source_type="audio", content_hash=str(uuid.uuid4()),
                file_id=unindexed_file_id,
            ))

        yield {"indexed": sources, "unindexed": unindexed_id}

        with db_manager.session() as session:
            for source_id in sources:
                session.execute(
                    sa_text("DELETE FROM vector_chunks WHERE source_id = :sid"), {"sid": str(source_id)}
                )
            session.execute(
                sa_text("DELETE FROM sources WHERE id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": [str(s) for s in sources] + [str(unindexed_id)]},
            )
            session.execute(
                sa_text("DELETE FROM files WHERE id = ANY(CAST(:ids AS uuid[]))"),
                {"ids": [str(f) for f in file_ids]},
            )

    # ==========================================
    # list_indexed_sources
    # ==========================================

    def test_lists_only_sources_that_actually_have_chunks(self, repository, seeded_corpus):
        """
        Storage's list_sources() answers "what was ingested". A coverage claim needs "what can be
        retrieved" -- a file can be ingested and still hold nothing indexable.
        """
        listed = repository.list_indexed_sources()

        listed_ids = {s.source_id for s in listed}
        assert set(seeded_corpus["indexed"]).issubset(listed_ids)
        assert seeded_corpus["unindexed"] not in listed_ids

    def test_carries_the_human_readable_name_and_chunk_counts(self, repository, seeded_corpus):
        by_id = {s.source_id: s for s in repository.list_indexed_sources()}

        first = by_id[seeded_corpus["indexed"][0]]
        assert first.name == "Recording 0"
        assert first.chunk_count == CHUNKS_PER_SOURCE
        assert first.first_start_sec == 0.0
        assert first.last_end_sec == (CHUNKS_PER_SOURCE - 1) * 30 + 30

    def test_min_chunks_filters_thin_sources_out(self, repository, seeded_corpus):
        assert repository.list_indexed_sources(min_chunks=CHUNKS_PER_SOURCE + 1) == []

    # ==========================================
    # pick_sweep_sources
    # ==========================================

    def test_picks_sources_ordered_by_their_best_matching_chunk(self, repository, seeded_corpus):
        picked = repository.pick_sweep_sources(_vector(1.0), ann_depth=2000, max_sources=50)

        indexed = seeded_corpus["indexed"]
        positions = {source_id: picked.index(source_id) for source_id in indexed if source_id in picked}
        assert set(indexed).issubset(set(picked))
        assert positions[indexed[0]] < positions[indexed[1]] < positions[indexed[2]]

    def test_each_source_appears_once_however_many_chunks_it_has(self, repository, seeded_corpus):
        picked = repository.pick_sweep_sources(_vector(1.0), ann_depth=2000, max_sources=50)

        assert len(picked) == len(set(picked))

    def test_max_sources_bounds_the_sweep(self, repository, seeded_corpus):
        picked = repository.pick_sweep_sources(_vector(1.0), ann_depth=2000, max_sources=2)

        assert len(picked) == 2

    def test_a_shallow_ann_probe_still_returns_the_nearest_source(self, repository, seeded_corpus):
        """
        The probe is the cost gate: its depth, not the corpus size, decides how much work this does.
        Even at depth 1 it has to return the single best-matching source.
        """
        picked = repository.pick_sweep_sources(_vector(1.0), ann_depth=1, max_sources=50)

        assert picked == [seeded_corpus["indexed"][0]]

    # ==========================================
    # the sweep's per-source search
    # ==========================================

    def test_a_picked_source_can_be_searched_with_the_existing_hybrid_sql(self, repository, seeded_corpus):
        """
        The sweep reuses execute_hybrid_search with a source filter rather than returning the ANN slice
        directly, so a swept source still gets the keyword half of retrieval.
        """
        source_id = seeded_corpus["indexed"][0]

        chunks = repository.execute_hybrid_search(
            query_vector=_vector(1.0),
            query_text="evidence handling",
            limit=3,
            source_id_filter=source_id,
        )

        assert 0 < len(chunks) <= 3
        assert {c.source_id for c in chunks} == {source_id}
