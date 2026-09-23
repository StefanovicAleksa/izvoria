# Canvas: rag_retrieval_postgres_repo_v6
# Path: app/features/rag_retrieval/data/postgres_search_repo.py

import logging
import re
import uuid
from typing import List, Optional
from sqlalchemy import text

from app.core.config import settings
from app.core.database.service.manager import db_manager
from app.features.rag_retrieval.domain.interfaces import IRetrievalRepository
from app.features.rag_retrieval.domain.entities import IndexedSource, RetrievedChunk
from app.features.rag_retrieval.utils.error_mapper import map_retrieval_error

logger = logging.getLogger(__name__)

# Sparse leg uses OR semantics: plainto_tsquery() ANDs every lexeme of a natural-language question,
# so "what did the doctor say about the MRI results" only matched chunks containing *all* of
# doctor & say & mri & result, and the keyword half of hybrid search almost never contributed.
# Rewriting its ' & ' operators to ' | ' keeps Postgres' own stemming/stop-word handling while
# letting any matching term count; ts_rank(..., 1) then rewards chunks matching more terms and
# normalizes for document length. A question of only stop words yields no sparse query at all.
_HYBRID_SEARCH_SQL = text("""
    WITH query_terms AS (
        SELECT replace(plainto_tsquery(CAST(:fts_config AS regconfig), :query_text)::text, ' & ', ' | ') AS q_text
    ),
    sparse_query AS (
        SELECT CASE WHEN q_text = '' THEN NULL ELSE q_text::tsquery END AS tsq
        FROM query_terms
    ),
    dense_search AS (
        SELECT v.chunk_id,
               RANK() OVER (ORDER BY v.embedding <=> CAST(:query_vector AS vector)) AS rank_idx
        FROM vector_chunks v
        WHERE (:source_id IS NULL OR v.source_id = :source_id)
        ORDER BY v.embedding <=> CAST(:query_vector AS vector)
        LIMIT :fetch_depth
    ),
    sparse_search AS (
        SELECT v.chunk_id,
               RANK() OVER (ORDER BY ts_rank(v.fts_document, sq.tsq, 1) DESC) AS rank_idx
        FROM vector_chunks v
        CROSS JOIN sparse_query sq
        WHERE sq.tsq IS NOT NULL
          AND v.fts_document @@ sq.tsq
          AND (:source_id IS NULL OR v.source_id = :source_id)
        ORDER BY ts_rank(v.fts_document, sq.tsq, 1) DESC
        LIMIT :fetch_depth
    ),
    fused AS (
        SELECT COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
               COALESCE(1.0 / (:rrf_k + d.rank_idx), 0.0) +
               COALESCE(1.0 / (:rrf_k + s.rank_idx), 0.0) AS rrf_score
        FROM dense_search d
        FULL OUTER JOIN sparse_search s ON d.chunk_id = s.chunk_id
    )
    SELECT v.chunk_id,
           v.source_id,
           v.text_payload,
           v.raw_text,
           v.speaker_labels,
           v.start_sec,
           v.end_sec,
           f.rrf_score,
           vc.extracted_file_id
    FROM fused f
    JOIN vector_chunks v ON v.chunk_id = f.chunk_id
    LEFT JOIN video_clips vc ON vc.chunk_id = f.chunk_id
    ORDER BY f.rrf_score DESC
    LIMIT :limit
""")


# For each anchor chunk, the chunk that starts just before it and the one that starts just after it in the same
# source: the transcript on either side of a hit, for answers that cross a chunk boundary.
_NEIGHBORS_SQL = text("""
    WITH anchors AS (
        SELECT chunk_id, source_id, start_sec
        FROM vector_chunks
        WHERE chunk_id = ANY(CAST(:chunk_ids AS uuid[]))
    )
    SELECT DISTINCT ON (n.chunk_id)
           n.chunk_id, n.source_id, n.text_payload, n.raw_text, n.speaker_labels, n.start_sec, n.end_sec,
           vc.extracted_file_id
    FROM anchors a
    CROSS JOIN LATERAL (
        (SELECT v.* FROM vector_chunks v
         WHERE v.source_id = a.source_id AND v.start_sec < a.start_sec
         ORDER BY v.start_sec DESC LIMIT 1)
        UNION ALL
        (SELECT v.* FROM vector_chunks v
         WHERE v.source_id = a.source_id AND v.start_sec > a.start_sec
         ORDER BY v.start_sec ASC LIMIT 1)
    ) n
    LEFT JOIN video_clips vc ON vc.chunk_id = n.chunk_id
    ORDER BY n.chunk_id
""")

_CHUNKS_BY_ID_SQL = text("""
    SELECT DISTINCT ON (v.chunk_id)
           v.chunk_id, v.source_id, v.text_payload, v.raw_text, v.speaker_labels, v.start_sec, v.end_sec,
           vc.extracted_file_id
    FROM vector_chunks v
    LEFT JOIN video_clips vc ON vc.chunk_id = v.chunk_id
    WHERE v.chunk_id = ANY(CAST(:chunk_ids AS uuid[]))
    ORDER BY v.chunk_id
""")


# Every source that actually has indexed chunks, with the human-readable name RetrievedChunk has no room
# for. `sources`/`files` are joined here rather than in the hot search path because only corpus-level
# callers need the name, and the search SQL runs per query.
_INDEXED_SOURCES_SQL = text("""
    SELECT v.source_id,
           s.name AS source_name,
           f.file_name,
           COUNT(*) AS chunk_count,
           MIN(v.start_sec) AS first_start_sec,
           MAX(v.end_sec) AS last_end_sec
    FROM vector_chunks v
    JOIN sources s ON s.id = v.source_id
    LEFT JOIN files f ON f.id = s.file_id
    GROUP BY v.source_id, s.name, f.file_name
    HAVING COUNT(*) >= :min_chunks
    ORDER BY chunk_count DESC
""")


# Which sources the per-document sweep should look inside.
#
# The naive form -- GROUP BY source_id over all of vector_chunks taking MIN(distance) -- computes a
# distance for every row in the table and cannot be served by any index, so it turns into a full scan that
# grows with the corpus. Instead one ANN probe (index-served, LIMIT :ann_depth) is grouped down to the
# sources it touched, ranked by how early each source's best chunk appeared. Cost is bounded by ann_depth,
# not by corpus size.
_SWEEP_SOURCES_SQL = text("""
    WITH ann AS (
        SELECT v.source_id,
               ROW_NUMBER() OVER (ORDER BY v.embedding <=> CAST(:query_vector AS vector)) AS rank_idx
        FROM vector_chunks v
        ORDER BY v.embedding <=> CAST(:query_vector AS vector)
        LIMIT :ann_depth
    )
    SELECT source_id
    FROM ann
    GROUP BY source_id
    ORDER BY MIN(rank_idx) ASC
    LIMIT :max_sources
""")


# The generation expression of vector_chunks.fts_document, e.g. "to_tsvector('english'::regconfig, text_payload)"
_FTS_EXPRESSION_SQL = text("""
    SELECT pg_get_expr(d.adbin, d.adrelid) AS expression
    FROM pg_attrdef d
    JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
    WHERE d.adrelid = to_regclass('vector_chunks') AND a.attname = 'fts_document'
""")
_REGCONFIG = re.compile(r"'([a-z_]+)'::regconfig")


class PostgresSearchRepository(IRetrievalRepository):

    def get_index_fts_config(self) -> Optional[str]:
        try:
            with db_manager.session() as session:
                expression = session.execute(_FTS_EXPRESSION_SQL).scalar()
            if expression is None:
                return None
            match = _REGCONFIG.search(expression)
            return match.group(1) if match else None
        except Exception as e:
            raise map_retrieval_error(e, context="Reading the index's text search configuration") from e

    def list_indexed_sources(self, min_chunks: int = 1) -> List[IndexedSource]:
        try:
            with db_manager.session() as session:
                rows = session.execute(_INDEXED_SOURCES_SQL, {"min_chunks": min_chunks}).mappings().all()

                return [
                    IndexedSource(
                        source_id=row["source_id"],
                        # `sources.name` is the display name; the filename is the fallback, and the id is
                        # the last resort -- a report's references table must never show a blank source.
                        name=row["source_name"] or row["file_name"] or str(row["source_id"]),
                        chunk_count=int(row["chunk_count"]),
                        first_start_sec=float(row["first_start_sec"] or 0.0),
                        last_end_sec=float(row["last_end_sec"] or 0.0),
                    )
                    for row in rows
                ]

        except Exception as e:
            raise map_retrieval_error(e, context="Listing indexed sources") from e

    def pick_sweep_sources(
            self,
            query_vector: List[float],
            ann_depth: int,
            max_sources: int
    ) -> List[uuid.UUID]:
        vector_str = f"[{','.join(map(str, query_vector))}]"

        try:
            with db_manager.session() as session:
                rows = session.execute(
                    _SWEEP_SOURCES_SQL,
                    {"query_vector": vector_str, "ann_depth": ann_depth, "max_sources": max_sources}
                ).mappings().all()

                return [row["source_id"] for row in rows]

        except Exception as e:
            raise map_retrieval_error(e, context="Picking sources for the corpus sweep") from e

    def fetch_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        if not chunk_ids:
            return []

        try:
            with db_manager.session() as session:
                rows = session.execute(
                    _CHUNKS_BY_ID_SQL, {"chunk_ids": [str(chunk_id) for chunk_id in chunk_ids]}
                ).mappings().all()

                chunks = [
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        source_id=row["source_id"],
                        text_payload=row["text_payload"],
                        raw_text=row["raw_text"],
                        speaker_labels=list(row["speaker_labels"] or []),
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                        # Fetched by id for an earlier conversation turn, not ranked for this query
                        relevance_score=0.0,
                        extracted_file_id=row["extracted_file_id"]
                    )
                    for row in rows
                ]
                by_id = {chunk.chunk_id: chunk for chunk in chunks}
                return [by_id[chunk_id] for chunk_id in dict.fromkeys(chunk_ids) if chunk_id in by_id]

        except Exception as e:
            raise map_retrieval_error(e, context="Fetching chunks by id") from e

    def fetch_neighbors(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        if not chunk_ids:
            return []

        try:
            with db_manager.session() as session:
                rows = session.execute(
                    _NEIGHBORS_SQL, {"chunk_ids": [str(chunk_id) for chunk_id in chunk_ids]}
                ).mappings().all()

                return [
                    RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        source_id=row["source_id"],
                        text_payload=row["text_payload"],
                        raw_text=row["raw_text"],
                        speaker_labels=list(row["speaker_labels"] or []),
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                        # Neighbours were not ranked; they are context for the chunk that was
                        relevance_score=0.0,
                        extracted_file_id=row["extracted_file_id"]
                    )
                    for row in rows
                ]

        except Exception as e:
            raise map_retrieval_error(e, context="Fetching neighbouring chunks") from e

    def execute_hybrid_search(
            self,
            query_vector: List[float],
            query_text: str,
            limit: int,
            source_id_filter: Optional[uuid.UUID] = None
    ) -> List[RetrievedChunk]:

        vector_str = f"[{','.join(map(str, query_vector))}]"
        fetch_depth = max(limit * 5, 60)

        try:
            with db_manager.session() as session:
                result = session.execute(
                    _HYBRID_SEARCH_SQL,
                    {
                        "query_vector": vector_str,
                        "query_text": query_text,
                        "fts_config": settings.FTS_LANGUAGE_CONFIG,
                        "source_id": source_id_filter,
                        "limit": limit,
                        "fetch_depth": fetch_depth,
                        "rrf_k": settings.RAG_RRF_K_CONSTANT
                    }
                )

                rows = result.mappings().all()

                retrieved_chunks = []
                for row in rows:
                    chunk = RetrievedChunk(
                        chunk_id=row["chunk_id"],
                        source_id=row["source_id"],
                        text_payload=row["text_payload"],
                        raw_text=row["raw_text"],
                        speaker_labels=list(row["speaker_labels"] or []),
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                        relevance_score=float(row["rrf_score"]),
                        extracted_file_id=row["extracted_file_id"]
                    )
                    retrieved_chunks.append(chunk)

                return retrieved_chunks

        except Exception as e:
            raise map_retrieval_error(e, context="Executing Hybrid RRF Postgres Query") from e
