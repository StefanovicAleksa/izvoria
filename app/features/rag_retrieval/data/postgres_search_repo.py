# Canvas: rag_retrieval_postgres_repo_v6
# Path: app/features/rag_retrieval/data/postgres_search_repo.py

import logging
import uuid
from typing import List, Optional
from sqlalchemy import text

from app.core.config import settings
from app.core.database.service.manager import db_manager
from app.features.rag_retrieval.domain.interfaces import IRetrievalRepository
from app.features.rag_retrieval.domain.entities import RetrievedChunk
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


class PostgresSearchRepository(IRetrievalRepository):

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
