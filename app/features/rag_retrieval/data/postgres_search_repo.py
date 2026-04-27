# Canvas: rag_retrieval_postgres_repo_v5
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

        # FIX: Replaced :query_vector::vector with CAST(:query_vector AS vector)
        # so SQLAlchemy correctly parses the bind parameter.
        sql_query = text("""
                         WITH dense_search AS (SELECT chunk_id,
                                                      source_id,
                                                      text_payload,
                                                      start_sec,
                                                      end_sec,
                                                      RANK() OVER (ORDER BY embedding <=> CAST(:query_vector AS vector)) as rank_idx
                                               FROM vector_chunks
                                               WHERE (:source_id IS NULL OR source_id = :source_id)
                                               ORDER BY embedding
                             <=> CAST (:query_vector AS vector)
                             LIMIT :fetch_depth
                             )
                            , sparse_search AS (
                         SELECT chunk_id, source_id, text_payload, start_sec, end_sec, RANK() OVER (ORDER BY ts_rank_cd(fts_document, plainto_tsquery('english', :query_text)) DESC) as rank_idx
                         FROM vector_chunks
                         WHERE fts_document @@ plainto_tsquery('english'
                             , :query_text)
                           AND (:source_id IS NULL
                            OR source_id = :source_id)
                         ORDER BY ts_rank_cd(fts_document, plainto_tsquery('english', :query_text)) DESC
                             LIMIT :fetch_depth
                             )
                         SELECT COALESCE(d.chunk_id, s.chunk_id)             as chunk_id,
                                COALESCE(d.source_id, s.source_id)           as source_id,
                                COALESCE(d.text_payload, s.text_payload)     as text_payload,
                                COALESCE(d.start_sec, s.start_sec)           as start_sec,
                                COALESCE(d.end_sec, s.end_sec)               as end_sec,
                                (COALESCE(1.0 / (:rrf_k + d.rank_idx), 0.0) +
                                 COALESCE(1.0 / (:rrf_k + s.rank_idx), 0.0)) as rrf_score,
                                vc.extracted_file_id
                         FROM dense_search d
                                  FULL OUTER JOIN sparse_search s ON d.chunk_id = s.chunk_id
                                  LEFT JOIN video_clips vc ON COALESCE(d.chunk_id, s.chunk_id) = vc.chunk_id
                         ORDER BY rrf_score DESC LIMIT :limit;
                         """)

        try:
            with db_manager.session() as session:
                result = session.execute(
                    sql_query,
                    {
                        "query_vector": vector_str,
                        "query_text": query_text,
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
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                        relevance_score=float(row["rrf_score"]),
                        extracted_file_id=row["extracted_file_id"]
                    )
                    retrieved_chunks.append(chunk)

                return retrieved_chunks

        except Exception as e:
            raise map_retrieval_error(e, context="Executing Hybrid RRF Postgres Query") from e