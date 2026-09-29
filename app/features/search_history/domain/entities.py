# Path: app/features/search_history/domain/entities.py

import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.features.search_history.domain.enums import SearchHistoryMode


class SearchRecord(BaseModel):
    """
    One search as the user saw it: the query, and the chunks it returned in rank order with their scores.

    chunk_ids are not foreign keys. Re-ingesting a file cascades its chunks away, and the record of what was
    searched for (and what it pointed at) must outlive that; a reader simply finds fewer of them.
    """
    id: uuid.UUID
    query: str
    mode: SearchHistoryMode
    result_limit: int
    chunk_ids: List[uuid.UUID] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, frozen=True)

    @model_validator(mode="after")
    def _scores_match_chunks(self) -> "SearchRecord":
        if len(self.scores) != len(self.chunk_ids):
            raise ValueError("A search record needs exactly one score per chunk.")
        return self


class SearchRecordSummary(BaseModel):
    """Listing row: enough for a history sidebar without loading the hits."""
    id: uuid.UUID
    query: str
    mode: SearchHistoryMode
    hit_count: int
    created_at: datetime

    model_config = ConfigDict(frozen=True)
