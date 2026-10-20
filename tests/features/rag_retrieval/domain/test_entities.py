# Path: tests/features/rag_retrieval/domain/test_entities.py

import uuid

import pytest
from pydantic import ValidationError

from app.features.rag_retrieval.domain.entities import RetrievedChunk, SearchQuery


class TestSearchQuery:
    @pytest.mark.parametrize("blank_prompt", ["", "   ", "\n\t"])
    def test_blank_prompt_is_rejected(self, blank_prompt):
        with pytest.raises(ValidationError):
            SearchQuery(prompt=blank_prompt)

    def test_prompt_is_stripped(self):
        assert SearchQuery(prompt="  what did the witness say?  ").prompt == "what did the witness say?"


class TestRetrievedChunkContentPayload:
    @staticmethod
    def _chunk(payload):
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=payload, start_sec=0,
                              end_sec=1, relevance_score=1.0)

    @pytest.mark.parametrize("payload, content", [
        ("[Source: Lecture 4 — Bayes.] [Audio Context: x] text", "[Audio Context: x] text"),
        ("[Audio Context: x] text", "[Audio Context: x] text"),
        ("text quoting [Source: y] later", "text quoting [Source: y] later"),
    ])
    def test_only_a_leading_source_line_is_removed(self, payload, content):
        assert self._chunk(payload).content_payload == content
