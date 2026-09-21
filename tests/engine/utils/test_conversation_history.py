# Path: tests/engine/utils/test_conversation_history.py

import uuid
from unittest.mock import Mock

from app.engine.utils.conversation_history import build_conversation_history, history_chunks_newest_first
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import ConversationHistory, ConversationTurn, NumberedChunk
from app.features.session_memory.domain.entities import SessionTurn


def _chunk(chunk_id=None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id or uuid.uuid4(), source_id=uuid.uuid4(), text_payload="t",
                          start_sec=0.0, end_sec=1.0, relevance_score=0.0)


def _turn(question, evidence_ids, first_doc_number):
    return SessionTurn(id=uuid.uuid4(), session_id=uuid.uuid4(), question=question, answer_text=f"{question}!",
                       evidence_chunk_ids=evidence_ids, first_doc_number=first_doc_number,
                       created_at="2026-01-01T00:00:00Z")


class TestBuildConversationHistory:
    def test_empty_history_starts_numbering_at_one(self):
        get_chunks = Mock()

        history = build_conversation_history([], get_chunks)

        assert history == ConversationHistory(turns=[], next_doc_number=1)
        get_chunks.assert_not_called()

    def test_turns_without_evidence_skip_the_database(self):
        get_chunks = Mock()

        history = build_conversation_history([_turn("q", [], 4)], get_chunks)

        assert history.turns == [ConversationTurn(question="q", answer="q!", evidence=[])]
        assert history.next_doc_number == 4
        get_chunks.assert_not_called()

    def test_numbers_come_from_each_turn_and_deleted_chunks_keep_their_gap(self):
        a, deleted, b, c = _chunk(), uuid.uuid4(), _chunk(), _chunk()
        get_chunks = Mock(return_value=[c, a, b])
        turns = [_turn("q1", [a.chunk_id, deleted, b.chunk_id], 5), _turn("q2", [c.chunk_id], 8)]

        history = build_conversation_history(turns, get_chunks)

        get_chunks.assert_called_once_with([a.chunk_id, deleted, b.chunk_id, c.chunk_id])
        assert history.turns[0].evidence == [NumberedChunk(doc_number=5, chunk=a), NumberedChunk(doc_number=7, chunk=b)]
        assert history.turns[1].evidence == [NumberedChunk(doc_number=8, chunk=c)]
        assert history.next_doc_number == 9


class TestHistoryChunksNewestFirst:
    def test_most_recent_turn_first_each_chunk_once(self):
        a, b, c = _chunk(), _chunk(), _chunk()
        history = ConversationHistory(turns=[
            ConversationTurn(question="q1", answer="a1", evidence=[NumberedChunk(doc_number=1, chunk=a),
                                                                   NumberedChunk(doc_number=2, chunk=b)]),
            ConversationTurn(question="q2", answer="a2", evidence=[NumberedChunk(doc_number=3, chunk=c),
                                                                   NumberedChunk(doc_number=4, chunk=a)]),
        ])

        assert [ch.chunk_id for ch in history_chunks_newest_first(history)] == [c.chunk_id, a.chunk_id, b.chunk_id]
