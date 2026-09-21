# Path: tests/features/qa_generation/utils/test_conversation_packer.py

import uuid
import pytest

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import ConversationHistory, ConversationTurn, NumberedChunk
from app.features.qa_generation.utils.context_packer import DOC_HEADER_TOKENS
from app.features.qa_generation.utils.conversation_packer import (
    MESSAGE_OVERHEAD_TOKENS, PackedConversation, PackedTurn, pack_conversation
)

SOURCE = uuid.uuid4()
TEMPLATE = 10


def _words(n: int) -> str:
    return " ".join(["w"] * n)


def _count(text: str) -> int:
    return len(text.split())


def _chunk(words: int, start: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=SOURCE, text_payload=_words(words),
                          start_sec=start, end_sec=start + 1, relevance_score=0.0)


def _chunk_cost(words: int) -> int:
    return words + DOC_HEADER_TOKENS


def _turn_text_cost(question: str, answer: str) -> int:
    return _count(question) + _count(answer) + TEMPLATE + 2 * MESSAGE_OVERHEAD_TOKENS


def _current_cost(question: str) -> int:
    return _count(question) + TEMPLATE + MESSAGE_OVERHEAD_TOKENS


def _pack(history, question, chunks, budget):
    return pack_conversation(history, question, chunks, budget, TEMPLATE, _count)


class TestPackConversation:
    def test_without_history_numbers_new_evidence_in_reading_order_from_next_doc_number(self):
        late, early = _chunk(3, 50), _chunk(3, 10)

        packed = _pack(ConversationHistory(next_doc_number=5), "Q", [late, early], budget=10_000)

        assert packed.turns == []
        assert [(n.doc_number, n.chunk.chunk_id) for n in packed.new_evidence] == [
            (5, early.chunk_id), (6, late.chunk_id)
        ]
        assert packed.doc_mapping == {"[Doc 5]": early.chunk_id, "[Doc 6]": late.chunk_id}

    def test_everything_fits_keeps_every_turn_and_its_evidence(self):
        old = _chunk(3, 0)
        history = ConversationHistory(
            turns=[ConversationTurn(question="first", answer="one [Doc 1]",
                                    evidence=[NumberedChunk(doc_number=1, chunk=old)])],
            next_doc_number=2
        )
        new = _chunk(3, 5)

        packed = _pack(history, "second", [new], budget=10_000)

        assert packed.turns == [PackedTurn(question="first", answer="one [Doc 1]",
                                           evidence=[NumberedChunk(doc_number=1, chunk=old)])]
        assert [(n.doc_number, n.chunk.chunk_id) for n in packed.new_evidence] == [(2, new.chunk_id)]
        assert [n.doc_number for n in packed.shown_evidence] == [1, 2]

    def test_chunk_already_shown_earlier_keeps_its_old_number(self):
        old = _chunk(3, 0)
        history = ConversationHistory(
            turns=[ConversationTurn(question="first", answer="a", evidence=[NumberedChunk(doc_number=1, chunk=old)])],
            next_doc_number=2
        )
        fresh = _chunk(3, 9)

        packed = _pack(history, "again", [old, fresh], budget=10_000)

        assert [n.doc_number for n in packed.turns[0].evidence] == [1]
        assert [(n.doc_number, n.chunk.chunk_id) for n in packed.new_evidence] == [(2, fresh.chunk_id)]

    def test_oldest_turns_lose_their_evidence_first(self):
        oldest_chunk, newer_chunk = _chunk(20, 0), _chunk(20, 5)
        history = ConversationHistory(
            turns=[
                ConversationTurn(question="q1", answer="a1 [Doc 1]",
                                 evidence=[NumberedChunk(doc_number=1, chunk=oldest_chunk)]),
                ConversationTurn(question="q2", answer="a2 [Doc 2]",
                                 evidence=[NumberedChunk(doc_number=2, chunk=newer_chunk)]),
            ],
            next_doc_number=3
        )
        budget = (_current_cost("q3") + _turn_text_cost("q1", "a1 [Doc 1]") + _turn_text_cost("q2", "a2 [Doc 2]")
                  + _chunk_cost(20))

        packed = _pack(history, "q3", [], budget)

        assert [t.question for t in packed.turns] == ["q1", "q2"]
        assert packed.turns[0].evidence == []
        assert [n.doc_number for n in packed.turns[1].evidence] == [2]

    def test_turns_that_do_not_fit_are_dropped_oldest_first(self):
        history = ConversationHistory(
            turns=[ConversationTurn(question=_words(50), answer="a1"),
                   ConversationTurn(question="q2", answer="a2")],
            next_doc_number=1
        )
        current = _chunk(5, 0)
        budget = _current_cost("q3") + _chunk_cost(5) + _turn_text_cost("q2", "a2")

        packed = _pack(history, "q3", [current], budget)

        assert [t.question for t in packed.turns] == ["q2"]
        assert [n.doc_number for n in packed.new_evidence] == [1]

    def test_older_turn_is_dropped_once_a_newer_one_does_not_fit(self):
        history = ConversationHistory(
            turns=[ConversationTurn(question="q1", answer="a1"),
                   ConversationTurn(question=_words(50), answer="a2")],
        )
        budget = _current_cost("q3") + _turn_text_cost("q1", "a1")

        packed = _pack(history, "q3", [], budget)

        assert packed.turns == []

    def test_current_evidence_has_priority_over_history_and_skips_what_does_not_fit(self):
        best, huge, small = _chunk(10, 0), _chunk(500, 5), _chunk(2, 9)
        history = ConversationHistory(turns=[ConversationTurn(question="q1", answer="a1")], next_doc_number=4)
        budget = _current_cost("q2") + _chunk_cost(10) + _chunk_cost(2)

        packed = _pack(history, "q2", [best, huge, small], budget)

        assert [n.chunk.chunk_id for n in packed.new_evidence] == [best.chunk_id, small.chunk_id]
        assert [n.doc_number for n in packed.new_evidence] == [4, 5]
        assert packed.turns == []

    def test_repeated_chunk_of_a_dropped_turn_is_shown_as_new_evidence(self):
        repeated = _chunk(3, 0)
        history = ConversationHistory(
            turns=[ConversationTurn(question=_words(100), answer="a1",
                                    evidence=[NumberedChunk(doc_number=1, chunk=repeated)])],
            next_doc_number=2
        )
        budget = _current_cost("q2") + _chunk_cost(3)

        packed = _pack(history, "q2", [repeated], budget)

        assert packed.turns == []
        assert [(n.doc_number, n.chunk.chunk_id) for n in packed.new_evidence] == [(2, repeated.chunk_id)]

    def test_duplicate_current_chunks_are_counted_once(self):
        chunk = _chunk(3, 0)

        packed = _pack(ConversationHistory(), "q", [chunk, chunk], budget=_current_cost("q") + _chunk_cost(3))

        assert [n.chunk.chunk_id for n in packed.new_evidence] == [chunk.chunk_id]

    def test_chunk_repeated_across_turns_is_shown_once_under_its_latest_number(self):
        chunk = _chunk(3, 0)
        history = ConversationHistory(
            turns=[
                ConversationTurn(question="q1", answer="a1", evidence=[NumberedChunk(doc_number=1, chunk=chunk)]),
                ConversationTurn(question="q2", answer="a2", evidence=[NumberedChunk(doc_number=2, chunk=chunk)]),
            ],
            next_doc_number=3
        )

        packed = _pack(history, "q3", [], budget=10_000)

        assert packed.turns[0].evidence == []
        assert [n.doc_number for n in packed.turns[1].evidence] == [2]

    @pytest.mark.parametrize("budget", [0, -5])
    def test_no_budget_shows_nothing(self, budget):
        history = ConversationHistory(turns=[ConversationTurn(question="q1", answer="a1")])

        packed = _pack(history, "q2", [_chunk(3, 0)], budget)

        assert packed == PackedConversation(turns=[], new_evidence=[])
