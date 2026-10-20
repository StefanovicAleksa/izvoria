# Path: tests/features/qa_reasoning/utils/test_evidence_formatter.py

import uuid

from app.features.qa_reasoning.utils.evidence_formatter import build_evidence_block, build_decision_history_block
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestEvidenceFormatter:
    def test_build_evidence_block_empty_pool(self):
        block = build_evidence_block([])

        assert block.formatted_text == "(No evidence gathered yet.)"
        assert block.tag_mapping == {}

    def test_build_evidence_block_tags_with_short_ids(self):
        first = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="fallback text",
                               raw_text="verbatim text", start_sec=1.0, end_sec=2.0, relevance_score=1.0)
        second = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="second payload",
                                start_sec=3.0, end_sec=4.5, relevance_score=1.0)

        block = build_evidence_block([first, second])

        assert block.formatted_text == "[E1] (1.0s-2.0s):\nverbatim text\n\n[E2] (3.0s-4.5s):\nsecond payload"
        assert block.tag_mapping == {"E1": first.chunk_id, "E2": second.chunk_id}
        assert str(first.chunk_id) not in block.formatted_text

    def test_build_evidence_block_skips_chunks_over_budget_and_numbers_the_rest(self):
        chunks = [
            RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text,
                           start_sec=0, end_sec=1, relevance_score=1.0)
            for text in ("one two", "w " * 50, "three")
        ]
        count_words = lambda text: len(text.split())  # noqa: E731

        # "[E1] (0.0s-1.0s): one two" = 4 words + 2 separator; the long one doesn't fit; "three" = 3 + 2
        block = build_evidence_block(chunks, token_budget=12, count_tokens=count_words)

        assert block.tag_mapping == {"E1": chunks[0].chunk_id, "E2": chunks[2].chunk_id}
        assert "[E2] (0.0s-1.0s):\nthree" in block.formatted_text
        assert "w w" not in block.formatted_text

    def test_build_evidence_block_with_nothing_fitting_reports_no_evidence(self):
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="text",
                               start_sec=0, end_sec=1, relevance_score=1.0)

        block = build_evidence_block([chunk], token_budget=0, count_tokens=lambda text: 1)

        assert block.formatted_text == "(No evidence gathered yet.)"
        assert block.tag_mapping == {}

    def test_build_evidence_block_truncates_and_falls_back(self):
        massive_text = "A" * 5000
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=massive_text,
                               start_sec=0, end_sec=1, relevance_score=1.0)

        block = build_evidence_block([chunk]).formatted_text

        assert "A" * 3000 in block
        assert "A" * 3001 not in block

    def test_build_decision_history_block_empty(self):
        assert build_decision_history_block([]) == "(No prior reasoning turns.)"

    def test_build_decision_history_block_lists_each_turn(self):
        decisions = [
            ReasonerDecision(decision=ReasonerDecisionType.REFINE_QUERY, chain_of_thought="x",
                             interim_analysis="y", refined_query="narrower query"),
            ReasonerDecision(decision=ReasonerDecisionType.ANSWER_READY, chain_of_thought="x", interim_analysis="y"),
        ]

        block = build_decision_history_block(decisions)

        assert "Iteration 1: refine_query" in block
        assert "narrower query" in block
        assert "Iteration 2: answer_ready" in block


class TestEvidenceSourceTitles:
    def test_a_titled_chunk_names_its_source(self):
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="p", raw_text="Bayes.",
                               start_sec=1.0, end_sec=2.0, relevance_score=1.0, source_title="Lecture 4")

        assert build_evidence_block([chunk]).formatted_text == "[E1] Lecture 4 (1.0s-2.0s):\nBayes."
