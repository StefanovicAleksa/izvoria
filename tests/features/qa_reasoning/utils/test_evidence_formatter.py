# Path: tests/features/qa_reasoning/utils/test_evidence_formatter.py

import uuid

from app.features.qa_reasoning.utils.evidence_formatter import build_evidence_block, build_decision_history_block
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestEvidenceFormatter:
    def test_build_evidence_block_empty_pool(self):
        assert build_evidence_block([]) == "(No evidence gathered yet.)"

    def test_build_evidence_block_tags_by_chunk_id(self):
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="fallback text",
                               raw_text="verbatim text", start_sec=1.0, end_sec=2.0, relevance_score=1.0)

        block = build_evidence_block([chunk])

        assert str(chunk.chunk_id) in block
        assert "verbatim text" in block
        assert "fallback text" not in block  # raw_text preferred over text_payload

    def test_build_evidence_block_truncates_and_falls_back(self):
        massive_text = "A" * 5000
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=massive_text,
                               start_sec=0, end_sec=1, relevance_score=1.0)

        block = build_evidence_block([chunk])

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
