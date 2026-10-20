# Path: tests/features/qa_generation/utils/test_context_packer.py

import uuid

import pytest

from app.core.anchors.domain.enums import AnchorKind
from app.features.qa_generation.utils.context_packer import (
    DOC_HEADER_TOKENS, DOCUMENT_HEADER_TOKENS, MAX_TITLE_WORDS, TITLE_TOKENS_PER_WORD, header_tokens,
    select_within_budget, order_for_reading, source_label, strip_repeated_overlap
)
from app.features.rag_retrieval.domain.entities import RetrievedChunk


def _chunk(raw_text: str, start: float, end: float, source_id: uuid.UUID = None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=source_id or uuid.uuid4(), text_payload="payload",
                          raw_text=raw_text, start_sec=start, end_sec=end, relevance_score=1.0)


def _page_chunk(raw_text: str, position: int, page: int, source_id: uuid.UUID = None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=source_id or uuid.uuid4(), text_payload="payload",
                          raw_text=raw_text, start_sec=0.0, end_sec=0.0, anchor_kind=AnchorKind.PAGE,
                          page_start=page, page_end=page, position=position, relevance_score=1.0)


def _words(text: str) -> int:
    return len(text.split())


class TestSelectWithinBudget:
    def test_keeps_priority_order_and_skips_what_does_not_fit(self):
        first = _chunk("a b c", 0, 1)            # 3 + 24 = 27
        too_big = _chunk("x " * 40, 1, 2)         # 40 + 24 = 64
        small = _chunk("d", 2, 3)                # 1 + 24 = 25

        selected = select_within_budget([first, too_big, small], 60, _words)

        assert selected == [first, small]

    def test_drops_duplicate_chunk_ids(self):
        chunk = _chunk("a", 0, 1)

        assert select_within_budget([chunk, chunk], 1000, _words) == [chunk]

    def test_zero_budget_selects_nothing(self):
        assert select_within_budget([_chunk("a", 0, 1)], 0, _words) == []

    def test_counts_payload_when_raw_text_is_missing(self):
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="one two three",
                               start_sec=0, end_sec=1, relevance_score=1.0)

        assert select_within_budget([chunk], 3 + DOC_HEADER_TOKENS, _words) == [chunk]
        assert select_within_budget([chunk], 2 + DOC_HEADER_TOKENS, _words) == []


class TestOrderForReading:
    def test_groups_sources_by_best_rank_and_sorts_each_by_time(self):
        source_a, source_b = uuid.uuid4(), uuid.uuid4()
        a_late, b_only, a_early = _chunk("", 50, 60, source_a), _chunk("", 5, 6, source_b), _chunk("", 10, 20, source_a)

        assert order_for_reading([b_only, a_late, a_early]) == [b_only, a_early, a_late]


class TestStripRepeatedOverlap:
    @pytest.fixture
    def source_id(self):
        return uuid.uuid4()

    def test_removes_the_sentence_repeated_from_the_previous_chunk(self, source_id):
        previous = _chunk("[s0 @ 00:00:00]: First part. So we go through the questioning now.", 0, 90, source_id)
        chunk = _chunk("[s0 @ 00:01:20]: So we go through the questioning now. Then the drill.\n[s1 @ 00:01:30]: Ok.",
                       80, 150, source_id)

        assert strip_repeated_overlap(previous, chunk) == "[s0 @ 00:01:20]: Then the drill.\n[s1 @ 00:01:30]: Ok."

    def test_drops_a_first_line_that_is_entirely_repeated(self, source_id):
        previous = _chunk("[s0 @ 00:00:00]: Intro. The whole repeated sentence here.", 0, 90, source_id)
        chunk = _chunk("[s0 @ 00:01:20]: The whole repeated sentence here.\n[s1 @ 00:01:30]: Reply.", 80, 150, source_id)

        assert strip_repeated_overlap(previous, chunk) == "[s1 @ 00:01:30]: Reply."

    @pytest.mark.parametrize("previous_start, previous_end, same_source", [
        (0, 70, True),     # previous ends before this chunk starts: not overlapping neighbours
        (0, 90, False),    # different recording
    ])
    def test_keeps_text_when_chunks_are_not_overlapping_neighbours(self, source_id, previous_start, previous_end,
                                                                    same_source):
        previous = _chunk("[s0 @ 00:00:00]: Intro. The whole repeated sentence here.", previous_start, previous_end,
                          source_id if same_source else uuid.uuid4())
        chunk = _chunk("[s0 @ 00:01:20]: The whole repeated sentence here. More.", 80, 150, source_id)

        assert strip_repeated_overlap(previous, chunk) == chunk.raw_text

    def test_ignores_short_coincidental_prefix(self, source_id):
        previous = _chunk("[s0 @ 00:00:00]: It ended with So", 0, 90, source_id)
        chunk = _chunk("[s0 @ 00:01:20]: So the next thing.", 80, 150, source_id)

        assert strip_repeated_overlap(previous, chunk) == chunk.raw_text

    def test_without_previous_chunk_returns_text(self):
        chunk = _chunk("[s0 @ 00:00:01]: Hello there.", 0, 5)

        assert strip_repeated_overlap(None, chunk) == "[s0 @ 00:00:01]: Hello there."


class TestDocumentChunks:
    def test_document_header_costs_more_than_a_transcript_header(self):
        assert header_tokens(_chunk("a", 0, 1)) == DOC_HEADER_TOKENS
        assert header_tokens(_page_chunk("a", 0, 1)) == DOCUMENT_HEADER_TOKENS > DOC_HEADER_TOKENS

    def test_reads_a_document_in_position_order_although_every_start_sec_is_zero(self):
        source = uuid.uuid4()
        page_3 = _page_chunk("third", 2, 3, source)
        page_1 = _page_chunk("first", 0, 1, source)
        page_2 = _page_chunk("second", 1, 2, source)

        assert order_for_reading([page_3, page_1, page_2]) == [page_1, page_2, page_3]

    def test_never_strips_text_from_a_document_chunk(self):
        source = uuid.uuid4()
        previous = _page_chunk("The Tenant shall pay rent monthly in advance.", 0, 1, source)
        chunk = _page_chunk("The Tenant shall pay rent monthly in advance. Late payment accrues interest.",
                            1, 1, source)

        assert strip_repeated_overlap(previous, chunk) == chunk.raw_text


class TestSourceTitleInHeaders:
    def test_a_title_costs_its_words_on_top_of_the_header(self):
        titled = _chunk("text", 0, 1).model_copy(update={"source_title": "Statistics 110, Lecture 4"})
        assert header_tokens(titled) == DOC_HEADER_TOKENS + TITLE_TOKENS_PER_WORD * 4

    def test_a_long_title_is_cut_to_max_words(self):
        titled = _chunk("text", 0, 1).model_copy(update={"source_title": "word " * 40})
        assert source_label(titled) == " ".join(["word"] * MAX_TITLE_WORDS) + "…"
        assert header_tokens(titled) == DOC_HEADER_TOKENS + TITLE_TOKENS_PER_WORD * MAX_TITLE_WORDS

    def test_no_title_no_label(self):
        assert source_label(_chunk("text", 0, 1)) == ""
