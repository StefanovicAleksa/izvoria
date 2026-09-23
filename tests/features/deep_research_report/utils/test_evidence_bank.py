# Path: tests/features/deep_research_report/utils/test_evidence_bank.py

import uuid

import pytest

from app.features.deep_research_report.domain.entities import EvidenceEntry
from app.features.deep_research_report.domain.enums import EvidenceOrigin
from app.features.deep_research_report.utils.evidence_bank import (
    DIGEST_CHARS, build_bank_index, build_digest, build_section_digest, chunks_for_section, dedupe_hits,
    format_timestamp, number_bank, pack_section_evidence,
)
from app.features.rag_retrieval.domain.entities import RetrievedChunk


def _chunk(
        source_id: uuid.UUID,
        start_sec: float = 0.0,
        raw_text: str = "[SPEAKER_00 @ 00:00:00]: The bag was sealed.",
        text_payload: str = "GENERATED CONTEXT. [SPEAKER_00 @ 00:00:00]: The bag was sealed.",
        relevance_score: float = 0.5,
        rerank_score=None,
        speakers=None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        source_id=source_id,
        text_payload=text_payload,
        raw_text=raw_text,
        speaker_labels=speakers if speakers is not None else ["SPEAKER_00"],
        start_sec=start_sec,
        end_sec=start_sec + 30,
        relevance_score=relevance_score,
        rerank_score=rerank_score,
    )


class TestBuildDigest:
    def test_digest_comes_from_raw_text_never_from_the_generated_payload(self):
        """
        The product's core rule: model-generated situating context lives in `text_payload` to make
        retrieval work, and is never citable source material. A digest built from it would let the
        planner assign -- and ultimately the report cite -- a sentence nobody said.
        """
        chunk = _chunk(
            uuid.uuid4(),
            raw_text="[SPEAKER_01 @ 00:00:10]: We logged it at nine.",
            text_payload="This clip discusses evidence handling. [SPEAKER_01 @ 00:00:10]: We logged it at nine.",
        )

        digest = build_digest(chunk)

        assert digest == "[SPEAKER_01 @ 00:00:10]: We logged it at nine."
        assert "This clip discusses evidence handling" not in digest

    def test_digest_falls_back_to_payload_only_when_there_is_no_raw_text(self):
        chunk = _chunk(uuid.uuid4(), raw_text="", text_payload="Only payload available.")

        assert build_digest(chunk) == "Only payload available."

    def test_digest_is_truncated_and_whitespace_normalised(self):
        chunk = _chunk(uuid.uuid4(), raw_text="word  \n\n word " * 200)

        digest = build_digest(chunk)

        assert digest.endswith("…")
        assert len(digest) <= DIGEST_CHARS + 1
        assert "\n" not in digest


class TestDedupeHits:
    def test_a_chunk_found_by_several_sub_questions_survives_once(self):
        source_id = uuid.uuid4()
        chunk = _chunk(source_id)

        deduped = dedupe_hits([
            (chunk, EvidenceOrigin.FAN_OUT, "q1"),
            (chunk, EvidenceOrigin.FAN_OUT, "q2"),
            (chunk, EvidenceOrigin.SWEEP, None),
        ])

        assert len(deduped) == 1

    def test_the_best_scoring_occurrence_wins(self):
        source_id = uuid.uuid4()
        weak = _chunk(source_id, rerank_score=0.2)
        strong = weak.model_copy(update={"rerank_score": 0.9})

        deduped = dedupe_hits([
            (weak, EvidenceOrigin.FAN_OUT, "weak question"),
            (strong, EvidenceOrigin.FAN_OUT, "strong question"),
        ])

        assert len(deduped) == 1
        chunk, _, found_by = deduped[0]
        assert chunk.rerank_score == 0.9
        assert found_by == "strong question"

    def test_rrf_score_is_used_when_the_reranker_did_not_run(self):
        source_id = uuid.uuid4()
        weak = _chunk(source_id, relevance_score=0.1, rerank_score=None)
        strong = weak.model_copy(update={"relevance_score": 0.8})

        deduped = dedupe_hits([
            (weak, EvidenceOrigin.FAN_OUT, "weak"),
            (strong, EvidenceOrigin.FAN_OUT, "strong"),
        ])

        assert deduped[0][2] == "strong"

    def test_a_negatively_reranked_occurrence_still_beats_an_unreranked_sweep_hit(self):
        """
        The two scores live on different scales, so the reranked occurrence must win outright.

        A cross-encoder emits raw logits and RAG_RERANK_MIN_SCORE always keeps a query's top candidate
        however it scored, so a rescued hit can sit at -2.0 while the sweep's RRF hit sits at ~0.016.
        Comparing the raw numbers handed the sweep copy the win, which dropped the chunk's rerank_score,
        flipped its origin to SWEEP and cleared found_by -- so the sub-question that actually surfaced it
        got credited nothing in QueryAttempt.chunks_added.
        """
        source_id = uuid.uuid4()
        reranked = _chunk(source_id, relevance_score=0.03, rerank_score=-2.0)
        swept = reranked.model_copy(update={"relevance_score": 0.016, "rerank_score": None})

        deduped = dedupe_hits([
            (swept, EvidenceOrigin.SWEEP, None),
            (reranked, EvidenceOrigin.FAN_OUT, "the question that found it"),
        ])

        assert len(deduped) == 1
        chunk, origin, found_by = deduped[0]
        assert chunk.rerank_score == -2.0
        assert origin == EvidenceOrigin.FAN_OUT
        assert found_by == "the question that found it"

    def test_the_winner_is_independent_of_the_order_the_occurrences_arrive_in(self):
        source_id = uuid.uuid4()
        reranked = _chunk(source_id, relevance_score=0.03, rerank_score=-2.0)
        swept = reranked.model_copy(update={"relevance_score": 0.016, "rerank_score": None})

        forwards = dedupe_hits([(reranked, EvidenceOrigin.FAN_OUT, "q"), (swept, EvidenceOrigin.SWEEP, None)])
        backwards = dedupe_hits([(swept, EvidenceOrigin.SWEEP, None), (reranked, EvidenceOrigin.FAN_OUT, "q")])

        assert forwards[0][1] == EvidenceOrigin.FAN_OUT
        assert backwards[0][1] == EvidenceOrigin.FAN_OUT

    def test_distinct_chunks_are_all_kept(self):
        source_id = uuid.uuid4()
        hits = [(_chunk(source_id, start_sec=i * 10), EvidenceOrigin.FAN_OUT, "q") for i in range(4)]

        assert len(dedupe_hits(hits)) == 4


class TestNumberBank:
    def test_numbers_are_assigned_in_reading_order_not_rank_order(self):
        """
        Reading order -- grouped by source, chronological within one -- so a reader following citations
        through the references table moves through each recording as a timeline.
        """
        source_id = uuid.uuid4()
        late = _chunk(source_id, start_sec=300)
        early = _chunk(source_id, start_sec=10)

        entries = number_bank(
            [(late, EvidenceOrigin.FAN_OUT, "q"), (early, EvidenceOrigin.FAN_OUT, "q")],
            {source_id: "Deposition"},
        )

        by_number = {e.doc_number: e for e in entries}
        assert by_number[1].chunk_id == early.chunk_id
        assert by_number[2].chunk_id == late.chunk_id

    def test_numbering_is_contiguous_from_one(self):
        source_id = uuid.uuid4()
        hits = [(_chunk(source_id, start_sec=i * 10), EvidenceOrigin.FAN_OUT, "q") for i in range(5)]

        entries = number_bank(hits, {source_id: "Recording"})

        assert [e.doc_number for e in entries] == [1, 2, 3, 4, 5]

    def test_provenance_is_denormalised_onto_every_entry(self):
        """Stored, not looked up later: a re-ingest deletes the chunk but must not break the citation."""
        source_id = uuid.uuid4()
        chunk = _chunk(source_id, start_sec=751, speakers=["SPEAKER_01", "SPEAKER_02"])

        entry = number_bank([(chunk, EvidenceOrigin.SWEEP, None)], {source_id: "Ramirez 2024-03-11"})[0]

        assert entry.source_name == "Ramirez 2024-03-11"
        assert entry.speaker_labels == ["SPEAKER_01", "SPEAKER_02"]
        assert entry.start_sec == 751
        assert entry.origin == EvidenceOrigin.SWEEP
        assert entry.found_by is None
        assert entry.digest

    def test_an_unknown_source_still_gets_a_readable_name(self):
        chunk = _chunk(uuid.uuid4())

        entry = number_bank([(chunk, EvidenceOrigin.FAN_OUT, "q")], {})[0]

        assert entry.source_name == "Unknown source"

    def test_scrubbed_ids_are_flagged(self):
        source_id = uuid.uuid4()
        chunk = _chunk(source_id)

        entry = number_bank(
            [(chunk, EvidenceOrigin.FAN_OUT, "q")], {source_id: "R"}, scrubbed_ids=[chunk.chunk_id]
        )[0]

        assert entry.scrubbed is True

    def test_empty_bank(self):
        assert number_bank([], {}) == []


class TestBuildBankIndex:
    def test_one_compact_line_per_entry(self):
        entry = EvidenceEntry(
            doc_number=7, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            source_name="Deposition", start_sec=751, end_sec=784,
            speaker_labels=["SPEAKER_01"], digest="The bag was sealed.",
        )

        index = build_bank_index([entry])

        assert index == "[Doc 7] Deposition (00:12:31–00:13:04, SPEAKER_01): The bag was sealed."

    def test_unlabelled_speakers_are_stated_rather_than_left_blank(self):
        entry = EvidenceEntry(
            doc_number=1, chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            source_name="R", start_sec=0, end_sec=1, speaker_labels=[], digest="x",
        )

        assert "unlabelled" in build_bank_index([entry])


class TestChunksForSection:
    def test_returns_pairs_in_ascending_doc_order(self):
        source_id = uuid.uuid4()
        chunks = [_chunk(source_id, start_sec=i * 10) for i in range(3)]
        entries = number_bank(
            [(c, EvidenceOrigin.FAN_OUT, "q") for c in chunks], {source_id: "R"}
        )
        by_number = {e.doc_number: e for e in entries}
        lookup = {c.chunk_id: c for c in chunks}

        pairs = chunks_for_section([3, 1], by_number, lookup)

        assert [number for number, _ in pairs] == [1, 3]

    def test_a_doc_number_whose_chunk_was_deleted_is_dropped_not_rendered_empty(self):
        """
        A re-ingest between gathering and writing deletes chunks. Rendering the header with no text under
        it would invite the writer to cite a document it cannot actually read.
        """
        source_id = uuid.uuid4()
        chunk = _chunk(source_id)
        entries = number_bank([(chunk, EvidenceOrigin.FAN_OUT, "q")], {source_id: "R"})
        by_number = {e.doc_number: e for e in entries}

        assert chunks_for_section([1], by_number, {}) == []

    def test_unknown_doc_numbers_are_ignored(self):
        assert chunks_for_section([42], {}, {}) == []

    def test_duplicate_numbers_collapse(self):
        source_id = uuid.uuid4()
        chunk = _chunk(source_id)
        entries = number_bank([(chunk, EvidenceOrigin.FAN_OUT, "q")], {source_id: "R"})
        by_number = {e.doc_number: e for e in entries}

        pairs = chunks_for_section([1, 1, 1], by_number, {chunk.chunk_id: chunk})

        assert len(pairs) == 1


class TestPackSectionEvidence:
    def test_keeps_pairs_while_they_fit(self):
        source_id = uuid.uuid4()
        pairs = [(n, _chunk(source_id, start_sec=n * 10)) for n in (1, 2, 3)]

        packed = pack_section_evidence(pairs, token_budget=1000, count_tokens=lambda _: 100)

        assert len(packed) == 3

    def test_a_chunk_that_does_not_fit_is_skipped_so_a_shorter_one_can_still_use_the_room(self):
        source_id = uuid.uuid4()
        huge = _chunk(source_id, raw_text="huge")
        small = _chunk(source_id, raw_text="small", start_sec=10)
        costs = {"huge": 10_000, "small": 10}

        packed = pack_section_evidence(
            [(1, huge), (2, small)], token_budget=100, count_tokens=lambda text: costs[text]
        )

        assert [number for number, _ in packed] == [2]

    def test_doc_numbers_are_preserved_not_renumbered(self):
        """The numbers are bank-wide and already fixed; repacking must not shift them."""
        source_id = uuid.uuid4()
        pairs = [(7, _chunk(source_id)), (12, _chunk(source_id, start_sec=10))]

        packed = pack_section_evidence(pairs, token_budget=10_000, count_tokens=lambda _: 10)

        assert [number for number, _ in packed] == [7, 12]

    def test_zero_budget_keeps_nothing(self):
        source_id = uuid.uuid4()

        assert pack_section_evidence([(1, _chunk(source_id))], 0, lambda _: 1) == []


class TestBuildSectionDigest:
    def test_trims_long_bodies_and_keeps_headings(self):
        digest = build_section_digest([("Handling", "word " * 500)], max_chars=50)

        assert digest.startswith("## Handling")
        assert digest.endswith("…")

    def test_joins_several_sections(self):
        digest = build_section_digest([("A", "one"), ("B", "two")])

        assert "## A" in digest and "## B" in digest


@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00:00"),
    (59, "00:00:59"),
    (751, "00:12:31"),
    (3661, "01:01:01"),
    (-5, "00:00:00"),
])
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected
