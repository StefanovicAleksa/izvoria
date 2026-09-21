# Canvas: test_citation_parser_v2
# Path: tests/features/qa_generation/utils/test_citation_parser.py

import uuid
import pytest
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.utils.citation_parser import (
    build_context_block, build_numbered_context_block, doc_numbers, number_for_reading, parse_citations,
    normalize_citation_tags
)


class TestCitationParser:
    @pytest.fixture
    def sample_chunks(self):
        return [
            RetrievedChunk(
                chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
                text_payload="X-ray shows a broken rib.",
                start_sec=10.0, end_sec=15.0, relevance_score=0.9
            ),
            RetrievedChunk(
                chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
                text_payload="Patient reports high pain.",
                start_sec=20.0, end_sec=25.0, relevance_score=0.8
            )
        ]

    def test_build_context_block_formats_tags(self, sample_chunks):
        block = build_context_block(sample_chunks)

        assert "[Doc 1]" in block.formatted_text
        assert "[Doc 2]" in block.formatted_text
        assert block.doc_mapping["[Doc 1]"] == sample_chunks[0].chunk_id
        assert block.doc_mapping["[Doc 2]"] == sample_chunks[1].chunk_id

    def test_parse_citations_matches_ids(self, sample_chunks):
        generated_text = "The rib is broken [Doc 1] and pain is high [Doc 2]."
        doc_mapping = {
            "[Doc 1]": sample_chunks[0].chunk_id,
            "[Doc 2]": sample_chunks[1].chunk_id
        }

        # Inject mock physical paths for retrieval simulation
        for chunk in sample_chunks:
            object.__setattr__(chunk, 'physical_path', '/vault/clip.mp4')

        citations = parse_citations(generated_text, sample_chunks, doc_mapping)

        assert len(citations) == 2
        assert citations[0].doc_id == "[Doc 1]"
        assert citations[0].start_sec == 10.0

    def test_parse_citations_carries_storage_links_and_no_placeholder_path(self, sample_chunks):
        clip_id = uuid.uuid4()
        object.__setattr__(sample_chunks[0], 'extracted_file_id', clip_id)
        doc_mapping = {"[Doc 1]": sample_chunks[0].chunk_id, "[Doc 2]": sample_chunks[1].chunk_id}

        citations = parse_citations("Claim [Doc 2] and another [Doc 1].", sample_chunks, doc_mapping)

        assert [(c.doc_id, c.chunk_id, c.source_id, c.extracted_file_id, c.physical_path) for c in citations] == [
            ("[Doc 1]", sample_chunks[0].chunk_id, sample_chunks[0].source_id, clip_id, None),
            ("[Doc 2]", sample_chunks[1].chunk_id, sample_chunks[1].source_id, None, None),
        ]

    def test_context_block_shows_only_raw_transcript_with_time_range(self):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            text_payload="[Audio Context: The doctor confirms a fracture.]\n\n[Speaker_0 @ 00:12:31]: The X-ray is clear.",
            raw_text="[Speaker_0 @ 00:12:31]: The X-ray is clear.",
            start_sec=751.0, end_sec=842.0, relevance_score=0.9
        )

        block = build_context_block([chunk])

        assert block.formatted_text == "[Doc 1] (00:12:31–00:14:02)\n[Speaker_0 @ 00:12:31]: The X-ray is clear.\n"
        assert "Audio Context" not in block.formatted_text

    def test_context_block_falls_back_to_payload_without_raw_text(self, sample_chunks):
        block = build_context_block(sample_chunks[:1])
        assert block.formatted_text == "[Doc 1] (00:00:10–00:00:15)\nX-ray shows a broken rib.\n"

    def test_context_block_orders_by_source_then_time_and_drops_repeated_overlap(self):
        source_a, source_b = uuid.uuid4(), uuid.uuid4()
        later_a = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=source_a, text_payload="p",
            raw_text="[Speaker_0 @ 00:01:08]: The level is 160.84 on the chart. Beyond there we look at 161.17.",
            start_sec=68.0, end_sec=155.0, relevance_score=0.9
        )
        other_b = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=source_b, text_payload="p", raw_text="[Speaker_1 @ 00:00:05]: Other.",
            start_sec=5.0, end_sec=9.0, relevance_score=0.8
        )
        earlier_a = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=source_a, text_payload="p",
            raw_text="[Speaker_0 @ 00:00:03]: We start here. The level is 160.84 on the chart.",
            start_sec=3.0, end_sec=85.0, relevance_score=0.7
        )

        # Best-first input: source A leads (its best chunk ranks first), and inside A time order wins
        block = build_context_block([later_a, other_b, earlier_a])

        assert block.doc_mapping == {
            "[Doc 1]": earlier_a.chunk_id, "[Doc 2]": later_a.chunk_id, "[Doc 3]": other_b.chunk_id
        }
        assert block.formatted_text == (
            "[Doc 1] (00:00:03–00:01:25)\n[Speaker_0 @ 00:00:03]: We start here. The level is 160.84 on the chart.\n\n"
            "[Doc 2] (00:01:08–00:02:35)\n[Speaker_0 @ 00:01:08]: Beyond there we look at 161.17.\n\n"
            "[Doc 3] (00:00:05–00:00:09)\n[Speaker_1 @ 00:00:05]: Other.\n"
        )


class TestNormalizeCitationTags:
    @pytest.mark.parametrize("text, doc_count, expected", [
        ("Fact [Doc 1].", 3, "Fact [Doc 1]."),
        ("Fact [doc 2].", 3, "Fact [Doc 2]."),
        ("Fact [Doc 1, Doc 3].", 3, "Fact [Doc 1][Doc 3]."),
        ("Fact [Doc 1, 3].", 3, "Fact [Doc 1][Doc 3]."),
        ("Fact [Docs 1-3].", 3, "Fact [Doc 1][Doc 2][Doc 3]."),
        ("Fact [Doc 1–2].", 3, "Fact [Doc 1][Doc 2]."),
        ("Fact [Doc 2 and 3].", 3, "Fact [Doc 2][Doc 3]."),
        ("Fact [Doc 1; Doc 1].", 3, "Fact [Doc 1]."),
        ("Fact [Doc 2, 9].", 3, "Fact [Doc 2]."),
        ("Fact [Doc 7].", 3, "Fact ."),
        ("Fact [Doc 1-500].", 3, "Fact [Doc 1]."),
        ("See [Documentation] and [Doc 1: page 4].", 3, "See [Documentation] and [Doc 1: page 4]."),
        ("[Speaker_0 @ 00:01:02] said so [Doc 1].", 3, "[Speaker_0 @ 00:01:02] said so [Doc 1]."),
    ])
    def test_normalizes_citation_groups(self, text, doc_count, expected):
        assert normalize_citation_tags(text, doc_count) == expected

    def test_parse_citations_finds_every_document_in_a_group(self):
        chunks = [
            RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=f"t{i}",
                           start_sec=float(i), end_sec=float(i + 1), relevance_score=1.0)
            for i in range(3)
        ]
        doc_mapping = {f"[Doc {i + 1}]": c.chunk_id for i, c in enumerate(chunks)}

        citations = parse_citations("Both happened [Doc 1, 3].", chunks, doc_mapping)

        assert [(c.doc_id, c.chunk_id) for c in citations] == [
            ("[Doc 1]", chunks[0].chunk_id), ("[Doc 3]", chunks[2].chunk_id)
        ]


class TestConversationNumbering:
    @staticmethod
    def _chunk(start: float, source_id: uuid.UUID, text: str = "text") -> RetrievedChunk:
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=source_id, text_payload=text,
                              start_sec=start, end_sec=start + 1, relevance_score=0.0)

    def test_context_block_numbers_from_start_number(self):
        source = uuid.uuid4()
        late, early = self._chunk(30, source, "late"), self._chunk(10, source, "early")

        block = build_context_block([late, early], start_number=7)

        assert block.doc_mapping == {"[Doc 7]": early.chunk_id, "[Doc 8]": late.chunk_id}
        assert block.formatted_text.index("[Doc 7]") < block.formatted_text.index("[Doc 8]")

    def test_number_for_reading_orders_by_source_then_time(self):
        first_source, second_source = uuid.uuid4(), uuid.uuid4()
        best = self._chunk(50, first_source)
        other = self._chunk(0, second_source)
        earlier_same_source = self._chunk(5, first_source)

        numbered = number_for_reading([best, other, earlier_same_source], start_number=3)

        assert [(n, c.chunk_id) for n, c in numbered] == [
            (3, earlier_same_source.chunk_id), (4, best.chunk_id), (5, other.chunk_id)
        ]

    def test_numbered_block_keeps_given_numbers_with_gaps(self):
        source = uuid.uuid4()
        a, b = self._chunk(0, source, "alpha"), self._chunk(100, source, "beta")

        block = build_numbered_context_block([(2, a), (9, b)])

        assert block.doc_mapping == {"[Doc 2]": a.chunk_id, "[Doc 9]": b.chunk_id}
        assert "[Doc 2] (00:00:00–00:00:01)\nalpha" in block.formatted_text
        assert "[Doc 9] (00:01:40–00:01:41)\nbeta" in block.formatted_text

    def test_doc_numbers_reads_mapping_keys(self):
        assert doc_numbers({"[Doc 2]": uuid.uuid4(), "[Doc 9]": uuid.uuid4()}) == {2, 9}

    @pytest.mark.parametrize("text, expected", [
        ("A [Doc 2] B [Doc 3]", "A [Doc 2] B "),
        ("A [Doc 1, 9]", "A [Doc 9]"),
        ("A [Docs 8-10]", "A [Doc 9]"),
    ])
    def test_normalize_with_valid_numbers_drops_tags_not_shown(self, text, expected):
        assert normalize_citation_tags(text, 0, valid_numbers={2, 9}) == expected

    def test_parse_citations_resolves_conversation_wide_numbers(self):
        source = uuid.uuid4()
        a, b = self._chunk(0, source), self._chunk(10, source)
        mapping = {"[Doc 4]": a.chunk_id, "[Doc 11]": b.chunk_id}

        citations = parse_citations("X [Doc 11]. Y [Doc 4, 5].", [a, b], mapping)

        assert [(c.doc_id, c.chunk_id) for c in citations] == [("[Doc 4]", a.chunk_id), ("[Doc 11]", b.chunk_id)]
