# Canvas: test_citation_parser_v2
# Path: tests/features/qa_generation/utils/test_citation_parser.py

import uuid
import pytest
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.utils.citation_parser import build_context_block, parse_citations


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
