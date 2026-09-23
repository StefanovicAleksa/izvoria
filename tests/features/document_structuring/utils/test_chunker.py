# Path: tests/features/document_structuring/utils/test_chunker.py

from app.features.document_structuring.domain.entities import Block
from app.features.document_structuring.domain.enums import BlockKind
from app.features.document_structuring.utils.chunker import build_chunks

SEP = " › "


def heading(text, level=2, page=1):
    return Block(kind=BlockKind.HEADING, text=text, level=level, page_no=page)


def para(words, page=1, word="lorem"):
    return Block(kind=BlockKind.PARAGRAPH, text=" ".join([word] * (words - 1) + ["end."]), page_no=page)


class TestBuildChunks:
    def test_short_document_is_one_chunk(self):
        blocks = [heading("RECEIPT", level=1), para(20), heading("Article 2 Items"), para(30)]

        [chunk] = build_chunks(blocks, max_words=250, short_document_words=400, separator=SEP)

        assert (chunk.page_start, chunk.page_end) == (1, 1)
        assert chunk.text.startswith("RECEIPT\n\n")

    def test_each_clause_starts_a_chunk_with_its_section_path(self):
        blocks = [heading("LEASE", level=1), para(150), heading("Article 1 Definitions"), para(150),
                  heading("Article 2 Rent", page=2), para(150, page=2)]

        chunks = build_chunks(blocks, max_words=250, short_document_words=100, separator=SEP)

        assert [c.section_path for c in chunks] == ["LEASE", "LEASE › Article 1 Definitions", "LEASE › Article 2 Rent"]
        assert chunks[1].text.startswith("Article 1 Definitions\n\n")
        assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (1, 1), (2, 2)]
        assert [c.ordinal for c in chunks] == [0, 1, 2]

    def test_sub_clauses_share_a_chunk_until_it_is_full(self):
        blocks = [heading("Article 4 Payment"), heading("4.1 Due date", level=3), para(40),
                  heading("4.2 Interest", level=3), para(40)]

        chunks = build_chunks(blocks, max_words=250, short_document_words=50, separator=SEP)

        assert len(chunks) == 1
        assert chunks[0].section_path == "Article 4 Payment › 4.1 Due date"

    def test_chunks_respect_the_word_budget_without_overlap(self):
        blocks = [heading("Article 1 Terms")] + [para(100, word=f"w{i}") for i in range(5)]

        chunks = build_chunks(blocks, max_words=250, short_document_words=100, separator=SEP)

        assert all(c.word_count <= 250 for c in chunks)
        joined = " ".join(c.text for c in chunks).split()
        assert sum(1 for w in joined if w.startswith("w")) == sum(99 for _ in range(5))

    def test_long_paragraph_splits_at_sentence_ends(self):
        sentence = "The Tenant shall keep the premises in good repair."
        blocks = [Block(kind=BlockKind.PARAGRAPH, text=" ".join([sentence] * 60), page_no=4)]

        chunks = build_chunks(blocks, max_words=100, short_document_words=50, separator=SEP)

        assert len(chunks) > 1
        assert all(c.text.endswith("repair.") for c in chunks)
        assert all(c.word_count <= 100 for c in chunks)

    def test_long_table_repeats_its_header_in_every_row_group(self):
        rows = "\n".join(f"| 20{i:02d} | EUR {i},000.00 | paid on time |" for i in range(80))
        table = Block(kind=BlockKind.TABLE, text=f"| Year | Rent | Note |\n| --- | --- | --- |\n{rows}", page_no=5)

        chunks = build_chunks([table], max_words=120, short_document_words=50, separator=SEP)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.startswith("| Year | Rent | Note |\n| --- | --- | --- |\n| 20")
        body_rows = [line for c in chunks for line in c.text.split("\n")[2:]]
        assert body_rows == rows.split("\n")

    def test_trailing_heading_joins_the_previous_chunk(self):
        blocks = [heading("Article 1 Terms"), para(300), heading("Signatures", level=1)]

        chunks = build_chunks(blocks, max_words=250, short_document_words=100, separator=SEP)

        assert chunks[-1].text.endswith("Signatures")
        assert all(any(not line.startswith("Signatures") for line in c.text.split("\n\n")) for c in chunks)

    def test_unpaged_document_has_no_pages(self):
        [chunk] = build_chunks([Block(kind=BlockKind.PARAGRAPH, text="Memo text.")], 250, 400, SEP)

        assert (chunk.page_start, chunk.page_end, chunk.section_path) == (None, None, None)

    def test_no_blocks_no_chunks(self):
        assert build_chunks([], 250, 400, SEP) == []


class TestShortMultiPageDocuments:
    def test_short_document_over_several_pages_is_still_split_by_clause(self):
        blocks = [heading("LEASE", level=1), para(40), heading("Article 7 Termination", page=2), para(40, page=2),
                  heading("Article 9 Rent", page=3), para(20, page=3)]

        chunks = build_chunks(blocks, max_words=250, short_document_words=400, separator=SEP)

        assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]
