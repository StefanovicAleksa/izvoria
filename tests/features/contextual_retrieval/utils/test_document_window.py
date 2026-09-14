# Path: tests/features/contextual_retrieval/utils/test_document_window.py

import pytest

from app.features.contextual_retrieval.utils.document_window import build_document_sections, assign_section


class TestBuildDocumentSections:
    def test_fits_in_one_section(self):
        assert build_document_sections([10, 10, 10], budget_tokens=100) == [(0, 3)]

    def test_overlapping_sections_cover_whole_document(self):
        sections = build_document_sections([10] * 10, budget_tokens=40)

        assert sections == [(0, 4), (2, 6), (4, 8), (6, 10)]
        covered = set()
        for start, end in sections:
            assert sum([10] * (end - start)) <= 40
            covered.update(range(start, end))
        assert covered == set(range(10))

    def test_oversized_sentence_still_gets_its_own_section(self):
        sections = build_document_sections([5, 500, 5], budget_tokens=50)

        assert sections[0] == (0, 1)
        assert (1, 2) in sections
        assert sections[-1][1] == 3

    def test_always_progresses(self):
        sections = build_document_sections([1] * 7, budget_tokens=1)
        assert sections == [(i, i + 1) for i in range(7)]

    def test_empty_and_invalid_budget(self):
        assert build_document_sections([], budget_tokens=10) == []
        with pytest.raises(ValueError):
            build_document_sections([1], budget_tokens=0)


class TestAssignSection:
    def test_prefers_section_with_most_margin(self):
        sections = [(0, 4), (2, 6), (4, 8)]
        # sentence 4 is at the edge of (2,6)'s end and (4,8)'s start; (2,6) margin=min(2,1)=1, (4,8) margin=0
        assert assign_section(sections, 4, 4) == 1

    def test_falls_back_to_largest_overlap_when_nothing_contains_chunk(self):
        sections = [(0, 3), (2, 5)]
        assert assign_section(sections, 1, 4) in (0, 1)
        assert assign_section([(0, 2), (1, 4)], 1, 4) == 1

    def test_empty_sections_raise(self):
        with pytest.raises(ValueError):
            assign_section([], 0, 0)
