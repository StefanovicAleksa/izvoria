# Path: tests/features/deep_research_report/utils/test_markdown_renderer.py

import re
import uuid
from datetime import datetime

import pytest

from app.features.deep_research_report.domain.entities import (
    EvidenceEntry, Report, ReportSection, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import ReportStatus, SectionStatus
from app.features.deep_research_report.utils.markdown_renderer import render, slugify


def _entry(number: int, source_id=None, name="Deposition — Ramirez 2024-03-11") -> EvidenceEntry:
    return EvidenceEntry(
        doc_number=number,
        chunk_id=uuid.uuid4(),
        source_id=source_id or uuid.uuid4(),
        source_name=name,
        start_sec=751,
        end_sec=784,
        speaker_labels=["SPEAKER_01"],
        digest="The bag was sealed.",
    )


def _report(
        sections=None,
        evidence=None,
        summary="Custody was maintained [Doc 1].",
        indexed=214,
        swept=50,
        scrub=False,
) -> Report:
    report_id = uuid.uuid4()
    return Report(
        id=report_id,
        topic="Chain of custody",
        status=ReportStatus.COMPLETED,
        plan=ResearchPlan(report_id=report_id, topic="Chain of custody"),
        sections=sections if sections is not None else [
            ReportSection(key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
                          markdown_body="The bag was sealed [Doc 1][Doc 3].", cited_doc_numbers=[1, 3]),
        ],
        evidence=evidence if evidence is not None else [_entry(n) for n in (1, 2, 3)],
        executive_summary=summary,
        indexed_source_count=indexed,
        swept_source_count=swept,
        scrub_enabled=scrub,
        completed_at=datetime(2026, 9, 21, 14, 30),
    )


class TestRenderStructure:
    def test_renders_title_summary_sections_and_references(self):
        markdown = render(_report())

        assert markdown.startswith("# Chain of custody")
        assert "## Executive summary" in markdown
        assert "## Handling" in markdown
        assert "## References" in markdown

    def test_uncovered_sections_are_listed_as_not_covered_rather_than_omitted(self):
        """
        Absence is a finding. A section the corpus could not support is stated, not quietly dropped --
        otherwise the report reads as though the question was never asked.
        """
        sections = [
            ReportSection(key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
                          markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1]),
            ReportSection(key="s2", title="Storage", order_index=1,
                          status=SectionStatus.SKIPPED_NO_EVIDENCE),
        ]

        markdown = render(_report(sections=sections))

        assert "## Not covered by the available material" in markdown
        assert "**Storage**" in markdown
        assert "## Storage" not in markdown

    def test_no_uncovered_heading_when_everything_was_written(self):
        assert "Not covered by the available material" not in render(_report())

    def test_summary_is_omitted_when_empty(self):
        assert "## Executive summary" not in render(_report(summary=""))


class TestReferences:
    def test_every_cited_doc_number_resolves_in_the_references_table(self):
        """
        The governing requirement: the file must be auditable with no database behind it.
        """
        markdown = render(_report())

        cited = set(re.findall(r"\[Doc (\d+)]", markdown.split("## References")[0]))
        table = markdown.split("## References")[1]
        for number in cited:
            assert f"| [Doc {number}] |" in table

    def test_gathered_but_uncited_evidence_is_not_listed(self):
        """The bank holds everything the sweep found; a reference the report never points at is noise."""
        markdown = render(_report())
        table = markdown.split("## References")[1]

        assert "| [Doc 1] |" in table
        assert "| [Doc 3] |" in table
        assert "| [Doc 2] |" not in table

    def test_a_pipe_in_a_source_name_is_escaped_so_the_table_survives(self):
        evidence = [_entry(1, name="Interview | Part 2")]
        sections = [ReportSection(key="s1", title="S", order_index=0, status=SectionStatus.WRITTEN,
                                  markdown_body="x [Doc 1].", cited_doc_numbers=[1])]

        table = render(_report(sections=sections, evidence=evidence, summary="")).split("## References")[1]

        assert r"Interview \| Part 2" in table

    def test_timespans_and_speakers_are_rendered(self):
        table = render(_report()).split("## References")[1]

        assert "00:12:31–00:13:04" in table
        assert "SPEAKER_01" in table

    def test_a_cited_number_with_no_bank_row_is_surfaced_not_silently_skipped(self):
        """An inconsistency the reader can see beats one the renderer hides."""
        sections = [ReportSection(key="s1", title="S", order_index=0, status=SectionStatus.WRITTEN,
                                  markdown_body="x [Doc 9].", cited_doc_numbers=[9])]

        table = render(_report(sections=sections, evidence=[], summary="")).split("## References")[1]

        assert "| [Doc 9] | _unresolved_ |" in table

    def test_a_report_citing_nothing_says_so(self):
        sections = [ReportSection(key="s1", title="S", order_index=0,
                                  status=SectionStatus.SKIPPED_NO_EVIDENCE)]

        markdown = render(_report(sections=sections, summary=""))

        assert "_No sources were cited._" in markdown


class TestProvenanceLine:
    def test_partial_coverage_is_stated_explicitly(self):
        """
        A report that swept 50 of 214 sources and reads as though it covered all 214 is dangerous in a
        product whose entire pitch is traceability.
        """
        markdown = render(_report(indexed=214, swept=50))

        assert "Coverage: swept 50 of 214 indexed sources." in markdown

    def test_full_coverage_is_stated_too(self):
        markdown = render(_report(indexed=12, swept=12))

        assert "Coverage: swept all 12 indexed sources." in markdown

    @pytest.mark.parametrize("scrub,expected", [(True, "enabled"), (False, "disabled")])
    def test_the_scrubber_setting_is_reported(self, scrub, expected):
        assert f"Retrieval scrubber: {expected}." in render(_report(scrub=scrub))

    def test_counts_cited_excerpts_and_distinct_sources(self):
        source_id = uuid.uuid4()
        evidence = [_entry(1, source_id=source_id), _entry(3, source_id=source_id)]

        markdown = render(_report(evidence=evidence))

        assert "from 2 cited excerpts across 1 source." in markdown


class TestCitationLinking:
    def test_file_mode_keeps_plain_tags(self):
        markdown = render(_report())

        assert "[Doc 1][Doc 3]" in markdown
        assert "onyx://doc/" not in markdown

    def test_link_mode_makes_citations_clickable_for_the_desktop(self):
        markdown = render(_report(), link_citations=True)

        assert "[Doc 1](onyx://doc/1)" in markdown
        assert "[Doc 3](onyx://doc/3)" in markdown

    def test_link_mode_does_not_rewrite_the_references_table_headers(self):
        table = render(_report(), link_citations=True).split("## References")[1]

        assert "| [Doc 1](onyx://doc/1) |" in table or "| [Doc 1] |" in table


@pytest.mark.parametrize("topic,expected", [
    ("Chain of custody", "chain-of-custody"),
    ("Chain of custody: the Ramirez matter!", "chain-of-custody-the-ramirez-matter"),
    ("  Spaces   everywhere  ", "spaces-everywhere"),
    ("!!!", None),
    ("", None),
])
def test_slugify(topic, expected):
    assert slugify(topic) == expected


def test_slugify_truncates_without_a_trailing_dash():
    slug = slugify("a" * 40 + " " + "b" * 40, max_length=42)

    assert len(slug) <= 42
    assert not slug.endswith("-")
