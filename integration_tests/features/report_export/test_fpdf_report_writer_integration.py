# Path: integration_tests/features/report_export/test_fpdf_report_writer_integration.py

"""
The PDF export end to end: a report with everything a real one carries (formulas inline and on their own line, a
table, adjacent citations, money, Serbian text, document and recording citations, an uncovered section, and a
formula that cannot be typeset), laid out with the real fonts and read back with pdfium.
"""

import uuid
from datetime import datetime

import pytest

from app.core.anchors.domain.enums import AnchorKind
from app.features.deep_research_report.domain.entities import (
    EvidenceEntry, OutlineSection, Report, ReportSection, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import ReportStatus, SectionStatus
from app.features.report_export.service.report_export_api import ReportExportAPI

pytestmark = pytest.mark.integration

pypdfium2 = pytest.importorskip("pypdfium2")
pypdfium2_raw = pytest.importorskip("pypdfium2.raw")

REPORT_ID = uuid.UUID(int=42)
BOOK = uuid.UUID(int=1)
LECTURE = uuid.UUID(int=2)

BODY = """The price elasticity of demand ($E_p$) is calculated as $(\\Delta Q / Q) / (\\Delta P / P)$ [Doc 1][Doc 2].
Its magnitude, $|E_p|$, decides whether demand is elastic [Doc 1].

$$\\frac{\\Delta Q}{Q} \\times \\frac{P}{\\Delta P}$$

### Determinants

- **Substitutes:** a rise in one raises demand for the other [Doc 2].
- **Complements:** the opposite [Doc 3].

| Good | Elasticity |
|---|---|
| Gas | $E_p = -0.2$ [Doc 3] |

A ceiling cut the price from $6.40 to $3.00 per mcf, and the average price $\\bar{P}$ fell [Doc 3].
A broken formula $\\frac{1}{$ is printed as written. Ponuda i tražnja: čćđšž [Doc 1].
"""


def _entry(number, source, name, kind, **fields):
    return EvidenceEntry(doc_number=number, chunk_id=uuid.uuid4(), source_id=source, source_name=name,
                         start_sec=fields.pop("start", 0.0), end_sec=fields.pop("end", 0.0), anchor_kind=kind,
                         **fields)


def _report(paragraphs: int = 1, title: str = "Supply, Demand and Price Elasticity") -> Report:
    sections = [OutlineSection(key="s1", title="Elasticity", order_index=0),
                OutlineSection(key="s2", title="Labour markets", order_index=1)]
    return Report(
        id=REPORT_ID,
        topic="create a report on microeconomics and the math behind it",
        status=ReportStatus.COMPLETED,
        plan=ResearchPlan(report_id=REPORT_ID, topic="create a report on microeconomics and the math behind it",
                          title=title, sections=sections),
        sections=[ReportSection(key="s1", title="Elasticity", order_index=0, status=SectionStatus.WRITTEN,
                                markdown_body="\n\n".join([BODY] * paragraphs), cited_doc_numbers=[1, 2, 3]),
                  ReportSection(key="s2", title="Labour markets", order_index=1,
                                status=SectionStatus.SKIPPED_NO_EVIDENCE)],
        evidence=[
            _entry(1, BOOK, "Microeconomics, 9th Edition.pdf", AnchorKind.PAGE, page_start=46, page_end=47,
                   section_path="PART 1 Introduction › 2.4 Elasticities of Supply and Demand",
                   digest="| | ΔQ/Q | | The price elasticity of demand measures the sensitivity of quantity demanded."),
            _entry(2, BOOK, "Microeconomics, 9th Edition.pdf", AnchorKind.PAGE, page_start=48, page_end=48,
                   digest="Substitutes and complements."),
            _entry(3, LECTURE, "UNIT 3 – Central Bank Execution.mp4", AnchorKind.TIME, start=754.0, end=802.0,
                   speaker_labels=["speaker_0", "speaker_1"],
                   digest="[speaker_0 @ 00:12:34]: The ceiling was three dollars."),
        ],
        executive_summary="Demand responds to price as $E_p$ measures [Doc 1].",
        indexed_source_count=2, swept_source_count=2, language="en",
        completed_at=datetime(2026, 10, 7, 16, 40),
    )


def _text(document) -> str:
    """Every page's words in one line: wrapped lines and cells are joined back."""
    return " ".join(" ".join(document[i].get_textpage().get_text_range().split()) for i in range(len(document)))


def _link_count(page) -> int:
    count = 0
    raw = page.raw
    for index in range(pypdfium2_raw.FPDFPage_GetAnnotCount(raw)):
        annotation = pypdfium2_raw.FPDFPage_GetAnnot(raw, index)
        if pypdfium2_raw.FPDFAnnot_GetSubtype(annotation) == pypdfium2_raw.FPDF_ANNOT_LINK:
            count += 1
        pypdfium2_raw.FPDFPage_CloseAnnot(annotation)
    return count


class TestPdfExport:
    def test_the_report_reads_back_with_its_title_text_and_references(self):
        document = pypdfium2.PdfDocument(ReportExportAPI().pdf(_report()))
        text = _text(document)

        assert document.get_metadata_dict()["Title"] == "Supply, Demand and Price Elasticity"
        assert "Supply, Demand and Price Elasticity" in text
        assert "Request: create a report on microeconomics and the math behind it" in text
        assert "Coverage: swept all 2 indexed sources." in text
        assert "Executive summary" in text and "1. Elasticity" in text
        assert "Not covered by the available material" in text and "Labour markets" in text
        # Money stays money; Serbian letters survive; a formula that can't be set is printed as written
        assert "$6.40 to $3.00" in text
        assert "čćđšž" in text
        assert "$\\frac{1}{$" in text
        # Typeset formulas are drawings, not their LaTeX source
        assert "\\Delta" not in text and "\\bar" not in text
        # References: grouped by source, with location, speakers and the cited words (minus table rules and stamps)
        assert "Microeconomics, 9th Edition.pdf" in text and "UNIT 3 – Central Bank Execution.mp4" in text
        assert "pp. 46–47 · 2.4 Elasticities of Supply and Demand" in text
        assert "speaker_0, speaker_1" in text
        assert "“ΔQ/Q The price elasticity of demand" in text
        assert "“The ceiling was three dollars.”" in text
        assert "[speaker_0 @" not in text

    def test_every_page_is_numbered_out_of_the_total(self):
        document = pypdfium2.PdfDocument(ReportExportAPI().pdf(_report(paragraphs=8)))
        total = len(document)

        assert total > 2
        for index in range(total):
            assert f"Page {index + 1} of {total}" in document[index].get_textpage().get_text_range()

    def test_sections_are_bookmarked(self):
        document = pypdfium2.PdfDocument(ReportExportAPI().pdf(_report()))

        assert [item.get_title() for item in document.get_toc()] == [
            "Executive summary", "Elasticity", "Not covered by the available material", "References"]

    def test_every_citation_is_a_link_to_its_reference(self):
        document = pypdfium2.PdfDocument(ReportExportAPI().pdf(_report()))
        text = _text(document)

        tags = text.count("Doc ")
        links = sum(_link_count(document[i]) for i in range(len(document)))
        # One link per tag in the prose; the reference rows' own tags are targets, not links
        references = 3
        assert links == tags - references > 0

    def test_a_report_from_before_titles_is_named_by_its_request_and_states_no_request_line(self):
        document = pypdfium2.PdfDocument(ReportExportAPI().pdf(_report(title="")))
        text = _text(document)

        assert document.get_metadata_dict()["Title"] == "create a report on microeconomics and the math behind it"
        assert "Request:" not in text

    def test_a_serbian_report_is_set_in_its_own_language(self):
        report = _report().model_copy(update={"language": "sr"})
        text = _text(pypdfium2.PdfDocument(ReportExportAPI().pdf(report)))

        assert "ISTRAŽIVAČKI IZVEŠTAJ" in text
        assert "Strana 1 od" in text
        assert "Zahtev: create a report" in text
