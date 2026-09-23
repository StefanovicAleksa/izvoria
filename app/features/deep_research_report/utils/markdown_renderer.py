# Path: app/features/deep_research_report/utils/markdown_renderer.py

"""
Renders a finished report as Markdown.

The governing requirement: **the file must be auditable on its own**, with no database behind it. Every
`[Doc N]` in the body resolves in the references table to a named source, a timespan and speakers, because
the person who has to rely on this report may be reading it months later, on a different machine, from an
email attachment.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional

from app.core.language.service.language_api import get_language_pack
from app.core.language.utils.script import cyrillic_to_latin, fold_to_ascii
from app.features.deep_research_report.domain.entities import EvidenceEntry, Report
from app.features.deep_research_report.utils.evidence_bank import format_timestamp

# Emitted instead of a plain tag when the desktop renders the report, so a citation is clickable there
# while the on-disk file stays plain text.
_DOC_LINK_SCHEME = "onyx://doc/"

_DOC_TAG = re.compile(r"\[Doc (\d+)]")


def render(report: Report, link_citations: bool = False) -> str:
    """
    The whole report as Markdown. `link_citations` turns `[Doc 7]` into `[Doc 7](onyx://doc/7)` for the
    desktop's click-to-verify panel; the file written to disk uses plain tags.
    """
    entries = report.evidence_by_doc_number
    strings = get_language_pack().report
    parts: List[str] = [
        f"# {report.topic}",
        "",
        _provenance(report),
        "",
    ]

    if report.executive_summary.strip():
        parts += [f"## {strings.executive_summary}", "", _cite(report.executive_summary.strip(), link_citations), ""]

    for section in report.written_sections:
        parts += [f"## {section.title}", "", _cite(section.markdown_body.strip(), link_citations), ""]

    uncovered = report.uncovered_sections
    if uncovered:
        parts += [f"## {strings.not_covered_heading}", ""]
        parts += [strings.not_covered_item.format(title=section.title) for section in uncovered]
        parts.append("")

    parts += _references(report, entries)

    return "\n".join(parts).rstrip() + "\n"


def _provenance(report: Report) -> str:
    """
    The coverage claim, stated rather than implied.

    A report that looked at 50 of 214 indexed sources is still useful; a report that looked at 50 and reads
    as though it covered all 214 is dangerous. Likewise the scrubber setting: it changes what was allowed
    to reach the writer, so the reader is told which way it was set.
    """
    strings = get_language_pack().report
    generated = (report.completed_at or report.created_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    cited = len(report.cited_doc_numbers)
    sources_cited = len({e.source_id for e in report.evidence if e.doc_number in set(report.cited_doc_numbers)})
    mode = strings.mode_names.get(report.mode.value, report.mode.value)

    lines = [strings.generated.format(
        generated=generated, cited=cited, sources=sources_cited,
        excerpt_noun=strings.noun("excerpt", cited), source_noun=strings.noun("source", sources_cited),
    )]

    if report.swept_source_count and report.indexed_source_count:
        if report.swept_source_count < report.indexed_source_count:
            lines.append(strings.coverage_partial.format(
                swept=report.swept_source_count, indexed=report.indexed_source_count
            ))
        else:
            lines.append(strings.coverage_all.format(indexed=report.indexed_source_count))

    # How hard it looked matters to the reader: "not covered" after one round and after three are
    # different claims about the corpus.
    if report.rounds_used > 1:
        lines.append(strings.rounds_many.format(rounds=report.rounds_used, mode=mode))
    else:
        lines.append(strings.rounds_single.format(mode=mode))

    lines.append(strings.scrubber.format(
        state=strings.scrubber_enabled if report.scrub_enabled else strings.scrubber_disabled
    ))
    return "\n".join(lines)


def _cite(text: str, link_citations: bool) -> str:
    if not link_citations:
        return text
    return _DOC_TAG.sub(lambda m: f"[Doc {m.group(1)}]({_DOC_LINK_SCHEME}{m.group(1)})", text)


def _references(report: Report, entries: Dict[int, EvidenceEntry]) -> List[str]:
    """
    One table row per cited document. Only cited evidence is listed: the bank holds everything the sweep
    gathered, but a reference the report never points at is noise in a document meant to be checked.
    """
    strings = get_language_pack().report
    cited = report.cited_doc_numbers
    if not cited:
        return [f"## {strings.references_heading}", "", strings.no_sources_cited, ""]

    rows = [
        f"## {strings.references_heading}",
        "",
        *strings.references_table_header.splitlines(),
    ]
    for number in cited:
        entry = entries.get(number)
        if entry is None:
            # A cited number with no bank row should be impossible -- citations are validated against the
            # bank -- but a reference table that silently skipped one would hide the inconsistency.
            rows.append(f"| [Doc {number}] | {strings.unresolved_reference} | — | — |")
            continue
        speakers = ", ".join(entry.speaker_labels) if entry.speaker_labels else strings.no_speakers
        rows.append(
            f"| [Doc {number}] | {_escape_cell(entry.source_name)} | "
            f"{format_timestamp(entry.start_sec)}–{format_timestamp(entry.end_sec)} | {_escape_cell(speakers)} |"
        )

    rows.append("")
    return rows


def _escape_cell(text: str) -> str:
    """A pipe in a source filename would otherwise split the table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def slugify(topic: str, max_length: int = 60) -> Optional[str]:
    """
    A filesystem-safe ASCII stem for the report's filename; None when the topic has nothing usable in it. Cyrillic
    is transliterated and diacritics dropped first ("Лечење шећера" -> "lecenje-secera"), or a Serbian topic would
    have no usable letters at all.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", fold_to_ascii(cyrillic_to_latin(topic)).lower()).strip("-")
    if not slug:
        return None
    return slug[:max_length].rstrip("-")
