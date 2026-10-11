"""
Engine results as JSON for the browser.

Adapted from the desktop app's engine_gateway mapping (frontend/izvoria_desktop/data/engine_gateway.py), which can't be
imported here: it is bound to Qt-free but desktop-only types and its i18n. The rules are the same: an answer's text
and its citation list agree on [Doc N] numbers, an excerpt is the verbatim `raw_text` (never the generated context),
and a document chunk is located by page/lines/section rather than by its 0.0 timestamps.

One difference: a file path never leaves the box. Evidence says only whether a source can be opened
(`has_source`); the browser asks for it by chunk id, and the API re-resolves the path inside the firm's vault.
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence

from web_deploy.api.engine_port import IEngineHost

logger = logging.getLogger(__name__)

_DOC_TAG = re.compile(r"\[Doc (\d+)\]")


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else None


def _anchor_value(item: Any) -> Optional[str]:
    kind = getattr(item, "anchor_kind", None)
    return getattr(kind, "value", kind)


def _is_document(item: Any) -> bool:
    return _anchor_value(item) not in (None, "time")


def _media_kind(path: Optional[str], extracted_file_id: Any, item: Any) -> str:
    """clip: the file is the cited passage; full_source: a whole recording to seek into; document / image."""
    if not path:
        return "unresolved"
    if _anchor_value(item) == "image":
        return "image"
    if _is_document(item):
        return "document"
    return "clip" if extracted_file_id else "full_source"


def _location(host: IEngineHost, item: Any) -> str:
    if not _is_document(item):
        return ""
    try:
        return host.format_location(_anchor_value(item), getattr(item, "page_start", None),
                                    getattr(item, "page_end", None), getattr(item, "section_path", None))
    except Exception as exc:
        logger.debug(f"Could not format a location: {exc}")
        return ""


def _anchor(item: Any) -> dict:
    return {
        "anchor_kind": _anchor_value(item) or "time",
        "page_start": getattr(item, "page_start", None),
        "page_end": getattr(item, "page_end", None),
    }


def doc_number(doc_id: Any) -> Optional[int]:
    digits = "".join(ch for ch in str(doc_id) if ch.isdigit())
    return int(digits) if digits else None


# ---------- chunks, hits, evidence ----------

def chunk_evidence(host: IEngineHost, number: int, chunk: Any) -> dict:
    path = getattr(chunk, "physical_path", None)
    return {
        "doc_number": number,
        "chunk_id": str(chunk.chunk_id),
        "source_id": str(chunk.source_id),
        "start_sec": chunk.start_sec,
        "end_sec": chunk.end_sec,
        "excerpt": chunk.raw_text or chunk.text_payload,
        "speakers": list(chunk.speaker_labels or ()),
        "media_kind": _media_kind(path, getattr(chunk, "extracted_file_id", None), chunk),
        "has_source": bool(path),
        "location": _location(host, chunk),
        **_anchor(chunk),
    }


def search_hit(host: IEngineHost, rank: int, chunk: Any) -> dict:
    hit = chunk_evidence(host, rank, chunk)
    hit.pop("doc_number")
    hit.update(rank=rank, relevance_score=chunk.relevance_score, rerank_score=chunk.rerank_score)
    return hit


def search_outcome(host: IEngineHost, result: Any, mode: str) -> dict:
    return {
        "query": result.query,
        "mode": mode,
        "execution_time_ms": result.execution_time_ms,
        "hits": [search_hit(host, rank, chunk) for rank, chunk in enumerate(result.results, start=1)],
    }


def citations_evidence(host: IEngineHost, engine: Any, citations: Sequence[Any]) -> List[dict]:
    """Citations carry ids and a resolved path but no text; the excerpts come from one indexed read."""
    chunks: Dict[Any, Any] = {}
    if citations:
        try:
            chunks = {c.chunk_id: c for c in engine.get_chunks([c.chunk_id for c in citations])}
        except Exception as exc:
            logger.warning(f"Could not hydrate citation excerpts: {exc}")

    evidence = []
    for citation in citations:
        chunk = chunks.get(citation.chunk_id)
        number = doc_number(citation.doc_id)
        path = getattr(citation, "physical_path", None)
        evidence.append({
            "doc_number": number if number is not None else len(evidence) + 1,
            "chunk_id": str(citation.chunk_id),
            "source_id": str(citation.source_id),
            "start_sec": citation.start_sec,
            "end_sec": citation.end_sec,
            "excerpt": (getattr(chunk, "raw_text", "") or "") if chunk is not None else "",
            "speakers": list(getattr(chunk, "speaker_labels", ()) or ()),
            "media_kind": _media_kind(path, getattr(citation, "extracted_file_id", None), citation),
            "has_source": bool(path),
            "location": _location(host, citation),
            **_anchor(citation),
        })
    return evidence


# ---------- chat ----------

def _normalized_answer(host: IEngineHost, response: Any) -> str:
    valid = {n for n in (doc_number(c.doc_id) for c in response.citations) if n is not None}
    try:
        return host.normalize_citation_tags(response.generated_answer, valid)
    except Exception as exc:
        logger.warning(f"Citation normalization failed, showing raw answer text: {exc}")
        return response.generated_answer


def chat_message(host: IEngineHost, engine: Any, response: Any) -> dict:
    return {
        "role": "assistant",
        "text": _normalized_answer(host, response),
        "outcome": "answered",
        "evidence": citations_evidence(host, engine, response.citations),
        "reasoning": [],
        "latency_ms": response.inference_time_ms,
    }


def reasoned_message(host: IEngineHost, engine: Any, result: Any) -> dict:
    reasoning = [
        {"iteration": s.iteration, "decision": s.decision, "interim_analysis": s.interim_analysis,
         "refined_query": s.refined_query, "chunks_added": s.chunks_added}
        for s in result.iteration_trace
    ]
    if result.qa_response is None:
        return {"role": "assistant", "text": result.clarification_question or "Could you narrow that question?",
                "outcome": "needs_clarification", "evidence": [], "reasoning": reasoning,
                "total_iterations": result.total_iterations}
    message = chat_message(host, engine, result.qa_response)
    message.update(reasoning=reasoning, critique_unresolved=result.critique_unresolved,
                   total_iterations=result.total_iterations)
    return message


def conversation(host: IEngineHost, engine: Any, turns: Sequence[Any]) -> List[dict]:
    """
    The conversation as it was shown. [Doc N] numbers are conversation-wide, so one map from number to chunk covers
    every tag in every answer. A Reasoned Answer is stored without tags; its cards are the evidence it showed first.
    """
    number_to_chunk: Dict[int, Any] = {}
    for turn in turns:
        for offset, chunk_id in enumerate(turn.evidence_chunk_ids):
            number_to_chunk[turn.first_doc_number + offset] = chunk_id
    chunks = {}
    if number_to_chunk:
        chunks = {c.chunk_id: c for c in engine.get_chunks(list(dict.fromkeys(number_to_chunk.values())))}

    messages = []
    for turn in turns:
        messages.append({"role": "user", "text": turn.question, "created_at": _iso(turn.created_at)})
        text = turn.answer_text
        try:
            text = host.normalize_citation_tags(text, set(number_to_chunk))
        except Exception as exc:
            logger.warning(f"Citation normalization failed for a stored turn: {exc}")
        cited = [int(n) for n in dict.fromkeys(_DOC_TAG.findall(text))]
        numbers = cited or [turn.first_doc_number + i for i in range(len(turn.evidence_chunk_ids))]
        evidence = [chunk_evidence(host, n, chunks[number_to_chunk[n]])
                    for n in numbers if n in number_to_chunk and number_to_chunk[n] in chunks]
        messages.append({
            "role": "assistant", "text": text, "outcome": "answered", "evidence": evidence, "reasoning": [],
            "mode": getattr(turn.mode, "value", turn.mode), "created_at": _iso(turn.created_at),
            "missing_evidence": len([n for n in numbers if n in number_to_chunk]) - len(evidence),
        })
    return messages


def conversation_ref(summary: Any) -> dict:
    return {"session_id": str(summary.session_id), "title": summary.title,
            "started_at": _iso(summary.started_at), "last_at": _iso(summary.last_at),
            "turn_count": summary.turn_count}


def search_record_ref(record: Any) -> dict:
    return {"record_id": str(record.id), "query": record.query, "mode": record.mode.value,
            "hit_count": record.hit_count, "created_at": _iso(record.created_at)}


def search_record(host: IEngineHost, engine: Any, record: Any) -> dict:
    """A past search as shown then: original rank, original scores."""
    chunks = {c.chunk_id: c for c in engine.get_chunks(list(record.chunk_ids))} if record.chunk_ids else {}
    hits = []
    for rank, (chunk_id, score) in enumerate(zip(record.chunk_ids, record.scores), start=1):
        chunk = chunks.get(chunk_id)
        if chunk is not None:
            hit = search_hit(host, rank, chunk)
            hit["relevance_score"] = score
            hits.append(hit)
    return {"query": record.query, "mode": record.mode.value, "hits": hits,
            "execution_time_ms": int(record.execution_time_ms), "saved_at": _iso(record.created_at),
            "missing_hits": len(record.chunk_ids) - len(hits)}


# ---------- reports ----------

def plan_draft(report_id: Any, plan: Any, **counts: int) -> dict:
    return {
        "report_id": str(report_id),
        "topic": plan.topic,
        "title": plan.title,
        "scope_note": plan.scope_note,
        "sections": [
            {"key": s.key, "title": s.title, "brief": s.brief,
             "questions": [q.text for q in plan.sub_questions_for(s.key)]}
            for s in plan.sections
        ],
        **counts,
    }


def proposal(result: Any) -> dict:
    return plan_draft(result.report_id, result.plan,
                      indexed_source_count=result.corpus.indexed_source_count,
                      indexed_chunk_count=result.corpus.indexed_chunk_count,
                      expected_job_count=result.expected_job_count,
                      estimated_minutes=result.estimated_minutes)


# As the engine's clean_title (app/features/deep_research_report/utils/titles.py), which this layer can't import
MAX_TITLE_CHARS = 120


def download_name(report: Any, suffix: str) -> str:
    """`Supply and Demand.pdf`: the report's title (or, before titles, its request), safe as a filename."""
    name = re.sub(r"[^\w\- ]", "_", report.title or report.topic)[:80].strip()
    return f"{name or 'report'}{suffix}"


class InvalidPlan(ValueError):
    pass


def apply_edits(host: IEngineHost, stored_plan: Any, draft: dict) -> Any:
    """
    The engine's plan rebuilt from the user's edited outline. Section keys are carried through (evidence is
    re-attached by key across revisions); a deleted section takes its sub-questions with it.
    """
    edited_sections = draft.get("sections")
    if not isinstance(edited_sections, list) or not edited_sections:
        raise InvalidPlan("A report needs at least one section.")
    by_key = {s.key: s for s in stored_plan.sections}
    sections, questions = [], []
    for order_index, edited in enumerate(edited_sections[:40]):
        if not isinstance(edited, dict):
            raise InvalidPlan("Each section must be an object.")
        title = str(edited.get("title") or "").strip()[:300]
        if not title:
            raise InvalidPlan("Every section needs a title.")
        brief = str(edited.get("brief") or "").strip()[:2000]
        key = str(edited.get("key") or "")[:64]
        base = by_key.get(key)
        sections.append(
            base.model_copy(update={"title": title, "brief": brief, "order_index": order_index})
            if base is not None else
            host.outline_section(key=key or f"u{order_index + 1}", title=title, brief=brief, order_index=order_index)
        )
        for text in (edited.get("questions") or [])[:20]:
            text = str(text).strip()[:500]
            if text:
                questions.append(host.sub_question(text=text, section_key=sections[-1].key,
                                                   order_index=len(questions)))
    update = {"sections": sections, "sub_questions": questions}
    if "title" in draft:
        # Tidied again by the engine when the plan is approved; "" keeps the request as the report's name
        update["title"] = " ".join(str(draft.get("title") or "").split())[:MAX_TITLE_CHARS]
    return stored_plan.model_copy(update=update)


def report_ref(summary: Any) -> dict:
    return {"report_id": str(summary.id), "topic": summary.topic, "title": summary.title,
            "status": summary.status.value,
            "created_at": _iso(summary.created_at), "completed_at": _iso(summary.completed_at),
            "section_count": summary.section_count, "evidence_count": summary.evidence_count}


def report_document(host: IEngineHost, engine: Any, report: Any) -> dict:
    """The rendered report with clickable citations, and the evidence behind every number it cites."""
    evidence = []
    try:
        cited = set(report.cited_doc_numbers)
        entries = [e for e in report.evidence if e.doc_number in cited]
        chunks = {c.chunk_id: c for c in engine.get_chunks([e.chunk_id for e in entries])} if entries else {}
        for entry in entries:
            chunk = chunks.get(entry.chunk_id)
            path = getattr(chunk, "physical_path", None) if chunk is not None else None
            evidence.append({
                "doc_number": entry.doc_number,
                "chunk_id": str(entry.chunk_id),
                "source_id": str(entry.source_id),
                "start_sec": entry.start_sec,
                "end_sec": entry.end_sec,
                # A re-ingest can delete the chunk; the bank's stored digest still names the source
                "excerpt": (chunk.raw_text if chunk is not None else entry.digest) or "",
                "speakers": list(entry.speaker_labels),
                "media_kind": _media_kind(path, getattr(chunk, "extracted_file_id", None) if chunk else None, entry),
                "has_source": bool(path),
                "location": _location(host, entry),
                **_anchor(entry),
            })
    except Exception as exc:
        logger.warning(f"Could not hydrate report evidence: {exc}")

    return {
        "report_id": str(report.id),
        "topic": report.topic,
        "title": report.title,
        "markdown": host.display_math(host.render_report(report)),
        "status": report.status.value,
        "mode": report.mode.value,
        "language": report.language,
        "evidence": evidence,
        "section_count": len(report.written_sections),
        "uncovered_count": len(report.uncovered_sections),
        "rounds_used": report.rounds_used,
        "indexed_source_count": report.indexed_source_count,
        "swept_source_count": report.swept_source_count,
    }


# ---------- library ----------

def source_ref(source: Any) -> dict:
    file = getattr(source, "file", None)
    return {
        "source_id": str(source.id),
        "name": source.name,
        "source_type": getattr(source.source_type, "value", source.source_type),
        "language": getattr(source, "language", None),
        "size_bytes": getattr(file, "file_size_bytes", None) if file else None,
        "created_at": _iso(getattr(source, "created_at", None)),
    }


def parse_uuid(raw: Any) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except (TypeError, ValueError):
        raise ValueError(f"'{raw}' is not a valid id.")
