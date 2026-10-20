"""A fake IEngineHost and engine: the API under test without torch, a GPU, a database or weights."""

import contextlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional

from web_deploy.api.engine_port import IEngineHost, LiveGate

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def chunk(text="The notice period is 30 days.", path: Optional[str] = None, anchor="time", **extra) -> NS:
    fields = dict(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), raw_text=text, text_payload="ctx " + text,
                  speaker_labels=["SPEAKER_00"], start_sec=12.0, end_sec=20.0, relevance_score=0.8,
                  rerank_score=0.9, physical_path=path, extracted_file_id=None, anchor_kind=NS(value=anchor),
                  page_start=None, page_end=None, section_path=None)
    fields.update(extra)
    return NS(**fields)


def citation(c: NS, number: int) -> NS:
    return NS(doc_id=f"Doc {number}", chunk_id=c.chunk_id, source_id=c.source_id, start_sec=c.start_sec,
              end_sec=c.end_sec, physical_path=c.physical_path, extracted_file_id=c.extracted_file_id,
              anchor_kind=c.anchor_kind, page_start=c.page_start, page_end=c.page_end, section_path=c.section_path)


class FakeEngine:
    """Records what it was asked, by whom and in which language."""

    def __init__(self, host: "FakeHost") -> None:
        self.host = host
        self.calls: List[tuple] = []
        self.chunks: Dict[uuid.UUID, NS] = {}
        self.reports: Dict[uuid.UUID, Any] = {}
        self.cancelled: List[uuid.UUID] = []
        self.gate: Optional[threading.Event] = None

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs, self.host.current_language()))
        if self.gate is not None:
            self.gate.wait(5)

    def add_chunk(self, c: NS) -> NS:
        self.chunks[c.chunk_id] = c
        return c

    def get_chunks(self, ids):
        return [self.chunks[i] for i in dict.fromkeys(ids) if i in self.chunks]

    def chat(self, query, session_id=None, owner_id=None):
        self._record("chat", query, session_id=session_id, owner_id=owner_id)
        c = self.add_chunk(chunk())
        return NS(generated_answer="Thirty days [Doc 1].", citations=[citation(c, 1)], inference_time_ms=420)

    def chat_reasoned(self, session_id, query, owner_id=None):
        self._record("chat_reasoned", query, session_id=session_id, owner_id=owner_id)
        return NS(qa_response=None, clarification_question="Which contract?", iteration_trace=[], total_iterations=1,
                  critique_unresolved=False)

    def search_standard(self, query, owner_id=None):
        self._record("search_standard", query, owner_id=owner_id)
        return NS(query=query.prompt, execution_time_ms=55, results=[self.add_chunk(chunk())])

    def search_verified(self, query, owner_id=None):
        self._record("search_verified", query, owner_id=owner_id)
        return NS(query=query.prompt, execution_time_ms=99, results=[])

    def list_conversations(self, limit, owner_id):
        self._record("list_conversations", limit, owner_id=owner_id)
        return [NS(session_id=uuid.uuid4(), title="Q", started_at=NOW, last_at=NOW, turn_count=2)]

    def get_conversation(self, session_id, owner_id):
        self._record("get_conversation", session_id, owner_id=owner_id)
        return []

    def delete_conversation(self, session_id, owner_id):
        self._record("delete_conversation", session_id, owner_id=owner_id)
        return 3

    def list_searches(self, limit, owner_id):
        self._record("list_searches", limit, owner_id=owner_id)
        return []

    def get_search(self, record_id, owner_id):
        self._record("get_search", record_id, owner_id=owner_id)
        return None

    def delete_search(self, record_id, owner_id):
        return True

    def propose_research_plan(self, topic, session_id=None, owner_id=None):
        self._record("propose_research_plan", topic, owner_id=owner_id)
        plan = self.host.plan(topic)
        return NS(report_id=plan.report_id, plan=plan, corpus=NS(indexed_source_count=4, indexed_chunk_count=90),
                  expected_job_count=12, estimated_minutes=9)

    def get_report(self, report_id, owner_id=None):
        report = self.reports.get(report_id)
        if report is None or (owner_id is not None and report.owner_id != owner_id):
            return None
        return report

    def rename_research_report(self, report_id, title, owner_id=None):
        self._record("rename_research_report", report_id, title=title, owner_id=owner_id)
        report = self.get_report(report_id, owner_id)
        if report is None:
            return None
        report.title = title
        return report

    def list_reports(self, session_id, limit, owner_id):
        return [r for r in self.reports.values() if r.owner_id == owner_id]

    @staticmethod
    def plan_cost(plan):
        return 12, 9

    def execute_research_report(self, report_id, plan):
        self._record("execute_research_report", report_id, plan=plan)
        return self.host.finished_report(self.reports[report_id])

    def execute_iterative_research_report(self, report_id, plan):
        self._record("execute_iterative_research_report", report_id, plan=plan)
        return self.host.finished_report(self.reports[report_id])

    def resume_research_report(self, report_id):
        self._record("resume_research_report", report_id)
        return self.host.finished_report(self.reports[report_id])

    def cancel_research_report(self, report_id):
        self.cancelled.append(report_id)
        return True

    SOURCE_ID = uuid.UUID("5a1d0c3e-0000-4000-8000-000000000001")

    def list_sources(self):
        return [NS(id=self.SOURCE_ID, name="deposition.mp4", source_type=NS(value="video"), language="en",
                   file=NS(file_size_bytes=1234), created_at=NOW)]

    def list_source_profiles(self):
        return [NS(source_id=self.SOURCE_ID, profile=NS(title="Deposition of J. Smith", topic="The lease dispute.",
                                                        origin=NS(value="generated")))]

    def edit_source_profile(self, source_id, title, topic=""):
        self._record("edit_source_profile", source_id, title=title, topic=topic)
        return NS(source_id=source_id, title=title, topic=topic, origin=NS(value="edited"))

    def busy_snapshot(self):
        return {"inflight": 0, "reports_running": 0}


class Plan(NS):
    def sub_questions_for(self, key):
        return [q for q in self.sub_questions if q.section_key == key]

    def model_copy(self, update):
        return Plan(**{**self.__dict__, **update})


class Section(NS):
    def model_copy(self, update):
        return Section(**{**self.__dict__, **update})


class FakeHost(IEngineHost):
    def __init__(self, root: Path) -> None:
        self._root = root
        self._ready = True
        self.engines: Dict[str, FakeEngine] = {}
        self._local = threading.local()
        self.gate = LiveGate()
        self.released = 0
        self.warmed = 0

    def start(self):
        self._ready = True

    def stop(self):
        self._ready = False

    @property
    def ready(self):
        return self._ready

    def busy(self):
        return {"inflight": 0, "reports_running": 0}

    @property
    def paused(self):
        return self.gate.paused

    def live(self):
        return self.gate.live()

    def pause(self, poll_seconds=0.01):
        if not self.gate.paused:
            self.gate.close_and_drain(poll_seconds)
            self.released += 1

    def resume(self):
        if self.gate.paused:
            self.warmed += 1
            self.gate.reopen()

    def engine(self, firm):
        return self.engines.setdefault(firm, FakeEngine(self))

    def firm_dir(self, firm):
        return self._root / firm

    def languages(self):
        return [{"code": "en", "name": "English"}, {"code": "sr", "name": "Srpski"}]

    def current_language(self):
        return getattr(self._local, "language", None)

    @contextlib.contextmanager
    def language_scope(self, code):
        previous = self.current_language()
        self._local.language = code
        try:
            yield code
        finally:
            self._local.language = previous

    def search_query(self, prompt, limit):
        return NS(prompt=prompt, limit=limit)

    def normalize_citation_tags(self, text, valid_numbers):
        return text

    def format_location(self, kind, page_start, page_end, section_path):
        return f"p. {page_start}" if page_start else ""

    def render_report(self, report):
        return f"# {report.topic}\n\nFinding [Doc 1](izvoria://doc/1)."

    def display_math(self, markdown):
        return markdown.replace("$E$", "$$E$$")

    def report_pdf(self, report):
        return b"%PDF-1.4 " + report.topic.encode()

    def outline_section(self, **fields):
        return Section(**fields)

    def sub_question(self, **fields):
        return NS(**fields)

    # ---------- report fixtures ----------

    @staticmethod
    def plan(topic):
        report_id = uuid.uuid4()
        return Plan(report_id=report_id, topic=topic, title="Termination Rights", scope_note="All of it.",
                    sections=[Section(key="s1", title="Background", brief="b", order_index=0)],
                    sub_questions=[NS(text="What happened?", section_key="s1", order_index=0)])

    def stored_report(self, engine: FakeEngine, owner_id: str, language: str = "sr", status: str = "awaiting_approval"):
        plan = self.plan("Termination clauses")
        report = NS(id=plan.report_id, topic=plan.topic, title=plan.title, plan=plan, status=NS(value=status),
                    owner_id=owner_id,
                    language=language, corpus=NS(indexed_source_count=4, indexed_chunk_count=90),
                    indexed_source_count=4)
        engine.reports[report.id] = report
        return report

    @staticmethod
    def finished_report(stored):
        c = chunk()
        return NS(id=stored.id, topic=stored.topic, title=stored.title, status=NS(value="completed"),
                  mode=NS(value="deep"),
                  language=stored.language, cited_doc_numbers=[1],
                  evidence=[NS(doc_number=1, chunk_id=c.chunk_id, source_id=c.source_id, start_sec=0.0, end_sec=1.0,
                               digest="digest", speaker_labels=[], anchor_kind=NS(value="time"), page_start=None,
                               page_end=None, section_path=None)],
                  written_sections=[1, 2], uncovered_sections=[], rounds_used=1, indexed_source_count=4,
                  swept_source_count=4)


def fake_queue():
    from web_deploy.ingest_worker.queue import IngestQueue
    return IngestQueue.connect("sqlite://")

