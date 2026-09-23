# Path: integration_tests/orchestration/test_research_scoping_phase_integration.py

import os
import time
import uuid

import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager as core_db_manager
from app.core.jobs.data.repositories import JobRepository
from app.core.jobs.domain.enums import JobStatus, JobType
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.features.deep_research_report.data.postgres_report_repo import PostgresReportRepository
from app.features.deep_research_report.data.report_file_store import ReportFileStore
from app.features.deep_research_report.data.report_llm_adapter import LlamaCppReportAdapter
from app.features.deep_research_report.domain.entities import CorpusDigest, IndexedSource
from app.features.deep_research_report.domain.enums import ReportStatus
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.service.phases.research_scoping_phase import ResearchScopingPhase

pytestmark = pytest.mark.integration

_TOPIC = "Chain of custody for the sealed evidence bag"

# Real excerpts, as the pipeline's grounding search would supply them. The point of showing these to
# the planner is that the outline is proposed against the corpus's own vocabulary rather than against
# the model's expectations of the topic, so the fixture has to read like real transcript.
_SAMPLES = [
    "SPEAKER_01: The evidence bag was sealed at the scene and initialled by me.",
    "SPEAKER_01: I logged it into the property room at nine forty that evening.",
    "SPEAKER_02: The property room log shows the seal intact when it was signed in.",
    "SPEAKER_02: Nobody accessed the locker between the tenth and the fourteenth.",
    "SPEAKER_01: The transfer to the lab was done by courier with a signed manifest.",
]


class _CountingLoader:
    """Counts real loads of the ~5.7 GB GGUF, so a residency claim can be asserted rather than assumed."""

    def __init__(self, inner):
        self._inner = inner
        self.load_count = 0
        self.unload_count = 0

    def load(self, manifest: ModelManifest):
        self.load_count += 1
        return self._inner.load(manifest)

    def unload(self, model) -> bool:
        self.unload_count += 1
        return self._inner.unload(model)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module", autouse=True)
def require_models():
    if not os.path.exists(settings.REPORT_AUTHORING_MODEL_PATH):
        pytest.skip(
            "Research scoping integration test skipped; missing weights: "
            f"{settings.REPORT_AUTHORING_MODEL_PATH}"
        )


@pytest.fixture(scope="module", autouse=True)
def setup_schema():
    Base.metadata.create_all(bind=core_db_manager._engine_wrapper._engine)  # noqa
    yield


@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_PATH", str(tmp_path / "reports"))
    return tmp_path / "reports"


@pytest.fixture
def small_plan(monkeypatch):
    """A narrow outline: enough to prove the shape without paying for an eight-section plan."""
    monkeypatch.setattr(settings, "REPORT_MAX_SECTIONS", 2)
    monkeypatch.setattr(settings, "REPORT_MAX_SUB_QUESTIONS", 3)
    monkeypatch.setattr(settings, "REPORT_PLAN_MAX_TOKENS", 900)


def _corpus() -> CorpusDigest:
    return CorpusDigest(
        indexed_source_count=1,
        indexed_chunk_count=len(_SAMPLES),
        sources=[IndexedSource(
            source_id=uuid.uuid4(), name="Deposition — Ramirez", chunk_count=len(_SAMPLES),
            first_start_sec=0.0, last_end_sec=150.0,
        )],
        sample_digests=_SAMPLES,
    )


@pytest.fixture
def harness(db_manager, reports_dir):
    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    async_service = AsyncModelService(BackgroundModelWorker(orchestrator))
    job_manager = JobManager(JobRepository(db_manager))

    # One adapter, wrapped once, shared by the API and the phase -- exactly as the bootstrapper wires
    # it, so the counts below are real loads of the file rather than of a second instance.
    loader = _CountingLoader(LlamaCppReportAdapter())
    report_api = ReportAuthoringAPI(
        adapter=loader,
        orchestrator=orchestrator,
        repository=PostgresReportRepository(db_manager),
        file_store=ReportFileStore(),
    )
    phase = ResearchScopingPhase(job_manager, report_api, async_service, loader)

    yield phase, loader, report_api, db_manager

    async_service.release_all()
    async_service.shutdown()


class TestResearchScopingPhaseIntegration:
    """
    The scoping phase against a real Qwen3.5-9B and a real Postgres. Skips rather than fails when the
    weights are not on this machine.
    """

    def test_a_real_scope_persists_an_approvable_plan(self, harness, small_plan):
        phase, _loader, report_api, _db = harness

        result = phase.execute([(_TOPIC, _corpus(), None)])

        assert result.status == PhaseStatus.COMPLETED
        assert result.phase == PipelinePhase.RESEARCH_SCOPING
        job_result, report = result.items[0]
        assert job_result.success is True

        # The row is what the user approves against, so it must be readable back, not just returned.
        stored = report_api.get_report(report.id)
        assert stored is not None
        assert stored.status == ReportStatus.AWAITING_APPROVAL
        assert stored.plan.sections, "a plan with no sections cannot be approved or written"
        assert len(stored.plan.sections) <= settings.REPORT_MAX_SECTIONS
        assert len(stored.plan.sub_questions) <= settings.REPORT_MAX_SUB_QUESTIONS
        assert job_result.section_count == len(stored.plan.sections)
        assert job_result.sub_question_count == len(stored.plan.sub_questions)

    def test_every_section_key_is_unique_and_every_sub_question_points_at_one(self, harness, small_plan):
        """
        `key` is the link the whole run hangs on: evidence is assigned to it, sections are written per
        key, and a resume finds its pending work by it. A duplicate or dangling key corrupts all three.
        """
        phase, _loader, _api, _db = harness

        _job, report = phase.execute([(_TOPIC, _corpus(), None)]).items[0]

        keys = [section.key for section in report.plan.sections]
        assert len(keys) == len(set(keys))
        for question in report.plan.sub_questions:
            assert question.section_key in keys

    def test_the_scoping_call_is_recorded_as_a_completed_job(self, harness, small_plan, db_manager):
        """Progress reporting and the report view's step count both read these rows."""
        phase, _loader, _api, _db = harness

        phase.execute([(_TOPIC, _corpus(), None)])

        with db_manager.session() as session:
            row = session.execute(sa_text(
                "SELECT status FROM jobs WHERE job_type = :t ORDER BY created_at DESC LIMIT 1"
            ), {"t": JobType.RESEARCH_SCOPING.value}).first()

        assert row is not None, "the scoping call must run inside a tracked job"
        assert row[0] == JobStatus.COMPLETED.value

    def test_a_resolved_preload_covers_the_call_and_the_model_is_released_after(
            self, harness, small_plan
    ):
        """
        The phase owns the residency when it resolved a preload: one load, and nothing left resident
        afterwards -- the next stage's `acquire_model(wait_for_clear=True)` would otherwise park.
        """
        phase, loader, _api, _db = harness

        phase.preload_model()
        result = phase.execute([(_TOPIC, _corpus(), None)])

        assert result.status == PhaseStatus.COMPLETED
        assert loader.load_count == 1
        assert loader.unload_count == 1

    def test_discarding_an_unresolved_preload_leaves_nothing_resident(self, harness, small_plan):
        """
        The pipeline's `finally` path. A preload fired and then abandoned must end up released, or the
        model stays loaded with nothing able to free it and every later acquisition parks forever.

        The release happens on the worker thread, as a done-callback on the load, so this waits for the
        counts to settle rather than asserting immediately.

        The converse -- that the discard must NOT free a model somebody has since acquired -- is
        `test_discarding_a_preload_does_not_free_a_model_another_caller_is_using`.
        """
        phase, loader, _api, _db = harness

        phase.preload_model()
        phase.discard_preload()

        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if loader.load_count and loader.unload_count == loader.load_count:
                break
            time.sleep(0.5)

        assert loader.load_count == 1, "the abandoned preload should still have been loaded"
        assert loader.unload_count == 1, "an abandoned preload must not stay resident"
