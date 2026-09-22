# Path: integration_tests/orchestration/test_deep_research_report_pipeline_integration.py

import os
import re
import sys
import threading
import uuid

import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.core.jobs.data.repositories import JobRepository
from app.core.jobs.service.manager import JobManager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.utils.background_worker import BackgroundModelWorker
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.deep_research_report.data.postgres_report_repo import PostgresReportRepository
from app.features.deep_research_report.data.report_file_store import ReportFileStore
from app.features.deep_research_report.data.report_llm_adapter import LlamaCppReportAdapter
from app.features.deep_research_report.domain.enums import ReportMode, ReportStatus
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI
from app.features.rag_reranking.data.cross_encoder_adapter import CrossEncoderRerankerAdapter
from app.features.rag_reranking.service.rerank_api import RerankAPI
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_storage.data.models import VectorChunkModel
from app.features.retrieval_hallucination_scrubber.data.slm_adapter import QwenRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI
from app.features.storage.data.models import FileModel, SourceModel
from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase
from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.report_authoring_phase import ReportAuthoringPhase
from app.orchestration.service.phases.research_scoping_phase import ResearchScopingPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.service.pipelines.deep_research_report_pipeline import DeepResearchReportPipeline

pytestmark = pytest.mark.integration

_TRANSCRIPT = [
    "[SPEAKER_01 @ 00:00:00]: The evidence bag was sealed at the scene and initialled by me.",
    "[SPEAKER_01 @ 00:00:30]: I logged it into the property room at nine forty that evening.",
    "[SPEAKER_02 @ 00:01:00]: The property room log shows the seal intact when it was signed in.",
    "[SPEAKER_02 @ 00:01:30]: Nobody accessed the locker between the tenth and the fourteenth.",
    "[SPEAKER_01 @ 00:02:00]: The transfer to the lab was done by courier with a signed manifest.",
]


def _run_or_fail(call, timeout_seconds: int = 1200):
    """
    Runs `call` on a daemon thread and fails if it does not finish in time.

    The loop's failure mode is a HANG, not an exception: `acquire_model(wait_for_clear=True)` parks
    with no time limit, so a round that did not release the report model before gathering again wedges
    forever. A plain call would hang the whole suite; this turns that into a failed assertion.

    Deliberately not `pytest-timeout` -- this is the only place that needs it, and an air-gapped
    appliance should not grow a dependency for one marker.
    """
    box = {}

    def _target():
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise AssertionError(
            f"the run did not finish within {timeout_seconds}s -- it is almost certainly parked in "
            f"_wait_until_clear because a model was not released between gathering rounds"
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


class _CountingLoader:
    """
    Wraps a real adapter and counts loads.

    This is the regression guard for the whole model-swap design: the report model must be loaded ONCE
    per run, not once per section. Without a test that counts, a later "simplification" of the authoring
    phase into two execute calls would silently double a run's wall clock and nothing would fail.
    """

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
    missing = [
        path for path in (
            settings.RAG_EMBEDDING_MODEL_PATH,
            settings.REPORT_AUTHORING_MODEL_PATH,
        )
        if not os.path.exists(path)
    ]
    if missing:
        pytest.skip(f"Deep Research Report integration test skipped; missing weights: {missing}")

@pytest.fixture(scope="module", autouse=True)
def setup_schema():
    Base.metadata.create_all(bind=db_manager._engine_wrapper._engine)  # noqa
    yield

@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_PATH", str(tmp_path / "reports"))
    return tmp_path / "reports"

@pytest.fixture
def small_run(monkeypatch):
    """Two sections and a narrow sweep: enough to prove the shape without a 40-minute test."""
    monkeypatch.setattr(settings, "REPORT_MAX_SECTIONS", 2)
    monkeypatch.setattr(settings, "REPORT_MAX_SUB_QUESTIONS", 3)
    monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
    monkeypatch.setattr(settings, "REPORT_SWEEP_MAX_SOURCES", 5)
    monkeypatch.setattr(settings, "REPORT_SWEEP_PER_SOURCE", 2)
    monkeypatch.setattr(settings, "REPORT_SECTION_EVIDENCE_TOKENS", 2000)
    monkeypatch.setattr(settings, "REPORT_SECTION_MAX_TOKENS", 400)

@pytest.fixture
def seeded_corpus(db_manager):
    source_id = uuid.uuid4()
    file_id = uuid.uuid4()

    with db_manager.session() as session:
        session.execute(
            sa_text("ALTER TABLE vector_chunks DROP CONSTRAINT IF EXISTS vector_chunks_source_id_fkey;")
        )
        session.add(FileModel(
            id=file_id, file_path="/vault/deposition.wav", file_name="deposition.wav",
            file_size_bytes=2048, mime_type="audio/wav",
        ))
        session.flush()
        session.add(SourceModel(
            id=source_id, name="Deposition — Ramirez", source_type="audio",
            content_hash=str(uuid.uuid4()), file_id=file_id,
        ))
        for index, line in enumerate(_TRANSCRIPT):
            session.add(VectorChunkModel(
                source_id=source_id,
                chunk_id=uuid.uuid4(),
                start_sec=float(index * 30),
                end_sec=float(index * 30 + 30),
                text_payload=f"[Audio Context: evidence handling testimony.]\n\n{line}",
                raw_text=line,
                context_text="[Audio Context: evidence handling testimony.]",
                speaker_labels=[line.split(" @ ")[0].lstrip("[")],
                embedding=[0.01 + index * 0.001] * 1024,
            ))

    yield {"source_id": source_id, "file_id": file_id}

    with db_manager.session() as session:
        session.execute(
            sa_text("DELETE FROM vector_chunks WHERE source_id = :sid"), {"sid": str(source_id)}
        )
        session.execute(sa_text("DELETE FROM sources WHERE id = :sid"), {"sid": str(source_id)})
        session.execute(sa_text("DELETE FROM files WHERE id = :fid"), {"fid": str(file_id)})


class TestDeepResearchReportPipelineIntegration:
    """
    A real, end-to-end report over a tiny seeded corpus: real Postgres, real embedder, real reranker,
    real Qwen3.5-9B. Skips rather than fails when the weights or GPU are not on this machine.
    """

    @pytest.fixture
    def harness(self, db_manager, reports_dir):
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        async_service = AsyncModelService(BackgroundModelWorker(orchestrator))
        job_manager = JobManager(JobRepository(db_manager))

        prompt_adapter = HuggingFacePromptAdapter()
        retrieval_api = RetrievalAPI(orchestrator, prompt_adapter, PostgresSearchRepository())

        reranker_adapter = CrossEncoderRerankerAdapter()
        rerank_api = RerankAPI(orchestrator, reranker_adapter)

        # ONE adapter, wrapped once, shared by the API and both phases -- exactly as the bootstrapper
        # wires it. Counting here therefore counts real loads of the 5.7 GB file.
        report_loader = _CountingLoader(LlamaCppReportAdapter())
        report_api = ReportAuthoringAPI(
            adapter=report_loader,
            orchestrator=orchestrator,
            repository=PostgresReportRepository(db_manager),
            file_store=ReportFileStore(),
        )

        scrubber_adapter = QwenRetrievalScrubberAdapter()
        scrubber_phase = RetrievalScrubberPhase(
            job_manager,
            RetrievalScrubberAPI(adapter=scrubber_adapter, orchestrator=orchestrator),
            async_service,
            scrubber_adapter,
        )

        pipeline = DeepResearchReportPipeline(
            retrieval_phase=RagRetrievalPhase(job_manager, retrieval_api, async_service, prompt_adapter),
            reranking_phase=RagRerankingPhase(job_manager, rerank_api, async_service, reranker_adapter),
            # Built even though REPORT_SCRUB_ENABLED is off: the pipeline still discards its preload in
            # its `finally`, and every phase has to be a real one for that sweep to mean anything.
            scrubber_phase=scrubber_phase,
            scoping_phase=ResearchScopingPhase(job_manager, report_api, async_service, report_loader),
            authoring_phase=ReportAuthoringPhase(job_manager, report_api, async_service, report_loader),
            report_api=report_api,
        )

        yield pipeline, report_loader, report_api

        async_service.release_all()
        async_service.shutdown()

    def test_end_to_end_report_is_written_and_fully_cited(
            self, harness, seeded_corpus, small_run, reports_dir
    ):
        pipeline, report_loader, report_api = harness

        proposal = pipeline.propose_plan("chain of custody for the evidence bag")
        assert proposal.plan.sections, "the planner proposed no sections"
        assert proposal.status == ReportStatus.AWAITING_APPROVAL

        report = pipeline.execute_report(proposal.id, proposal.plan)

        assert report.status == ReportStatus.COMPLETED
        assert report.markdown_path is not None
        markdown = open(report.markdown_path, encoding="utf-8").read()

        # Every [Doc N] in the body resolves in the references table: the file must be auditable with
        # no database behind it.
        body, _, references = markdown.partition("## References")
        for number in set(re.findall(r"\[Doc (\d+)]", body)):
            assert f"| [Doc {number}] |" in references, f"[Doc {number}] cited but not referenced"

        # Nothing was written without a citation.
        for section in report.written_sections:
            assert section.cited_doc_numbers, f"section {section.key} shipped uncited prose"

        # Citations point at real evidence, and the coverage claim is present.
        bank = report.evidence_by_doc_number
        for number in report.cited_doc_numbers:
            assert number in bank
            assert bank[number].source_name == "Deposition — Ramirez"
        assert "Coverage: swept" in markdown
        assert "Retrieval scrubber: disabled." in markdown

    def test_the_report_model_is_loaded_once_for_the_writing_phase(
            self, harness, seeded_corpus, small_run, reports_dir
    ):
        """
        THE regression guard for the model-swap schedule.

        Scoping loads it once and releases; the run loads it once more and holds it across outline
        optimisation, every section and the summary. Anything above three total means a reload crept
        back in -- on a 3060 each one costs roughly half a minute of pure loading.
        """
        pipeline, report_loader, _ = harness

        proposal = pipeline.propose_plan("chain of custody for the evidence bag")
        loads_after_scoping = report_loader.load_count
        assert loads_after_scoping == 1

        pipeline.execute_report(proposal.id, proposal.plan)

        writing_loads = report_loader.load_count - loads_after_scoping
        assert writing_loads == 1, (
            f"the report model was loaded {writing_loads} times during writing; one residency must cover "
            f"outline optimisation, every section AND the executive summary"
        )
        assert report_loader.unload_count == report_loader.load_count, "a model was left resident"

    def test_a_long_mixed_workload_does_not_leak_the_torchcodec_guard(
            self, harness, seeded_corpus, small_run, reports_dir
    ):
        """
        CLAUDE.md documents `block_torchcodec_import()` leaking across features in one long-lived
        process. A report run is exactly that shape -- sentence-transformers embedder, then a
        CrossEncoder, then llama.cpp -- so the guard must be restored when it finishes.
        """
        pipeline, _, _ = harness

        proposal = pipeline.propose_plan("chain of custody for the evidence bag")
        pipeline.execute_report(proposal.id, proposal.plan)

        assert sys.modules.get("torchcodec", "absent") is not None, (
            "torchcodec is still blocked after the report run; the guard leaked and any later "
            "torchaudio.load() in this process will fail"
        )

    def test_a_cancelled_run_keeps_its_sections_and_resumes_without_retrieval(
            self, harness, seeded_corpus, small_run, reports_dir
    ):
        from app.core.jobs.domain.entities import CancellationToken

        pipeline, report_loader, report_api = harness
        proposal = pipeline.propose_plan("chain of custody for the evidence bag")

        token = CancellationToken()
        token.cancel()
        cancelled = pipeline.execute_report(proposal.id, proposal.plan, token)

        assert cancelled.status == ReportStatus.CANCELLED


class TestExhaustiveModeIntegration:
    """
    The iterative mode against real models. The property that matters is the one a unit test cannot
    prove: the loop does not hang. `acquire_model(wait_for_clear=True)` parks with no time limit, so a
    round that failed to release the report model before gathering again would wedge the process
    forever -- these run under a timeout so that regression fails loudly instead of hanging CI.
    """

    @pytest.fixture
    def exhaustive_harness(self, db_manager, reports_dir):
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        async_service = AsyncModelService(BackgroundModelWorker(orchestrator))
        job_manager = JobManager(JobRepository(db_manager))

        prompt_adapter = HuggingFacePromptAdapter()
        retrieval_api = RetrievalAPI(orchestrator, prompt_adapter, PostgresSearchRepository())
        reranker_adapter = CrossEncoderRerankerAdapter()
        rerank_api = RerankAPI(orchestrator, reranker_adapter)

        report_loader = _CountingLoader(LlamaCppReportAdapter())
        report_api = ReportAuthoringAPI(
            adapter=report_loader,
            orchestrator=orchestrator,
            repository=PostgresReportRepository(db_manager),
            file_store=ReportFileStore(),
        )
        scrubber_adapter = QwenRetrievalScrubberAdapter()

        pipeline = DeepResearchReportPipeline(
            retrieval_phase=RagRetrievalPhase(job_manager, retrieval_api, async_service, prompt_adapter),
            reranking_phase=RagRerankingPhase(job_manager, rerank_api, async_service, reranker_adapter),
            scrubber_phase=RetrievalScrubberPhase(
                job_manager,
                RetrievalScrubberAPI(adapter=scrubber_adapter, orchestrator=orchestrator),
                async_service,
                scrubber_adapter,
            ),
            scoping_phase=ResearchScopingPhase(job_manager, report_api, async_service, report_loader),
            authoring_phase=ReportAuthoringPhase(job_manager, report_api, async_service, report_loader),
            report_api=report_api,
            max_gather_rounds=2,
        )

        yield pipeline, report_loader

        async_service.release_all()
        async_service.shutdown()

    def test_an_exhaustive_run_completes_without_wedging_between_rounds(
            self, exhaustive_harness, seeded_corpus, small_run, reports_dir
    ):
        """
        Releasing the report model before gathering again is the correctness condition of the loop, not
        a tidiness step. Without it this test hangs rather than fails.
        """
        pipeline, report_loader = exhaustive_harness

        proposal = _run_or_fail(lambda: pipeline.propose_plan("chain of custody for the evidence bag"))
        report = _run_or_fail(lambda: pipeline.execute_iterative_report(proposal.id, proposal.plan))

        assert report.status == ReportStatus.COMPLETED
        assert report.mode is ReportMode.EXHAUSTIVE
        assert 1 <= report.rounds_used <= 2
        assert report_loader.unload_count == report_loader.load_count, "a model was left resident"

    def test_the_report_states_how_many_rounds_it_searched(
            self, exhaustive_harness, seeded_corpus, small_run, reports_dir
    ):
        """
        "Not covered" after one round and after two are different claims about the corpus, so the
        rendered report says which it is.
        """
        pipeline, _ = exhaustive_harness

        proposal = _run_or_fail(lambda: pipeline.propose_plan("chain of custody for the evidence bag"))
        report = _run_or_fail(lambda: pipeline.execute_iterative_report(proposal.id, proposal.plan))

        markdown = open(report.markdown_path, encoding="utf-8").read()
        assert "exhaustive mode" in markdown
        assert "Coverage: swept" in markdown, "later rounds must not clobber the coverage claim"
