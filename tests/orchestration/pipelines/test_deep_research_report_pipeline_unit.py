# Path: tests/orchestration/pipelines/test_deep_research_report_pipeline_unit.py

import uuid
from unittest.mock import MagicMock, Mock, call

import pytest

from app.core.config import settings
from app.core.jobs.domain.entities import CancellationToken
from app.core.jobs.exceptions.execution import JobCancelled
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, FollowUpQuery, OutlineSection, Report, ReportSection, ResearchPlan,
    SubQuestion,
)
from app.features.deep_research_report.domain.enums import ReportMode, ReportStatus, SectionStatus
from app.features.rag_retrieval.domain.entities import (
    BatchSearchResult, IndexedSource, RetrievedChunk, SearchResult,
)
from app.orchestration.domain.entities import (
    EvidenceSweepJobResult, PhaseBatchResult, RagRerankingJobResult, ResearchScopingJobResult,
)
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.exceptions.workflow import WorkflowStateError
from app.orchestration.service.pipelines.deep_research_report_pipeline import DeepResearchReportPipeline

REPORT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOURCE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _chunk(start_sec: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        source_id=SOURCE_ID,
        text_payload="context. transcript",
        raw_text="[SPEAKER_00 @ 00:00:00]: transcript",
        speaker_labels=["SPEAKER_00"],
        start_sec=start_sec,
        end_sec=start_sec + 30,
        relevance_score=0.5,
    )


def _plan(report_id=REPORT_ID) -> ResearchPlan:
    return ResearchPlan(
        report_id=report_id,
        topic="Chain of custody",
        sections=[
            OutlineSection(key="s1", title="Handling", order_index=0),
            OutlineSection(key="s2", title="Storage", order_index=1),
        ],
        sub_questions=[
            SubQuestion(text="Who sealed it?", section_key="s1", order_index=0),
            SubQuestion(text="Where was it kept?", section_key="s2", order_index=1),
        ],
    )


def _stored_report(status=ReportStatus.AWAITING_APPROVAL, sections=None, evidence=None) -> Report:
    return Report(
        id=REPORT_ID,
        topic="Chain of custody",
        status=status,
        plan=_plan(),
        sections=sections if sections is not None else [
            ReportSection(key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
                          markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1]),
            ReportSection(key="s2", title="Storage", order_index=1, status=SectionStatus.WRITTEN,
                          markdown_body="Stored [Doc 2].", cited_doc_numbers=[2]),
        ],
        evidence=evidence if evidence is not None else [],
    )


@pytest.fixture
def parent():
    """
    One parent mock so `mock_calls` records the ORDER of calls ACROSS phases.

    That ordering is the design: each preload is fired a stage early so its load overlaps the previous
    stage's compute, and the report model is preloaded exactly once for the whole writing phase.
    """
    return MagicMock()


@pytest.fixture
def phases(parent):
    retrieval = parent.retrieval
    reranking = parent.reranking
    scrubber = parent.scrubber
    scoping = parent.scoping
    authoring = parent.authoring

    retrieval.list_indexed_sources.return_value = [
        IndexedSource(source_id=SOURCE_ID, name="Deposition", chunk_count=30)
    ]
    retrieval.expand_neighbors.side_effect = lambda chunks: list(chunks)
    retrieval.get_chunks.return_value = []

    chunk_a, chunk_b = _chunk(0), _chunk(60)
    batch = BatchSearchResult(
        results=[
            SearchResult(query="Who sealed it?", results=[chunk_a]),
            SearchResult(query="Where was it kept?", results=[chunk_b]),
        ],
        sweep_results=[],
        swept_source_count=1,
    )

    # Each round returns DIFFERENT chunks, the way a re-worded search actually would. A fixture that
    # returned the same ones every time would make every round after the first look like "added
    # nothing new" and silently stop the loop before the behaviour under test ran.
    def _sweep(**kwargs):
        round_batch = BatchSearchResult(
            results=[SearchResult(query=text, results=[_chunk(len(_sweep.seen) * 1000 + index * 30)])
                     for index, (text, _) in enumerate(kwargs["queries"])],
            sweep_results=[],
            swept_source_count=1 if kwargs.get("sweep") is not None else 0,
        )
        _sweep.seen.append(round_batch)
        reranking.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING,
            status=PhaseStatus.COMPLETED,
            items=[
                (RagRerankingJobResult(search_id=uuid.uuid4(), success=True), result)
                for result in round_batch.results
            ],
        )
        return (
            PhaseBatchResult(
                phase=PipelinePhase.RESEARCH_SWEEP,
                status=PhaseStatus.COMPLETED,
                items=[
                    (EvidenceSweepJobResult(report_id=REPORT_ID, success=True, query=result.query,
                                            chunks_found=len(result.results)), result)
                    for result in round_batch.results
                ],
            ),
            round_batch,
        )

    _sweep.seen = []
    retrieval.execute_sweep.side_effect = _sweep

    reranking.execute.return_value = PhaseBatchResult(
        phase=PipelinePhase.RAG_RERANKING,
        status=PhaseStatus.COMPLETED,
        items=[
            (RagRerankingJobResult(search_id=uuid.uuid4(), success=True), result)
            for result in batch.results
        ],
    )

    authoring.execute_report.return_value = PhaseBatchResult(
        phase=PipelinePhase.REPORT_SECTION, status=PhaseStatus.COMPLETED, successful_count=2
    )
    authoring.was_cancelled.return_value = False
    authoring.stopped_for_gaps.return_value = False
    authoring.follow_ups.return_value = []
    authoring.summary.return_value = "Summary [Doc 1]."

    return retrieval, reranking, scrubber, scoping, authoring


@pytest.fixture
def report_api():
    api = Mock()
    api.get_report.return_value = _stored_report()
    api.render_and_finalize.return_value = _stored_report(ReportStatus.COMPLETED)
    api.render_partial.return_value = (_stored_report(ReportStatus.CANCELLED), "/reports/partial.md")
    return api


@pytest.fixture
def make_pipeline(phases, report_api):
    """
    `max_gather_rounds` is captured in __init__ (the bootstrapper passes the setting once), so a test
    that wants a different budget has to build the pipeline with it rather than monkeypatching.
    """
    def _build(max_gather_rounds: int = 1) -> DeepResearchReportPipeline:
        retrieval, reranking, scrubber, scoping, authoring = phases
        return DeepResearchReportPipeline(
            retrieval_phase=retrieval,
            reranking_phase=reranking,
            scrubber_phase=scrubber,
            scoping_phase=scoping,
            authoring_phase=authoring,
            report_api=report_api,
            max_gather_rounds=max_gather_rounds,
        )
    return _build


@pytest.fixture
def pipeline(phases, report_api):
    retrieval, reranking, scrubber, scoping, authoring = phases
    return DeepResearchReportPipeline(
        retrieval_phase=retrieval,
        reranking_phase=reranking,
        scrubber_phase=scrubber,
        scoping_phase=scoping,
        authoring_phase=authoring,
        report_api=report_api,
    )


def _names(parent, keep=("preload_model", "execute", "execute_sweep", "execute_report")):
    """The cross-phase call order, as 'phase.method' strings."""
    ordered = []
    for name, _, _ in parent.mock_calls:
        parts = name.split(".")
        if len(parts) == 2 and parts[1] in keep:
            ordered.append(name)
    return ordered


class TestSwapBudget:
    """
    The tests that protect the model-swap schedule. If someone 'simplifies' the authoring phase into two
    execute calls, or drops a preload, these fail -- which is the whole point of having them.
    """

    def test_preloads_are_fired_one_stage_early_so_loads_overlap_compute(
            self, pipeline, parent, monkeypatch
    ):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)

        pipeline.execute_report(REPORT_ID, _plan())

        assert _names(parent) == [
            "retrieval.preload_model",
            "retrieval.execute_sweep",
            # fired only once there is something to rerank: a preload on a path that returned early
            # would be orphaned by the next round and could never be released
            "reranking.preload_model",
            "reranking.execute",
            # the 5.7 GB load is queued before the banking SQL, so it loads during that work
            "authoring.preload_model",
            "authoring.execute_report",
        ]

    def test_the_report_model_is_preloaded_exactly_once_for_the_whole_writing_phase(
            self, pipeline, parent, phases, monkeypatch
    ):
        """
        Outline optimisation and every section write share ONE residency. A per-section arrangement of
        the same work would cost roughly eight loads of a 5.7 GB file instead of one.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases

        pipeline.execute_report(REPORT_ID, _plan())

        assert authoring.preload_model.call_count == 1
        assert authoring.execute_report.call_count == 1

    def test_the_scrubber_is_skipped_entirely_when_disabled(self, pipeline, parent, phases, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, scrubber, _, _ = phases

        pipeline.execute_report(REPORT_ID, _plan())

        scrubber.preload_model.assert_not_called()
        scrubber.execute.assert_not_called()

    def test_with_scrubbing_on_the_heavy_load_hides_behind_the_scrub_pass(
            self, pipeline, parent, phases, monkeypatch
    ):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", True)
        _, _, scrubber, _, _ = phases
        scrubber.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RETRIEVAL_SCRUBBING, status=PhaseStatus.COMPLETED, items=[]
        )

        pipeline.execute_report(REPORT_ID, _plan())

        order = _names(parent)
        assert order.index("scrubber.preload_model") < order.index("reranking.execute")
        assert order.index("scrubber.execute") < order.index("authoring.preload_model")

    def test_one_batched_search_covers_every_sub_question_and_the_sweep(self, pipeline, phases):
        """
        One embedder acquisition for the whole fan-out. Calling the retrieval phase per sub-question
        would be the same SQL at the cost of one 2.2 GB load each.
        """
        retrieval, _, _, _, _ = phases

        pipeline.execute_report(REPORT_ID, _plan())

        assert retrieval.execute_sweep.call_count == 1
        kwargs = retrieval.execute_sweep.call_args.kwargs
        assert [text for text, _ in kwargs["queries"]] == ["Who sealed it?", "Where was it kept?"]
        assert kwargs["sweep"] is not None

    def test_every_phase_discards_its_preload_when_the_run_fails(self, pipeline, parent, phases, report_api):
        """
        A preload left parked holds the single background worker and blocks every later preload in the
        process -- not just this run's.
        """
        retrieval, reranking, scrubber, scoping, authoring = phases
        # The fixture drives execute_sweep with a side_effect, which takes precedence over return_value.
        retrieval.execute_sweep.side_effect = None
        retrieval.execute_sweep.return_value = (
            PhaseBatchResult(phase=PipelinePhase.RESEARCH_SWEEP, status=PhaseStatus.FAILED,
                             errors=["boom"]),
            BatchSearchResult(),
        )

        with pytest.raises(WorkflowStateError):
            pipeline.execute_report(REPORT_ID, _plan())

        # Discarded per loop iteration AND once more in the run's finally. The per-iteration sweep is
        # what stops a round's unresolved preload from being orphaned by the next round's.
        for phase in (retrieval, reranking, scrubber, scoping, authoring):
            assert phase.discard_preload.call_count >= 1


class TestEvidenceBank:
    def test_the_bank_is_persisted_with_report_wide_numbers_before_writing_starts(
            self, pipeline, report_api, monkeypatch
    ):
        """The commit that makes a run resumable without touching a retrieval model again."""
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)

        pipeline.execute_report(REPORT_ID, _plan())

        report_api.save_evidence_bank.assert_called_once()
        args = report_api.save_evidence_bank.call_args
        evidence = args.args[1]
        assert [e.doc_number for e in evidence] == [1, 2]
        assert args.args[2] == 1  # swept_source_count
        assert args.kwargs["scrub_enabled"] is False
        assert args.kwargs["mode"] is ReportMode.DEEP
        assert args.kwargs["rounds_used"] == 1

    def test_source_names_are_resolved_onto_every_entry(self, pipeline, report_api, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)

        pipeline.execute_report(REPORT_ID, _plan())

        evidence = report_api.save_evidence_bank.call_args.args[1]
        assert {e.source_name for e in evidence} == {"Deposition"}

    def test_the_approved_plan_is_validated_and_stored_before_anything_runs(
            self, pipeline, report_api, monkeypatch
    ):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)

        pipeline.execute_report(REPORT_ID, _plan())

        report_api.approve_plan.assert_called_once()
        stored_plan = report_api.approve_plan.call_args.args[1]
        assert stored_plan.revision == 2


class TestOutlineValidation:
    def test_an_invalid_outline_is_reported_as_such_not_as_a_crash(self, pipeline, report_api):
        """
        A user's outline being wrong is not a run failing. Marking the report FAILED would strand it
        over a fixable typo, and the caller could not tell them what to correct.
        """
        from app.features.deep_research_report.exceptions.pipeline import OutlineValidationError

        plan = _plan().model_copy(update={"sections": []})

        with pytest.raises(OutlineValidationError):
            pipeline.execute_report(REPORT_ID, plan)

        report_api.set_status.assert_not_called()
        report_api.approve_plan.assert_not_called()


class TestCancellation:
    def test_cancelling_before_gathering_stops_without_searching(self, pipeline, phases, report_api):
        retrieval, _, _, _, _ = phases
        token = CancellationToken()
        token.cancel()

        report = pipeline.execute_report(REPORT_ID, _plan(), token)

        retrieval.execute_sweep.assert_not_called()
        report_api.set_status.assert_called_once_with(
            REPORT_ID, ReportStatus.CANCELLED, "Cancelled by the user."
        )
        assert report is not None

    def test_a_cancelled_run_still_renders_what_was_written(self, pipeline, phases, report_api, monkeypatch):
        """Twenty minutes of cited sections should not be thrown away because the user stopped."""
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        authoring.was_cancelled.return_value = True

        pipeline.execute_report(REPORT_ID, _plan())

        report_api.render_partial.assert_called_once_with(REPORT_ID)
        report_api.render_and_finalize.assert_not_called()

    def test_the_token_reaches_the_search_and_the_writer(self, pipeline, phases, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        token = CancellationToken()

        pipeline.execute_report(REPORT_ID, _plan(), token)

        assert retrieval.execute_sweep.call_args.kwargs["cancel_token"] is token
        assert authoring.execute_report.call_args.kwargs["cancel_token"] is token

    def test_a_job_cancelled_raised_mid_gather_is_not_treated_as_a_failure(
            self, pipeline, phases, report_api, monkeypatch
    ):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, _ = phases
        retrieval.execute_sweep.side_effect = JobCancelled("stopped")

        report = pipeline.execute_report(REPORT_ID, _plan())

        assert report is not None
        report_api.set_status.assert_called_once_with(
            REPORT_ID, ReportStatus.CANCELLED, "Cancelled by the user."
        )


class TestResume:
    def test_resume_writes_only_pending_sections_and_loads_no_retrieval_model(
            self, pipeline, phases, report_api
    ):
        """
        The bank is the run's memory and it is already complete: a crash in section six of eight costs
        two sections and one model load, not a whole re-gather.
        """
        retrieval, reranking, _, _, authoring = phases
        evidence = [
            EvidenceEntry(doc_number=1, chunk_id=uuid.uuid4(), source_id=SOURCE_ID,
                          source_name="Deposition", start_sec=0, end_sec=30)
        ]
        report_api.get_report.return_value = _stored_report(
            status=ReportStatus.CANCELLED,
            sections=[
                ReportSection(key="s1", title="Handling", order_index=0, status=SectionStatus.WRITTEN,
                              markdown_body="Sealed [Doc 1].", cited_doc_numbers=[1]),
                ReportSection(key="s2", title="Storage", order_index=1, status=SectionStatus.PENDING),
            ],
            evidence=evidence,
        )

        pipeline.resume_report(REPORT_ID)

        retrieval.execute_sweep.assert_not_called()
        reranking.execute.assert_not_called()
        kwargs = authoring.execute_report.call_args.kwargs
        assert kwargs["pending_keys"] == ["s2"]
        assert kwargs["optimize"] is False

    def test_resume_without_an_evidence_bank_is_refused_with_a_reason(self, pipeline, report_api):
        report_api.get_report.return_value = _stored_report(
            status=ReportStatus.CANCELLED, evidence=[]
        )

        with pytest.raises(WorkflowStateError, match="before its evidence was gathered"):
            pipeline.resume_report(REPORT_ID)

    def test_resume_with_nothing_pending_just_finalizes(self, pipeline, phases, report_api):
        _, _, _, _, authoring = phases
        report_api.get_report.return_value = _stored_report(
            status=ReportStatus.CANCELLED,
            evidence=[EvidenceEntry(doc_number=1, chunk_id=uuid.uuid4(), source_id=SOURCE_ID,
                                    source_name="D", start_sec=0, end_sec=1)],
        )

        pipeline.resume_report(REPORT_ID)

        authoring.execute_report.assert_not_called()
        report_api.render_and_finalize.assert_called_once()

    def test_resuming_a_missing_report_is_refused(self, pipeline, report_api):
        report_api.get_report.return_value = None

        with pytest.raises(WorkflowStateError, match="does not exist"):
            pipeline.resume_report(REPORT_ID)


class TestProposePlan:
    def test_refuses_a_topic_when_nothing_is_indexed(self, pipeline, phases):
        """
        Stated plainly rather than producing an empty report: there is nothing to research, and the user
        needs to know it is an ingestion problem.
        """
        retrieval, _, _, _, _ = phases
        retrieval.list_indexed_sources.return_value = []

        with pytest.raises(WorkflowStateError, match="no ingested media has been indexed"):
            pipeline.propose_plan("Chain of custody")

    def test_the_planner_sees_real_excerpts_and_the_real_source_list(self, pipeline, phases):
        retrieval, _, _, scoping, _ = phases
        scoping.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RESEARCH_SCOPING,
            status=PhaseStatus.COMPLETED,
            items=[(ResearchScopingJobResult(report_id=REPORT_ID, success=True), _stored_report())],
        )

        pipeline.propose_plan("Chain of custody")

        corpus: CorpusDigest = scoping.execute.call_args.args[0][0][1]
        assert corpus.indexed_source_count == 1
        assert corpus.sources[0].name == "Deposition"
        assert corpus.sample_digests  # real transcript, not an invented sample

    def test_scoping_failure_halts_with_a_workflow_error(self, pipeline, phases):
        _, _, _, scoping, _ = phases
        scoping.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RESEARCH_SCOPING, status=PhaseStatus.FAILED, errors=["model died"]
        )

        with pytest.raises(WorkflowStateError, match="scoping failed"):
            pipeline.propose_plan("Chain of custody")

    def test_discards_preloads_even_when_scoping_fails(self, pipeline, phases):
        retrieval, reranking, scrubber, scoping, authoring = phases
        scoping.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RESEARCH_SCOPING, status=PhaseStatus.FAILED, errors=["model died"]
        )

        with pytest.raises(WorkflowStateError):
            pipeline.propose_plan("Chain of custody")

        # Discarded per loop iteration AND once more in the run's finally. The per-iteration sweep is
        # what stops a round's unresolved preload from being orphaned by the next round's.
        for phase in (retrieval, reranking, scrubber, scoping, authoring):
            assert phase.discard_preload.call_count >= 1


class TestExhaustiveRounds:
    """
    The iterative mode. The invariant under all of it: the report model is acquired once per round, and
    the final round's residency also writes -- so a one-round exhaustive run costs what a deep run costs.
    """

    @staticmethod
    def _gap_round(authoring, follow_up_text="re-worded query"):
        """
        Make the authoring phase report an unsupported section with something to search for.

        `stopped_for_gaps` mirrors the flag it was called with, because the real phase can only stop
        when `stop_on_gaps=True` was passed -- a mock that always says "stopped" would take the gap path
        on the final writing round too, which the phase cannot do.
        """
        authoring.stopped_for_gaps.side_effect = (
            lambda _result: bool(authoring.execute_report.call_args.kwargs["stop_on_gaps"])
        )
        authoring.follow_ups.return_value = [
            FollowUpQuery(section_key="s2", text=follow_up_text)
        ]

    def test_a_covered_outline_writes_in_one_round(self, make_pipeline, phases, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases

        make_pipeline().execute_iterative_report(REPORT_ID, _plan())

        assert retrieval.execute_sweep.call_count == 1
        assert authoring.execute_report.call_count == 1

    def test_one_round_exhaustive_costs_the_same_as_deep(self, make_pipeline, phases, monkeypatch):
        """The regression guard on the whole design: iterating must not tax the common case."""
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, reranking, _, _, authoring = phases

        make_pipeline().execute_iterative_report(REPORT_ID, _plan())

        assert retrieval.preload_model.call_count == 1
        assert reranking.preload_model.call_count == 1
        assert authoring.preload_model.call_count == 1

    def test_an_unsupported_section_triggers_another_gathering_round(
            self, make_pipeline, phases, monkeypatch
    ):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=3).execute_iterative_report(REPORT_ID, _plan())

        # Rounds 1 and 2 both gather and hand back; the third is the ceiling and writes.
        assert retrieval.execute_sweep.call_count == 3

    def test_the_follow_up_queries_are_what_the_next_round_searches(
            self, make_pipeline, phases, monkeypatch
    ):
        """
        The point of the whole loop: re-ask using wording the model could see in the bank, rather than
        repeating a sub-question that already failed.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        self._gap_round(authoring, follow_up_text="what did Draghi say about the wedge")

        make_pipeline(max_gather_rounds=2).execute_iterative_report(REPORT_ID, _plan())

        second_round = retrieval.execute_sweep.call_args_list[1].kwargs
        assert [text for text, _ in second_round["queries"]] == ["what did Draghi say about the wedge"]
        # The corpus-wide sweep is topic-wide, so repeating it would find the same sources.
        assert second_round["sweep"] is None

    def test_a_round_that_adds_nothing_new_stops_the_loop(self, make_pipeline, phases, monkeypatch):
        """
        Code decides *when* to stop, on an objective fact. Re-searching forever because the model keeps
        proposing queries would be the failure mode of asking it to judge completeness.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        self._gap_round(authoring)
        # Every round finds exactly the same chunk, so the deduped bank stops growing.
        fixed = retrieval.execute_sweep.side_effect(queries=[("q", None)], sweep=None)
        retrieval.execute_sweep.side_effect = None
        retrieval.execute_sweep.return_value = fixed

        make_pipeline(max_gather_rounds=5).execute_iterative_report(REPORT_ID, _plan())

        # Round 2 adds nothing, so the loop stops rather than running all five rounds.
        assert retrieval.execute_sweep.call_count == 2

    def test_gaps_with_no_follow_ups_stop_the_loop(self, make_pipeline, phases, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        authoring.stopped_for_gaps.return_value = True
        authoring.follow_ups.return_value = []

        make_pipeline(max_gather_rounds=3).execute_iterative_report(REPORT_ID, _plan())

        assert retrieval.execute_sweep.call_count == 1
        # Still writes: an uncovered section is reported as not covered, not left unwritten.
        assert authoring.execute_report.call_count == 2

    def test_only_the_last_round_is_allowed_to_write(self, make_pipeline, phases, monkeypatch):
        """
        stop_on_gaps is what releases the model so the next round can acquire the embedder. Without it
        the run would park forever on acquire_model(wait_for_clear=True).
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=3).execute_iterative_report(REPORT_ID, _plan())

        flags = [c.kwargs["stop_on_gaps"] for c in authoring.execute_report.call_args_list]
        assert flags[-1] is False
        assert all(flags[:-1])

    def test_deep_mode_never_stops_for_gaps(self, make_pipeline, phases, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases

        make_pipeline().execute_report(REPORT_ID, _plan())

        assert all(not c.kwargs["stop_on_gaps"] for c in authoring.execute_report.call_args_list)

    def test_the_mode_and_round_count_are_persisted(self, make_pipeline, phases, report_api, monkeypatch):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=2).execute_iterative_report(REPORT_ID, _plan())

        last = report_api.save_evidence_bank.call_args
        assert last.kwargs["mode"] is ReportMode.EXHAUSTIVE
        assert last.kwargs["rounds_used"] == 2

    def test_the_sweep_coverage_count_is_carried_forward(self, make_pipeline, phases, report_api, monkeypatch):
        """
        Later rounds run no sweep and report 0. Passing that through would set the column to 0 and the
        renderer would silently drop the coverage line entirely.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=2).execute_iterative_report(REPORT_ID, _plan())

        for call in report_api.save_evidence_bank.call_args_list:
            assert call.args[2] == 1

    def test_evidence_is_deduped_across_rounds_not_only_within_one(
            self, make_pipeline, phases, report_api, monkeypatch
    ):
        """
        research_report_evidence carries UniqueConstraint(report_id, chunk_id), so a round-1 chunk
        re-found in round 2 would raise an IntegrityError and kill the run.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=2).execute_iterative_report(REPORT_ID, _plan())

        for call in report_api.save_evidence_bank.call_args_list:
            chunk_ids = [e.chunk_id for e in call.args[1]]
            assert len(chunk_ids) == len(set(chunk_ids))

    def test_preloads_are_discarded_every_round_not_just_at_the_end(
            self, make_pipeline, phases, monkeypatch
    ):
        """
        An unresolved preload left over from round N becomes unreachable the moment round N+1 overwrites
        the future -- nothing can release that model, and the next acquisition parks forever.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=3).execute_iterative_report(REPORT_ID, _plan())

        # Once per round plus the run's own finally.
        assert retrieval.discard_preload.call_count >= 3

    def test_the_assignment_pass_is_told_what_has_already_been_searched(
            self, make_pipeline, phases, monkeypatch
    ):
        """
        Round 2 has to be able to write a genuinely new query, and it cannot re-word a search it cannot
        see. The history carries what each one returned, so a query that found nothing is recognisable
        as vocabulary the archive does not have.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=3).execute_iterative_report(REPORT_ID, _plan())

        second_pass = authoring.execute_report.call_args_list[1].kwargs["asked"]
        texts = [a.text for a in second_pass]
        assert "Who sealed it?" in texts and "Where was it kept?" in texts
        assert all(isinstance(a.chunks_added, int) for a in second_pass)

    def test_the_history_accumulates_across_rounds(self, make_pipeline, phases, monkeypatch):
        """Round 3 must know what round 2 asked, not only what round 1 asked."""
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring, follow_up_text="what did Draghi say about the wedge")

        # Four, so that the third round is still a gathering round -- the last round discards
        # follow-ups and is deliberately handed no history.
        make_pipeline(max_gather_rounds=4).execute_iterative_report(REPORT_ID, _plan())

        third_pass = authoring.execute_report.call_args_list[2].kwargs["asked"]
        texts = [a.text for a in third_pass]
        assert "what did Draghi say about the wedge" in texts
        assert "Who sealed it?" in texts

    def test_deep_mode_is_never_handed_a_search_history(self, make_pipeline, phases, monkeypatch):
        """
        A single-round run discards follow-ups, so showing it the history would spend context to change
        nothing -- and would stop deep mode's outline prompt being the one it has always sent.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases

        make_pipeline().execute_report(REPORT_ID, _plan())

        assert authoring.execute_report.call_args.kwargs["asked"] is None

    def test_the_final_writing_round_is_not_handed_a_history_either(
            self, make_pipeline, phases, monkeypatch
    ):
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=2).execute_iterative_report(REPORT_ID, _plan())

        assert authoring.execute_report.call_args_list[-1].kwargs["asked"] is None

    def test_a_query_is_credited_only_with_chunks_new_to_the_bank(
            self, make_pipeline, phases, monkeypatch
    ):
        """
        The count answers "did asking this buy us anything", which is what decides whether asking again
        is worth three model loads -- not how many rows came back.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        _, _, _, _, authoring = phases
        self._gap_round(authoring)

        make_pipeline(max_gather_rounds=3).execute_iterative_report(REPORT_ID, _plan())

        first_round = [a for a in authoring.execute_report.call_args_list[1].kwargs["asked"]
                       if a.round_index == 1]
        # The fixture returns one distinct chunk per query per round.
        assert [a.chunks_added for a in first_round] == [1, 1]

    def test_a_round_with_nothing_new_to_ask_is_not_spent(
            self, make_pipeline, phases, monkeypatch
    ):
        """
        The loop's other stop rule. When every remaining gap can only be re-asked with a query this run
        has already run, the round would return the same chunks and reach the same conclusion -- after
        paying an embedder, a reranker and a ~5.7 GB report-model load for it. `optimize_outline` says
        so by returning no follow-ups, and the pipeline must write rather than gather again.
        """
        monkeypatch.setattr(settings, "REPORT_SCRUB_ENABLED", False)
        retrieval, _, _, _, authoring = phases
        self._gap_round(authoring)
        # Round 2's assignment finds the gap again but has nothing unasked left to search for.
        authoring.follow_ups.side_effect = [
            [FollowUpQuery(section_key="s2", text="re-worded query")], [],
        ]

        make_pipeline(max_gather_rounds=5).execute_iterative_report(REPORT_ID, _plan())

        assert retrieval.execute_sweep.call_count == 2
        # The budget allowed five; the run still wrote instead of burning the remaining three.
        assert authoring.execute_report.call_count == 3


class TestRerankAttribution:
    """
    `_rerank` must attribute a reranked pool back to the sub-question that produced it.

    `RagRerankingPhase.execute` skips any input whose job result is unsuccessful rather than emitting a
    placeholder, so pairing its output positionally against the queries shifts every later query onto
    another sub-question's chunks. Nothing fails, and the report cites evidence under the wrong heading.
    """

    @staticmethod
    def _pipeline(reranking):
        return DeepResearchReportPipeline(
            retrieval_phase=Mock(),
            reranking_phase=reranking,
            scrubber_phase=Mock(),
            scoping_phase=Mock(),
            authoring_phase=Mock(),
            report_api=Mock(),
        )

    def test_each_question_keeps_its_own_reranked_chunks_when_a_middle_query_is_skipped(self):
        first, second, third = _chunk(0), _chunk(100), _chunk(200)
        queries = [("q1", Mock()), ("q2", Mock()), ("q3", Mock())]
        results = [
            SearchResult(query="q1", results=[first]),
            SearchResult(query="q2", results=[second]),
            SearchResult(query="q3", results=[third]),
        ]

        reranking = Mock()

        def execute(pairs, top_n):
            # The phase's real behaviour: an input it does not process contributes NO item at all.
            emitted = [
                (RagRerankingJobResult(search_id=job.search_id, success=True, reranked_chunks_count=1),
                 payload)
                for job, payload in pairs
                if payload.query != "q2"
            ]
            return PhaseBatchResult(
                phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.COMPLETED,
                total_processed=len(emitted), successful_count=len(emitted), items=emitted,
            )

        reranking.execute.side_effect = execute

        ranked = self._pipeline(reranking)._rerank(REPORT_ID, queries, results)

        assert [question for question, _ in ranked] == ["q1", "q2", "q3"]
        assert ranked[0][1] == [first]
        # q2 was skipped by the phase, so it falls back to its own RRF order -- not q3's chunks.
        assert ranked[1][1] == [second]
        assert ranked[2][1] == [third]

    def test_a_failed_phase_keeps_every_query_on_its_own_rrf_order(self):
        queries = [("q1", Mock()), ("q2", Mock())]
        results = [SearchResult(query="q1", results=[_chunk(0)]),
                   SearchResult(query="q2", results=[_chunk(50)])]
        reranking = Mock()
        reranking.execute.return_value = PhaseBatchResult(
            phase=PipelinePhase.RAG_RERANKING, status=PhaseStatus.FAILED, errors=["no weights"],
        )

        ranked = self._pipeline(reranking)._rerank(REPORT_ID, queries, results)

        assert ranked == [("q1", results[0].results), ("q2", results[1].results)]
