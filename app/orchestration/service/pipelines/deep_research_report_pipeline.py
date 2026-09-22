# Path: app/orchestration/service/pipelines/deep_research_report_pipeline.py

import logging
import uuid
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.jobs.domain.entities import CancellationToken
from app.core.jobs.exceptions.execution import JobCancelled
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, EvidenceEntry, IndexedSource as ReportIndexedSource, Report, ReportSection, ResearchPlan,
)
from app.features.deep_research_report.domain.enums import (
    EvidenceOrigin, ReportMode, ReportStatus, SectionStatus,
)
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI
from app.features.deep_research_report.utils.evidence_bank import build_digest, dedupe_hits, number_bank
from app.features.deep_research_report.exceptions.pipeline import OutlineValidationError
from app.features.deep_research_report.utils.outline_validator import validate_and_normalise
from app.features.rag_retrieval.domain.entities import (
    CorpusSweep, RetrievedChunk, SearchQuery, SearchResult,
)
from app.orchestration.domain.entities import (
    PhaseBatchResult, RagRerankingJobResult, RagRetrievalJobResult, ResearchScopingJobResult,
    RetrievalScrubbingJobResult,
)
from app.orchestration.domain.enums import PhaseStatus
from app.orchestration.exceptions.workflow import WorkflowExecutionError, WorkflowStateError
from app.orchestration.service.phases.rag_reranking_phase import RagRerankingPhase
from app.orchestration.service.phases.rag_retrieval_phase import RagRetrievalPhase
from app.orchestration.service.phases.report_authoring_phase import ReportAuthoringPhase
from app.orchestration.service.phases.research_scoping_phase import ResearchScopingPhase
from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase

logger = logging.getLogger(__name__)

# How many real excerpts the planner sees from the grounding search. Enough to show it the corpus's
# vocabulary, few enough to leave the window free.
_SCOPING_SAMPLE_SIZE = 15


class DeepResearchReportPipeline:
    """
    Topic in, multi-page cited report out.

    Two entrypoints, because the run is long enough that starting it on a misread topic is expensive:
    `propose_plan` scopes cheaply and stops for approval, `execute_report` does the 30-minute run.

    The model-swap schedule is the load-bearing part of this class. Preloads are fired one stage EARLY --
    `trigger_preload` parks until VRAM clears, so the next model loads during the current stage's compute
    instead of after it. A whole run costs six model loads, two of them the ~5.7 GB report model; a naive
    per-section arrangement of exactly the same work costs about forty, sixteen of them heavy.
    """

    def __init__(
            self,
            retrieval_phase: RagRetrievalPhase,
            reranking_phase: RagRerankingPhase,
            scrubber_phase: RetrievalScrubberPhase,
            scoping_phase: ResearchScopingPhase,
            authoring_phase: ReportAuthoringPhase,
            report_api: ReportAuthoringAPI,
            max_gather_rounds: int = 1,
    ):
        self._retrieval_phase = retrieval_phase
        self._reranking_phase = reranking_phase
        self._scrubber_phase = scrubber_phase
        self._scoping_phase = scoping_phase
        self._authoring_phase = authoring_phase
        self._report_api = report_api
        # Accepted and stored, but NOT yet acted on: gathering currently runs exactly one round. Wiring
        # the second round means looping back to _gather() with sub-questions aimed at the sections the
        # outline pass could not support -- and it must trigger_unload the report model first, because
        # acquiring the embedder with wait_for_clear=True while holding the 9B would park forever.
        self._max_gather_rounds = max_gather_rounds

    # ==========================================
    # CLEANUP
    # ==========================================

    def _discard_pending_preloads(self) -> None:
        """
        Guarantees no background preload outlives this run.

        Every phase must be listed. A preload left parked holds the worker and blocks every later preload
        in the process -- not just this run's.
        """
        for phase in (
                self._retrieval_phase, self._reranking_phase, self._scrubber_phase,
                self._scoping_phase, self._authoring_phase,
        ):
            try:
                phase.discard_preload()
            except Exception as e:
                logger.error(f"Failed to discard pending preload for {type(phase).__name__}: {e}")

    # ==========================================
    # STAGE A -- SCOPING
    # ==========================================

    def propose_plan(
            self,
            topic: str,
            session_id: Optional[uuid.UUID] = None,
            source_id_filter: Optional[uuid.UUID] = None,
    ) -> Report:
        """
        Two model loads: the embedder for one grounding search, then the report model to propose an outline.

        The grounding search is what keeps scoping honest. Without it the planner would propose an outline
        from the topic alone -- plausible, well-structured, and quite possibly about material this archive
        does not contain.
        """
        logger.info(f"========== Scoping a Deep Research Report for '{topic}' ==========")

        try:
            indexed_sources = self._retrieval_phase.list_indexed_sources()
            if not indexed_sources:
                raise WorkflowStateError(
                    "Cannot research this topic: no ingested media has been indexed yet, so there is "
                    "nothing to research. Ingest some recordings first."
                )

            # Fired before the search runs, so the 5.7 GB load overlaps embedding and SQL.
            self._retrieval_phase.preload_model()
            self._scoping_phase.preload_model()

            grounding, _ = self._retrieval_phase.execute_sweep(
                report_id=uuid.uuid4(),  # no report row exists yet; this id is only for job bookkeeping
                queries=[(topic, SearchQuery(
                    prompt=topic,
                    limit=min(_SCOPING_SAMPLE_SIZE, 20),
                    source_id_filter=source_id_filter,
                ))],
            )
            samples = self._grounding_samples(grounding)

            corpus = CorpusDigest(
                indexed_source_count=len(indexed_sources),
                indexed_chunk_count=sum(s.chunk_count for s in indexed_sources),
                sources=[
                    ReportIndexedSource(
                        source_id=s.source_id, name=s.name, chunk_count=s.chunk_count,
                        first_start_sec=s.first_start_sec, last_end_sec=s.last_end_sec,
                    )
                    for s in indexed_sources
                ],
                sample_digests=samples,
            )

            result = self._scoping_phase.execute([(topic, corpus, session_id)])
            if result.status == PhaseStatus.FAILED:
                raise WorkflowStateError(f"Pipeline halted: scoping failed. Errors: {result.errors}")

            report = self._first_payload(result, ResearchScopingJobResult)
            if report is None:
                raise WorkflowStateError("Pipeline halted: scoping produced no plan.")

            logger.info(
                f"========== Proposed a {len(report.plan.sections)}-section plan for report "
                f"{report.id} =========="
            )
            return report

        except WorkflowStateError:
            raise
        except Exception as e:
            logger.error(f"Critical Deep Research Report scoping crash: {e}")
            raise WorkflowExecutionError(
                f"Critical Deep Research Report scoping crash: {str(e)}", original_exception=e
            ) from e
        finally:
            self._discard_pending_preloads()

    @staticmethod
    def _grounding_samples(result: PhaseBatchResult) -> List[str]:
        """Real excerpts from the corpus, deterministically digested -- no model involved."""
        samples: List[str] = []
        for _, payload in result.items:
            if isinstance(payload, SearchResult):
                samples.extend(build_digest(chunk) for chunk in payload.results[:_SCOPING_SAMPLE_SIZE])
        return samples[:_SCOPING_SAMPLE_SIZE]

    @staticmethod
    def _first_payload(result: PhaseBatchResult, job_result_type):
        for job_result, payload in result.items:
            if isinstance(job_result, job_result_type) and job_result.success and payload is not None:
                return payload
        return None

    # ==========================================
    # STAGE B -- THE RUN
    # ==========================================

    def execute_report(
            self,
            report_id: uuid.UUID,
            approved_plan: ResearchPlan,
            cancel_token: Optional[CancellationToken] = None,
    ) -> Report:
        """
        Deep mode: gather once → optimise the outline → write section by section → render.

        Model timeline: embedder (all sub-questions and the sweep in one batch), reranker (all queries in
        one residency), optionally the scrubber, then the report model held across outline optimisation,
        every section write and the summary.
        """
        return self._run(report_id, approved_plan, cancel_token, max_rounds=1, mode=ReportMode.DEEP)

    def execute_iterative_report(
            self,
            report_id: uuid.UUID,
            approved_plan: ResearchPlan,
            cancel_token: Optional[CancellationToken] = None,
    ) -> Report:
        """
        Exhaustive mode: the same pipeline, but gathering repeats while sections come back unsupported.

        A section with no evidence does not distinguish "the corpus does not cover this" from "the
        sub-question was worded badly". Deep mode cannot tell those apart; this mode re-asks, using
        wording the model can now see in the evidence bank.
        """
        return self._run(
            report_id, approved_plan, cancel_token,
            max_rounds=max(1, self._max_gather_rounds), mode=ReportMode.EXHAUSTIVE,
        )

    def _run(
            self,
            report_id: uuid.UUID,
            approved_plan: ResearchPlan,
            cancel_token: Optional[CancellationToken],
            max_rounds: int,
            mode: ReportMode,
    ) -> Report:
        """
        The gather → assign → (maybe gather again) → write loop shared by both modes.

        The model is acquired once per round, and the final round's residency also writes the report, so a
        one-round run costs exactly what deep mode costs. With `max_rounds=1` the loop body runs once with
        `stop_on_gaps=False`, which is byte-for-byte the original single-pass path.
        """
        logger.info(
            f"========== Starting {mode.value} Deep Research Report {report_id} "
            f"(up to {max_rounds} gathering round(s)) =========="
        )

        try:
            stored = self._report_api.get_report(report_id)
            if stored is None:
                raise WorkflowStateError(f"Pipeline halted: report {report_id} does not exist.")

            plan = validate_and_normalise(stored.plan, approved_plan)
            self._report_api.approve_plan(report_id, plan)
            self._raise_if_cancelled(cancel_token, "before gathering")

            hits: List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]] = []
            queries = [(q.text, SearchQuery(prompt=q.text, limit=min(settings.REPORT_RERANK_TOP_N, 20)))
                       for q in plan.sub_questions]
            swept_total = 0
            rounds_used = 0

            for round_index in range(1, max_rounds + 1):
                try:
                    rounds_used = round_index

                    # The corpus-wide sweep runs in round 1 only: it is topic-wide, so repeating it finds
                    # the same sources. Later rounds search the follow-up queries alone.
                    sweep = self._topic_sweep(plan.topic) if round_index == 1 else None

                    round_hits, swept = self._gather_round(report_id, plan, queries, sweep, cancel_token)
                    before = len(hits)
                    # Deduped across ALL rounds, not per round: `research_report_evidence` carries a unique
                    # constraint on (report_id, chunk_id), so a round-1 chunk re-found in round 2 would
                    # raise an IntegrityError and kill the run.
                    hits = dedupe_hits(hits + round_hits)
                    added = len(hits) - before
                    # Round 1's value is the real coverage claim; later rounds run no sweep and report 0.
                    swept_total = max(swept_total, swept)

                    logger.info(
                        f"Report {report_id} round {round_index}: {added} new chunk(s), "
                        f"{len(hits)} in the bank."
                    )

                    if round_index > 1 and added == 0:
                        # Nothing new to assign; write with what we have rather than spending another
                        # assignment call to reach the same conclusion. _write_final fires its own
                        # preload, so none is wasted by firing one here first.
                        logger.info(f"Round {round_index} added nothing new; writing.")
                        return self._write_final(
                            report_id, plan, hits, swept_total, mode, rounds_used, cancel_token
                        )

                    # Queued before the banking SQL: trigger_preload parks until the reranker releases,
                    # so the heavy load runs during neighbour expansion and the bank's writes rather
                    # than starting cold afterwards.
                    self._authoring_phase.preload_model()
                    evidence, chunk_lookup = self._bank(report_id, hits, swept_total, mode, rounds_used)
                    self._raise_if_cancelled(cancel_token, f"before round {round_index} assignment")

                    is_last_round = round_index == max_rounds
                    result = self._author(
                        report_id, plan, evidence, chunk_lookup,
                        stop_on_gaps=not is_last_round, cancel_token=cancel_token,
                    )

                    if self._authoring_phase.was_cancelled(result):
                        return self._finish_cancelled(report_id)

                    if not self._authoring_phase.stopped_for_gaps(result):
                        return self._complete(report_id, plan, result)

                    # By here `_author` has resolved and unloaded the preload fired above, so the
                    # paths below start from a clean slot.
                    follow_ups = self._authoring_phase.follow_ups(result)
                    if not follow_ups:
                        logger.info("Sections are unsupported but there is nothing else to search for; writing.")
                        return self._write_final(
                            report_id, plan, hits, swept_total, mode, rounds_used, cancel_token
                        )

                    logger.info(
                        f"Report {report_id}: gathering again for {len(follow_ups)} unsupported section(s)."
                    )
                    queries = [(f.text, SearchQuery(prompt=f.text,
                                                    limit=min(settings.REPORT_RERANK_TOP_N, 20)))
                               for f in follow_ups]

                finally:
                    # Per iteration, not just at the end of the run. A preload fired this round and never
                    # resolved would otherwise be orphaned by the next round's preload_model(), leaving a
                    # model resident that nothing can release -- which parks the next acquisition forever.
                    self._discard_pending_preloads()

            # Budget exhausted with gaps still open: write what the corpus supports.
            return self._write_final(report_id, plan, hits, swept_total, mode, rounds_used, cancel_token)

        except JobCancelled:
            return self._finish_cancelled(report_id)
        except OutlineValidationError:
            # The user's outline is wrong, not the run. Nothing has been approved or gathered yet, so
            # the report stays where it was and they can correct the outline and start again -- marking
            # it FAILED would strand a report over a fixable typo.
            logger.info(f"Report {report_id} was not started: the approved outline is not valid.")
            raise
        except WorkflowStateError as e:
            self._report_api.set_status(report_id, ReportStatus.FAILED, str(e))
            raise
        except Exception as e:
            logger.error(f"Critical Deep Research Report crash: {e}")
            self._report_api.set_status(report_id, ReportStatus.FAILED, str(e))
            raise WorkflowExecutionError(
                f"Critical Deep Research Report crash: {str(e)}", original_exception=e
            ) from e
        finally:
            self._discard_pending_preloads()

    @staticmethod
    def _topic_sweep(topic: str) -> CorpusSweep:
        return CorpusSweep(
            prompt=topic,
            per_source_limit=settings.REPORT_SWEEP_PER_SOURCE,
            max_sources=settings.REPORT_SWEEP_MAX_SOURCES,
            ann_depth=settings.REPORT_SWEEP_ANN_DEPTH,
        )

    def _write_final(
            self,
            report_id: uuid.UUID,
            plan: ResearchPlan,
            hits,
            swept_total: int,
            mode: ReportMode,
            rounds_used: int,
            cancel_token: Optional[CancellationToken],
    ) -> Report:
        """One last acquisition that assigns and writes, with no option to stop for gaps."""
        self._authoring_phase.preload_model()
        evidence, chunk_lookup = self._bank(report_id, hits, swept_total, mode, rounds_used)
        result = self._author(
            report_id, plan, evidence, chunk_lookup, stop_on_gaps=False, cancel_token=cancel_token
        )
        if self._authoring_phase.was_cancelled(result):
            return self._finish_cancelled(report_id)
        return self._complete(report_id, plan, result)

    def _bank(
            self,
            report_id: uuid.UUID,
            hits,
            swept_total: int,
            mode: ReportMode,
            rounds_used: int,
    ):
        """
        Numbers the accumulated hits and persists them.

        Persisted BEFORE each assignment pass, not once at the end: `optimize_outline` saves its
        per-section assignments inside the model call, so a crash mid-round would otherwise leave
        assignments pointing at doc numbers from an older bank.
        """
        with_neighbors = self._add_neighbors(hits)
        source_names = {s.source_id: s.name for s in self._retrieval_phase.list_indexed_sources()}
        evidence = number_bank(with_neighbors, source_names)
        self._report_api.save_evidence_bank(
            report_id, evidence, swept_total,
            scrub_enabled=settings.REPORT_SCRUB_ENABLED, mode=mode, rounds_used=rounds_used,
        )
        chunk_lookup = {chunk.chunk_id: chunk for chunk, _, _ in with_neighbors}
        return evidence, chunk_lookup

    def _author(
            self,
            report_id: uuid.UUID,
            plan: ResearchPlan,
            evidence,
            chunk_lookup,
            stop_on_gaps: bool,
            cancel_token: Optional[CancellationToken],
            pending_keys: Optional[List[str]] = None,
            optimize: bool = True,
    ):
        result = self._authoring_phase.execute_report(
            report_id=report_id,
            plan=plan,
            evidence=evidence,
            chunk_lookup=chunk_lookup,
            pending_keys=pending_keys,
            optimize=optimize,
            stop_on_gaps=stop_on_gaps,
            cancel_token=cancel_token,
        )
        if result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: no section could be written. {result.errors}")
        return result

    def _complete(self, report_id: uuid.UUID, plan: ResearchPlan, result) -> Report:
        """The summary was written on the same residency; render and finalize from the stored report."""
        return self._finalize(report_id, plan, self._authoring_phase.summary(result))

    def resume_report(
            self,
            report_id: uuid.UUID,
            cancel_token: Optional[CancellationToken] = None,
    ) -> Report:
        """
        Finishes a cancelled or failed report from its persisted evidence bank.

        Touches NO retrieval model: the bank is the run's memory and it is already complete, so a crash in
        section six of eight costs two sections and one model load to recover, not a whole re-gather.

        A resumed exhaustive report never gathers again either. Re-gathering would renumber the bank
        underneath sections that are already WRITTEN and whose `cited_doc_numbers` are frozen, silently
        repointing every citation in them.
        """
        logger.info(f"========== Resuming Deep Research Report {report_id} ==========")

        try:
            stored = self._report_api.get_report(report_id)
            if stored is None:
                raise WorkflowStateError(f"Cannot resume: report {report_id} does not exist.")
            if not stored.evidence:
                raise WorkflowStateError(
                    f"Cannot resume report {report_id}: it was cancelled before its evidence was gathered. "
                    f"Start it again instead."
                )

            pending = [s.key for s in stored.sections if s.status == SectionStatus.PENDING]
            if not pending:
                logger.info(f"Report {report_id} has no pending sections; finalizing it as it stands.")
                # No new summary: _finalize keeps the stored one rather than blanking it.
                return self._finalize(report_id, stored.plan, summary="")

            chunk_lookup = self._hydrate_chunks([e.chunk_id for e in stored.evidence])
            self._report_api.set_status(report_id, ReportStatus.WRITING)

            return self._write(
                report_id, stored.plan, stored.evidence, chunk_lookup, cancel_token,
                pending_keys=pending, optimize=False,
            )

        except JobCancelled:
            return self._finish_cancelled(report_id)
        except WorkflowStateError:
            raise
        except Exception as e:
            logger.error(f"Critical Deep Research Report resume crash: {e}")
            raise WorkflowExecutionError(
                f"Critical Deep Research Report resume crash: {str(e)}", original_exception=e
            ) from e
        finally:
            self._discard_pending_preloads()

    # ==========================================
    # GATHERING
    # ==========================================

    def _gather_round(
            self,
            report_id: uuid.UUID,
            plan: ResearchPlan,
            queries: List[Tuple[str, SearchQuery]],
            sweep: Optional[CorpusSweep],
            cancel_token: Optional[CancellationToken],
    ) -> Tuple[List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]], int]:
        """
        One gathering round: one embedder residency, then one reranker residency, then optionally one
        scrubber residency. Returns this round's hits and how many sources the sweep covered.

        Every query goes through a single batched search. Calling the retrieval phase once per query would
        be the same SQL at the cost of one 2.2 GB model load each.
        """
        self._retrieval_phase.preload_model()

        sweep_result, batch = self._retrieval_phase.execute_sweep(
            report_id=report_id,
            queries=queries,
            sweep=sweep,
            candidates=settings.REPORT_CANDIDATES_PER_QUERY,
            cancel_token=cancel_token,
        )
        if sweep_result.status == PhaseStatus.FAILED:
            raise WorkflowStateError(f"Pipeline halted: evidence gathering failed. {sweep_result.errors}")

        hits: List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]] = []

        if batch.results:
            # Fired only once there is something to rerank. A preload on a path that returns early would
            # be orphaned by the next round's preload_model() and could never be released.
            self._reranking_phase.preload_model()
            if settings.REPORT_SCRUB_ENABLED:
                self._scrubber_phase.preload_model()
            ranked = self._rerank(report_id, queries, batch.results)
            for question_text, chunks in ranked:
                hits.extend((chunk, EvidenceOrigin.FAN_OUT, question_text) for chunk in chunks)

        hits.extend((chunk, EvidenceOrigin.SWEEP, None) for chunk in batch.sweep_results)

        deduped = dedupe_hits(hits)
        if settings.REPORT_SCRUB_ENABLED:
            deduped = self._scrub(report_id, plan, deduped, cancel_token)

        return deduped, batch.swept_source_count

    def _rerank(
            self,
            report_id: uuid.UUID,
            queries: List[Tuple[str, SearchQuery]],
            results: List[SearchResult],
    ) -> List[Tuple[str, List[RetrievedChunk]]]:
        """
        Every sub-question's candidate pool reranked under ONE reranker residency.

        With the scrubber out of the gathering loop, this is the quality gate: the cross-encoder's score
        floor does the filtering the scrubber used to, at a fraction of the cost and without its habit of
        discarding the one chunk that answered the question.
        """
        if not results:
            return []

        # The reranking phase reads `search_id`/`success` off whatever job result it is handed, so the
        # per-query bookkeeping is shimmed into the shape it already understands.
        pairs = [
            (
                RagRetrievalJobResult(
                    search_id=uuid.uuid4(), success=True, retrieved_chunks_count=len(result.results)
                ),
                result,
            )
            for result in results
        ]

        rerank_result = self._reranking_phase.execute(pairs, top_n=settings.REPORT_RERANK_TOP_N)
        if rerank_result.status == PhaseStatus.FAILED:
            logger.warning(
                f"Reranking failed for report {report_id}; keeping RRF order. {rerank_result.errors}"
            )
            return [(question, result.results) for (question, _), result in zip(queries, results)]

        # Paired positionally against the ORIGINAL results, not against the successes only: the phase
        # emits one item per input in order, with a None payload where a single query's rerank failed.
        # Filtering to successes first would shift every later query onto the wrong sub-question.
        payloads = [payload for _, payload in rerank_result.items]
        ranked: List[Tuple[str, List[RetrievedChunk]]] = []
        for index, (question, _) in enumerate(queries):
            payload = payloads[index] if index < len(payloads) else None
            if payload is not None:
                ranked.append((question, payload.results))
            elif index < len(results):
                # That one query keeps its RRF order rather than losing its evidence entirely.
                ranked.append((question, results[index].results))
        return ranked

    def _scrub(
            self,
            report_id: uuid.UUID,
            plan: ResearchPlan,
            hits: List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]],
            cancel_token: Optional[CancellationToken],
    ) -> List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]]:
        """
        Optional relevance pass, off by default (REPORT_SCRUB_ENABLED).

        It is one sequential 7B call per chunk and drops the answering chunk for roughly one question in
        nine. For a report, recall dominates precision -- an irrelevant chunk is simply never cited, but a
        dropped one cannot be -- so this runs only when explicitly switched on, and is capped.
        """
        capped = hits[:settings.REPORT_SCRUB_MAX_CHUNKS]
        overflow = hits[settings.REPORT_SCRUB_MAX_CHUNKS:]
        if overflow:
            logger.info(
                f"Scrubbing the first {len(capped)} chunk(s) of report {report_id}; "
                f"{len(overflow)} admitted unscrubbed under REPORT_SCRUB_MAX_CHUNKS."
            )

        job = RagRetrievalJobResult(search_id=uuid.uuid4(), success=True, retrieved_chunks_count=len(capped))
        pool = SearchResult(query=plan.topic, results=[chunk for chunk, _, _ in capped])

        scrub_result = self._scrubber_phase.execute([(job, pool)])
        if scrub_result.status == PhaseStatus.FAILED:
            logger.warning(
                f"Scrubbing failed for report {report_id}; keeping every gathered chunk. {scrub_result.errors}"
            )
            return hits

        survivors = set()
        for job_result, chunks in scrub_result.items:
            if isinstance(job_result, RetrievalScrubbingJobResult) and job_result.success and chunks:
                survivors.update(chunk.chunk_id for chunk in chunks)

        kept = [hit for hit in capped if hit[0].chunk_id in survivors] + overflow
        logger.info(f"Scrubber kept {len(kept)}/{len(hits)} chunk(s) for report {report_id}.")
        return kept

    def _add_neighbors(
            self,
            hits: List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]],
    ) -> List[Tuple[RetrievedChunk, EvidenceOrigin, Optional[str]]]:
        """
        Adjacent transcript around the best hits, so a passage that straddles a chunk boundary is whole.
        Pure SQL, no model.
        """
        if not hits:
            return hits

        chunks = [chunk for chunk, _, _ in hits]
        expanded = self._retrieval_phase.expand_neighbors(chunks)

        known = {chunk.chunk_id for chunk in chunks}
        neighbors = [
            (chunk, EvidenceOrigin.NEIGHBOR, None)
            for chunk in expanded if chunk.chunk_id not in known
        ]
        if neighbors:
            logger.info(f"Added {len(neighbors)} neighbouring chunk(s) to the evidence bank.")
        return hits + neighbors

    def _hydrate_chunks(self, chunk_ids: List[uuid.UUID]) -> Dict[uuid.UUID, RetrievedChunk]:
        """
        The transcripts behind a persisted bank, for a resume. No model, and ids whose chunks were deleted
        by a re-ingest are simply absent -- the writer skips them and the citation still renders from the
        bank's stored provenance.
        """
        chunks = self._retrieval_phase.get_chunks(chunk_ids)
        return {chunk.chunk_id: chunk for chunk in chunks}

    # ==========================================
    # WRITING
    # ==========================================

    def _write(
            self,
            report_id: uuid.UUID,
            plan: ResearchPlan,
            evidence: List[EvidenceEntry],
            chunk_lookup: Dict[uuid.UUID, RetrievedChunk],
            cancel_token: Optional[CancellationToken],
            pending_keys: Optional[List[str]] = None,
            optimize: bool = True,
    ) -> Report:
        """Used by the resume path, which never re-gathers and so never stops for gaps."""
        result = self._author(
            report_id, plan, evidence, chunk_lookup,
            stop_on_gaps=False, cancel_token=cancel_token,
            pending_keys=pending_keys, optimize=optimize,
        )
        if self._authoring_phase.was_cancelled(result):
            return self._finish_cancelled(report_id)
        return self._complete(report_id, plan, result)

    def _finalize(self, report_id: uuid.UUID, plan: ResearchPlan, summary: str) -> Report:
        """
        Renders and completes the report.

        An empty `summary` falls back to whatever is already stored rather than overwriting it: the
        nothing-pending resume path finalizes without writing a new one, and blanking a report's existing
        summary because this particular call produced none would be silent data loss.
        """
        if not summary:
            stored = self._report_api.get_report(report_id)
            summary = stored.executive_summary if stored else ""

        report = self._report_api.render_and_finalize(report_id, summary)
        logger.info(
            f"========== Deep Research Report {report_id} completed: {len(report.written_sections)} "
            f"section(s), {len(report.cited_doc_numbers)} cited excerpt(s) -> {report.markdown_path} =========="
        )
        return report

    def _finish_cancelled(self, report_id: uuid.UUID) -> Report:
        """
        A cancelled report is a partial report, not a lost one.

        The sections already written are cited and valid, so they are rendered to disk and the row is left
        CANCELLED -- which is also what `resume_report` reads to know what is still pending.
        """
        logger.info(f"Deep Research Report {report_id} was cancelled; keeping what was written.")
        self._report_api.set_status(report_id, ReportStatus.CANCELLED, "Cancelled by the user.")
        report, path = self._report_api.render_partial(report_id)
        if path:
            logger.info(f"Partial report written to {path}")
        if report is None:
            raise WorkflowStateError(f"Report {report_id} was cancelled and could not be read back.")
        return report

    @staticmethod
    def _raise_if_cancelled(cancel_token: Optional[CancellationToken], context: str) -> None:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled(context)
