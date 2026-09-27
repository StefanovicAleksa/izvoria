# Canvas: onyx_engine_v4
# Path: app/engine/service/onyx_engine.py

import logging
import threading
import time
import uuid
import gc
from typing import Dict, List, Optional


from app.core.config import settings
from app.core.database.service.manager import db_manager
from app.core.database.service.migrations import upgrade_to_head
from app.core.model_manager.service.async_model_service import AsyncModelService

from app.engine.domain.interfaces import IOnyxEngine
from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.engine.exceptions.lifecycle import EngineOfflineError, EngineBusyError
from app.engine.utils.error_mapper import map_engine_error
from app.engine.utils.conversation_history import build_conversation_history, history_chunks_newest_first

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import Source, SourceType
from app.features.rag_retrieval.domain.entities import RetrievedChunk, SearchQuery, SearchResult
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.qa_generation.domain.entities import QAResponse, ConversationHistory
from app.features.session_memory.service.session_memory_api import SessionMemoryAPI
from app.features.session_memory.domain.enums import SessionTurnMode
from app.core.jobs.domain.entities import CancellationToken
from app.features.deep_research_report.domain.entities import (
    CorpusDigest, Report, ReportSummary, ResearchPlan, ResearchPlanProposal,
)
from app.features.deep_research_report.domain.enums import ReportStatus
from app.features.deep_research_report.service.report_api import ReportAuthoringAPI

from app.orchestration.service.pipelines.batch_media_pipeline import BatchMediaPipeline
from app.orchestration.service.pipelines.batch_audio_pipeline import BatchAudioPipeline
from app.orchestration.service.pipelines.batch_document_pipeline import BatchDocumentPipeline
from app.orchestration.service.pipelines.batch_image_pipeline import BatchImagePipeline
from app.orchestration.service.pipelines.live_retrieval_pipeline import LiveRetrievalPipeline
from app.orchestration.service.pipelines.live_verified_retrieval_pipeline import LiveVerifiedRetrievalPipeline
from app.orchestration.service.pipelines.live_qa_pipeline import LiveQAPipeline
from app.orchestration.service.pipelines.live_reasoning_pipeline import LiveReasoningPipeline
from app.orchestration.service.pipelines.deep_research_report_pipeline import DeepResearchReportPipeline
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ReasonedChatResponse

logger = logging.getLogger(__name__)


class OnyxEngine(IOnyxEngine):
    """
    The unified Facade for the Onyx AI Appliance.
    Strictly coordinates State Locks and isolates pipelines.
    """

    def __init__(
            self,
            async_model_service: AsyncModelService,
            media_pipeline: BatchMediaPipeline,
            audio_pipeline: BatchAudioPipeline,
            live_pipeline: LiveRetrievalPipeline,
            live_verified_pipeline: LiveVerifiedRetrievalPipeline,
            live_qa_pipeline: LiveQAPipeline,
            storage_api: StorageAPI,
            live_reasoning_pipeline: Optional[LiveReasoningPipeline] = None,
            session_memory_api: Optional[SessionMemoryAPI] = None,
            retrieval_api: Optional[RetrievalAPI] = None,
            research_report_pipeline: Optional[DeepResearchReportPipeline] = None,
            report_api: Optional[ReportAuthoringAPI] = None,
            document_pipeline: Optional[BatchDocumentPipeline] = None,
            image_pipeline: Optional[BatchImagePipeline] = None
    ):
        self._async_service = async_model_service
        self._media_pipeline = media_pipeline
        self._audio_pipeline = audio_pipeline
        self._live_pipeline = live_pipeline
        self._live_verified_pipeline = live_verified_pipeline
        self._live_qa_pipeline = live_qa_pipeline
        self._storage_api = storage_api
        self._live_reasoning_pipeline = live_reasoning_pipeline
        self._session_memory_api = session_memory_api
        self._retrieval_api = retrieval_api
        self._research_report_pipeline = research_report_pipeline
        self._report_api = report_api
        self._document_pipeline = document_pipeline
        self._image_pipeline = image_pipeline

        self._state = EngineState.OFFLINE
        self._state_lock = threading.RLock()
        # Live cancellation tokens, keyed by report id. Guarded by the same lock as the state, and set
        # from whichever thread asked to stop -- never from the one running the report.
        self._research_cancels: Dict[uuid.UUID, CancellationToken] = {}

    # ==========================================
    # LIFECYCLE MANAGEMENT
    # ==========================================

    def start(self) -> None:
        with self._state_lock:
            if self._state != EngineState.OFFLINE:
                return

            self._state = EngineState.BOOTING
            logger.info("Booting Onyx Enterprise AI Engine...")

            try:
                # Creates a fresh database, or brings an existing one to this build's schema without touching its data
                upgrade_to_head(db_manager.get_engine())

                # An index built for another language can't be searched with this one's stemmer
                if self._retrieval_api is not None:
                    self._retrieval_api.verify_index_language()

                self._force_aggressive_garbage_collection()

                self._state = EngineState.ONLINE
                logger.info("Onyx Engine is ONLINE and ready for tasks.")

            except Exception as e:
                self._state = EngineState.ERROR
                raise map_engine_error(e, context="Starting Engine")

    def stop(self) -> None:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                return

            logger.info("Shutting down Onyx Engine...")
            try:
                self._force_aggressive_garbage_collection()

                self._async_service.shutdown()
                db_manager.dispose()

                self._state = EngineState.OFFLINE
                logger.info("Onyx Engine is securely OFFLINE.")
            except Exception as e:
                self._state = EngineState.ERROR
                raise map_engine_error(e, context="Stopping Engine")

    def get_state(self) -> EngineState:
        with self._state_lock:
            return self._state

    # ==========================================
    # VRAM ORCHESTRATION GUARDS
    # ==========================================

    def _prepare_for_ingestion(self) -> None:
        """Locks the engine for batch processing."""
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Cannot ingest media: Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Cannot ingest media: Hardware is currently locked by another ingestion process.")
            if self._state == EngineState.RESEARCHING:
                raise EngineBusyError(
                    "Cannot ingest media: a Deep Research Report run is holding the GPU. Stop it first."
                )
            self._state = EngineState.INGESTING

        logger.info("Locking Engine for batch ingestion...")
        try:
            self._force_aggressive_garbage_collection()
        except Exception as e:
            with self._state_lock:
                self._state = EngineState.ONLINE
            raise map_engine_error(e, context="VRAM Eviction for Ingestion")

    def _cleanup_after_ingestion(self) -> None:
        """Unlocks the engine and clears any lingering pipeline artifacts."""
        try:
            logger.info("Batch ingestion finished. Purging pipeline artifacts from VRAM...")
            self._async_service.release_all()
            self._force_aggressive_garbage_collection()
        except Exception as e:
            logger.error(f"Failed to cleanly purge VRAM: {e}")
        finally:
            with self._state_lock:
                self._state = EngineState.ONLINE

    @staticmethod
    def _force_aggressive_garbage_collection() -> None:
        """Forces the destruction of stale memory objects holding heavy weights."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    # ==========================================
    # BATCH PIPELINE DELEGATION
    # ==========================================

    def ingest_videos(self, target_path: str) -> IngestionReport:
        self._prepare_for_ingestion()
        start_time = time.time()
        errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {"scanned": 0, "success": 0, "failed": 0, "vectors": 0, "clips": 0}

        try:
            results = self._media_pipeline.process_directory(target_path, SourceType.VIDEO)
            self._aggregate_metrics(results, errors, metrics)
        except Exception as e:
            raise map_engine_error(e, context=f"Video Ingestion [{target_path}]")
        finally:
            self._cleanup_after_ingestion()

        return self._build_report(target_path, metrics, errors, time.time() - start_time)

    def ingest_audios(self, target_path: str) -> IngestionReport:
        self._prepare_for_ingestion()
        start_time = time.time()
        errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {"scanned": 0, "success": 0, "failed": 0, "vectors": 0, "clips": 0}

        try:
            results = self._audio_pipeline.process_directory(target_path)
            self._aggregate_metrics(results, errors, metrics)
        except Exception as e:
            raise map_engine_error(e, context=f"Audio Ingestion [{target_path}]")
        finally:
            self._cleanup_after_ingestion()

        return self._build_report(target_path, metrics, errors, time.time() - start_time)

    def ingest_documents(self, target_path: str) -> IngestionReport:
        if self._document_pipeline is None:
            raise EngineOfflineError("Document ingestion is not wired into this engine instance.")
        self._prepare_for_ingestion()
        start_time = time.time()
        errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {"scanned": 0, "success": 0, "failed": 0, "vectors": 0, "clips": 0}

        try:
            results = self._document_pipeline.process_directory(target_path)
            self._aggregate_metrics(results, errors, metrics)
        except Exception as e:
            raise map_engine_error(e, context=f"Document Ingestion [{target_path}]")
        finally:
            self._cleanup_after_ingestion()

        return self._build_report(target_path, metrics, errors, time.time() - start_time)

    def ingest_images(self, target_path: str) -> IngestionReport:
        if self._image_pipeline is None:
            raise EngineOfflineError("Image ingestion is not wired into this engine instance.")
        self._prepare_for_ingestion()
        start_time = time.time()
        errors: Dict[str, str] = {}
        metrics: Dict[str, int] = {"scanned": 0, "success": 0, "failed": 0, "vectors": 0, "clips": 0}

        try:
            results = self._image_pipeline.process_directory(target_path)
            self._aggregate_metrics(results, errors, metrics)
        except Exception as e:
            raise map_engine_error(e, context=f"Image Ingestion [{target_path}]")
        finally:
            self._cleanup_after_ingestion()

        return self._build_report(target_path, metrics, errors, time.time() - start_time)

    # ==========================================
    # LIVE INFERENCE DELEGATION
    # ==========================================

    def search_standard(self, query: SearchQuery) -> SearchResult:
        self._enforce_online_state()
        try:
            search_id = uuid.uuid4()
            result = self._live_pipeline.execute_search(search_id, query)
            self._resolve_physical_paths(result)
            return result
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Standard Search [{query.prompt}]")

    def search_verified(self, query: SearchQuery) -> SearchResult:
        self._enforce_online_state()
        try:
            search_id = uuid.uuid4()
            result = self._live_verified_pipeline.execute_verified_search(search_id, query)
            self._resolve_physical_paths(result)
            return result
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Verified Search [{query.prompt}]")

    def chat(self, query: SearchQuery, session_id: Optional[uuid.UUID] = None) -> QAResponse:
        """
        Without a session_id, a standalone question. With one, a turn of that conversation: the answer model sees
        the earlier turns (chat and Reasoned Answer alike) and their evidence, and this turn is recorded.
        """
        self._enforce_online_state()
        if session_id is not None:
            self._require_conversation_support()

        try:
            search_id = uuid.uuid4()
            if session_id is None:
                result = self._live_qa_pipeline.execute_chat(search_id, query)
                self._resolve_citation_paths(result)
                return result

            turns = self._session_memory_api.get_history(session_id)
            history = self._load_history(turns)
            chat_result = self._live_qa_pipeline.execute_conversation_chat(
                search_id, query, history, self._session_memory_api.format_prior_context(turns)
            )
            self._session_memory_api.record_turn(
                session_id, query.prompt, chat_result.qa_response.generated_answer,
                mode=SessionTurnMode.CHAT,
                standalone_query=chat_result.standalone_query,
                evidence_chunk_ids=chat_result.new_chunk_ids,
                first_doc_number=history.next_doc_number
            )
            self._resolve_citation_paths(chat_result.qa_response)
            return chat_result.qa_response
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Chat QA Generation [{query.prompt}]")

    def chat_reasoned(self, session_id: uuid.UUID, query: SearchQuery) -> ReasonedChatResponse:
        self._enforce_online_state()
        if self._live_reasoning_pipeline is None:
            raise EngineOfflineError("Reasoned Answer is not wired into this engine instance.")
        self._require_conversation_support()

        try:
            search_id = uuid.uuid4()
            turns = self._session_memory_api.get_history(session_id)
            history = self._load_history(turns)
            result = self._live_reasoning_pipeline.execute_reasoning(
                search_id, query,
                prior_session_context=self._session_memory_api.format_prior_context(turns),
                seed_chunks=history_chunks_newest_first(history)
            )

            if result.qa_response is not None:
                self._resolve_citation_paths(result.qa_response)
                answer_text = result.qa_response.generated_answer
            else:
                answer_text = f"[Clarification requested] {result.clarification_question}"

            # Seeded chunks keep the numbers earlier turns gave them; only chunks new to the conversation are numbered
            shown = {n.chunk.chunk_id for turn in history.turns for n in turn.evidence}
            self._session_memory_api.record_turn(
                session_id, query.prompt, answer_text,
                mode=SessionTurnMode.REASONING,
                standalone_query=result.standalone_query,
                evidence_chunk_ids=[c for c in result.evidence_chunk_ids if c not in shown],
                first_doc_number=history.next_doc_number
            )

            return result
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Reasoned Answer [{query.prompt}]")

    # ==========================================
    # DEEP RESEARCH REPORT MODE
    # ==========================================

    def _prepare_for_research(self, report_id: uuid.UUID) -> CancellationToken:
        """
        Locks the engine for a report run and registers its cancellation token.

        Mirrors `_prepare_for_ingestion`: a report holds the GPU for tens of minutes, so live calls are
        rejected with a clear reason rather than silently queueing behind it or fighting it for VRAM.
        """
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Cannot run a research report: Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Cannot run a research report: hardware is locked for batch ingestion.")
            if self._state == EngineState.RESEARCHING:
                raise EngineBusyError("Cannot run a research report: another report is already running.")
            self._state = EngineState.RESEARCHING
            token = CancellationToken()
            self._research_cancels[report_id] = token

        logger.info(f"Locking Engine for Deep Research Report {report_id}...")
        try:
            self._force_aggressive_garbage_collection()
        except Exception as e:
            self._cleanup_after_research(report_id)
            raise map_engine_error(e, context="VRAM Eviction for Research")
        return token

    def _cleanup_after_research(self, report_id: uuid.UUID) -> None:
        """Unlocks the engine and drops the run's token, whatever happened."""
        try:
            logger.info("Research report finished. Purging pipeline artifacts from VRAM...")
            self._async_service.release_all()
            self._force_aggressive_garbage_collection()
        except Exception as e:
            logger.error(f"Failed to cleanly purge VRAM: {e}")
        finally:
            with self._state_lock:
                self._research_cancels.pop(report_id, None)
                self._state = EngineState.ONLINE

    def _require_research_support(self) -> None:
        if self._research_report_pipeline is None or self._report_api is None:
            raise EngineOfflineError("Deep Research Report mode is not wired into this engine instance.")

    def propose_research_plan(
            self,
            topic: str,
            session_id: Optional[uuid.UUID] = None,
            source_id_filter: Optional[uuid.UUID] = None
    ) -> ResearchPlanProposal:
        """
        Scopes a report: searches the corpus once, proposes an outline, and stops for approval.

        Short enough to run as an ordinary live call (about a minute, two model loads). The long run is a
        separate call on purpose -- a misread topic is cheap to fix here and expensive to fix later.
        """
        self._enforce_online_state()
        self._require_research_support()

        try:
            report = self._research_report_pipeline.propose_plan(topic, session_id, source_id_filter)
            sub_questions = len(report.plan.sub_questions)
            sections = len(report.plan.sections)
            return ResearchPlanProposal(
                report_id=report.id,
                plan=report.plan,
                corpus=report.corpus or CorpusDigest(indexed_source_count=report.indexed_source_count,
                                                    indexed_chunk_count=0),
                # The caller needs a denominator up front: unlike ingestion, a report knows how much
                # work it is about to do. Counted against the job types the report view actually ticks
                # on -- per round, one batched search plus ONE RERANK JOB PER SUB-QUESTION plus one
                # outline pass, then one job per section and one summary. The rerank fan-out was the
                # term missing here: with 12 sub-questions the bar reached 100% at the outline pass and
                # sat there through all eight section writes, which are the slowest part of the run.
                # Exhaustive mode's round ceiling is used, so the bar can still only finish early.
                expected_job_count=(
                    settings.REPORT_MAX_GATHER_ROUNDS * (2 + sub_questions) + sections + 1
                ),
                estimated_minutes=self._estimate_minutes(sections, sub_questions),
            )
        except Exception as e:
            self._release_models_after_failure()
            raise map_engine_error(e, context=f"Proposing a research plan for [{topic}]")

    @staticmethod
    def _estimate_minutes(sections: int, sub_questions: int) -> int:
        """
        A rough wall-clock estimate for the UI, from measured shapes on a 3060: gathering is dominated by
        the model loads and the reranker, each section by its prefill plus about 1600 generated tokens.
        Deliberately coarse -- it exists so a person can decide whether to start, not to be accurate.
        """
        rounds = max(1, settings.REPORT_MAX_GATHER_ROUNDS)
        return max(5, (4 + (sub_questions // 4)) * rounds + (sections * 3))

    def execute_iterative_research_report(self, report_id: uuid.UUID, approved_plan: ResearchPlan) -> Report:
        """
        Exhaustive mode: the same run, but gathering repeats while sections come back unsupported.

        Takes the GPU exclusively exactly as the deep mode does; cancel it the same way. Costs one
        embedder load, one reranker load and one report-model load per extra round, and stops early as
        soon as a round turns up nothing new.
        """
        self._require_research_support()
        token = self._prepare_for_research(report_id)

        try:
            return self._research_report_pipeline.execute_iterative_report(report_id, approved_plan, token)
        except Exception as e:
            raise map_engine_error(e, context=f"Running exhaustive research report [{report_id}]")
        finally:
            self._cleanup_after_research(report_id)

    def execute_research_report(self, report_id: uuid.UUID, approved_plan: ResearchPlan) -> Report:
        """
        The long run. Holds the GPU exclusively; cancel it from another thread with
        `cancel_research_report`, which returns a partial, resumable report rather than raising.
        """
        self._require_research_support()
        token = self._prepare_for_research(report_id)

        try:
            return self._research_report_pipeline.execute_report(report_id, approved_plan, token)
        except Exception as e:
            raise map_engine_error(e, context=f"Running research report [{report_id}]")
        finally:
            self._cleanup_after_research(report_id)

    def resume_research_report(self, report_id: uuid.UUID) -> Report:
        """Writes the sections a cancelled or failed run left pending. Loads no retrieval model."""
        self._require_research_support()
        token = self._prepare_for_research(report_id)

        try:
            return self._research_report_pipeline.resume_report(report_id, token)
        except Exception as e:
            raise map_engine_error(e, context=f"Resuming research report [{report_id}]")
        finally:
            self._cleanup_after_research(report_id)

    def cancel_research_report(self, report_id: uuid.UUID) -> bool:
        """
        Signals a running report to stop, and says whether one was actually running.

        Safe to call from any thread -- it only sets an Event -- which is the point: the thread running the
        report is busy inside a blocking generation call and cannot be asked politely any other way.

        The run stops at the next section boundary, so up to about ninety seconds may pass before it does.
        Everything written by then is kept and the report stays resumable.
        """
        with self._state_lock:
            token = self._research_cancels.get(report_id)

        if token is None:
            logger.info(f"Cancel requested for report {report_id}, which is not currently running.")
            return False

        logger.info(f"Cancelling research report {report_id} after the section in flight.")
        token.cancel()
        return True

    def get_report(self, report_id: uuid.UUID) -> Optional[Report]:
        """A stored report, whatever its state. No model, no engine lock."""
        self._require_research_support()
        try:
            return self._report_api.get_report(report_id)
        except Exception as e:
            raise map_engine_error(e, context=f"Loading research report [{report_id}]")

    def list_reports(self, session_id: Optional[uuid.UUID] = None, limit: int = 50) -> List[ReportSummary]:
        """Stored reports, newest first, optionally scoped to one conversation."""
        self._require_research_support()
        try:
            return self._report_api.list_reports(session_id, limit)
        except Exception as e:
            raise map_engine_error(e, context="Listing research reports")

    # ==========================================
    # READ-SIDE ACCESS (for a UI)
    # ==========================================

    def get_chunks(self, chunk_ids: List[uuid.UUID]) -> List[RetrievedChunk]:
        """
        The verbatim text behind a set of chunk ids, with clip paths resolved.

        A QACitation carries ids and timestamps but no text, so a UI that wants to show the cited
        excerpt beside its clip has to read the chunks back. This is a plain indexed read: no
        model is loaded and no VRAM is budgeted.

        Duplicate ids are collapsed and ids that no longer exist are skipped upstream, so the
        result may be shorter than the request and is not positionally aligned with it.
        """
        self._enforce_online_state()
        # _retrieval_api is optional on __init__; guard it directly rather than reusing
        # _require_conversation_support(), which also demands session memory and would reject
        # this read for the wrong reason.
        if self._retrieval_api is None:
            raise EngineOfflineError("Retrieval is not wired into this engine instance.")

        try:
            chunks = self._retrieval_api.get_chunks(list(chunk_ids))
        except Exception as e:
            raise map_engine_error(e, context="Fetching chunks by id")

        # get_chunks does not resolve physical paths the way a search result does.
        for chunk in chunks:
            try:
                path = self._physical_path(chunk.extracted_file_id, chunk.source_id)
                if path:
                    object.__setattr__(chunk, 'physical_path', path)
            except Exception as e:
                logger.warning(f"Failed to resolve physical path for chunk {chunk.chunk_id}: {e}")
        return chunks

    def list_sources(self) -> List[Source]:
        """Everything ingested into the vault, newest first, each with its file row attached."""
        if self._state == EngineState.OFFLINE:
            raise EngineOfflineError("Engine is offline.")
        try:
            return self._storage_api.list_sources()
        except Exception as e:
            raise map_engine_error(e, context="Listing ingested sources")

    def _require_conversation_support(self) -> None:
        if self._session_memory_api is None or self._retrieval_api is None:
            raise EngineOfflineError("Conversation memory is not wired into this engine instance.")

    def _load_history(self, turns) -> ConversationHistory:
        return build_conversation_history(turns, self._retrieval_api.get_chunks)

    def _release_models_after_failure(self) -> None:
        """A failed live call must not leave a model resident for the next request to wait on."""
        try:
            self._async_service.release_all()
        except Exception as e:
            logger.error(f"Failed to release models after live inference failure: {e}")

    def _enforce_online_state(self) -> None:
        with self._state_lock:
            if self._state == EngineState.OFFLINE:
                raise EngineOfflineError("Engine is offline.")
            if self._state == EngineState.INGESTING:
                raise EngineBusyError("Hardware is locked for batch ingestion. Inference paused.")
            if self._state == EngineState.RESEARCHING:
                raise EngineBusyError(
                    "Hardware is locked by a Deep Research Report run. Chat and search resume when it "
                    "finishes, or when you stop it."
                )

    def _physical_path(self, extracted_file_id: Optional[uuid.UUID], source_id: uuid.UUID) -> Optional[str]:
        """The chunk's extracted clip if there is one, otherwise the ingested source file (e.g. audio-only)."""
        if extracted_file_id:
            return str(self._storage_api.get_file_path(extracted_file_id))
        source = self._storage_api.get_source(source_id)
        return str(source.file.file_path) if source and source.file else None

    def _resolve_physical_paths(self, search_result: SearchResult) -> None:
        """Hydrates chunks with physical file paths before sending to the UI."""
        for chunk in search_result.results:
            try:
                path = self._physical_path(chunk.extracted_file_id, chunk.source_id)
                if path:
                    object.__setattr__(chunk, 'physical_path', path)
            except Exception as e:
                logger.warning(f"Failed to resolve physical path for chunk {chunk.chunk_id}: {e}")

    def _resolve_citation_paths(self, response: QAResponse) -> None:
        """Hydrates citations with the same clip (or source) paths a search result gets, before sending to the UI."""
        for citation in response.citations:
            try:
                path = self._physical_path(citation.extracted_file_id, citation.source_id)
                if path:
                    object.__setattr__(citation, 'physical_path', path)
            except Exception as e:
                logger.warning(f"Failed to resolve physical path for citation {citation.chunk_id}: {e}")

    # ==========================================
    # UTILITIES & CLI
    # ==========================================

    @staticmethod
    def _aggregate_metrics(
            pipeline_results: Dict[PipelinePhase, PhaseBatchResult],
            error_dict: Dict[str, str],
            metrics_dict: Dict[str, int]
    ) -> None:
        if not pipeline_results: return

        if PipelinePhase.SCAN in pipeline_results:
            metrics_dict["scanned"] += pipeline_results[PipelinePhase.SCAN].successful_count
        if PipelinePhase.RAG_INDEXING in pipeline_results:
            rag_res = pipeline_results[PipelinePhase.RAG_INDEXING]
            metrics_dict["success"] += rag_res.successful_count
            metrics_dict["failed"] += rag_res.failed_count
            for item in rag_res.items:
                if getattr(item, "success", False):
                    metrics_dict["vectors"] += getattr(item, "dense_vectors_saved", 0)
        if PipelinePhase.CLIP_EXTRACTION in pipeline_results:
            clip_res = pipeline_results[PipelinePhase.CLIP_EXTRACTION]
            for item in clip_res.items:
                if getattr(item, "success", False):
                    metrics_dict["clips"] += getattr(item, "extracted_clips_count", 0)

        if PipelinePhase.DOCUMENT_CONVERSION in pipeline_results:
            for item in pipeline_results[PipelinePhase.DOCUMENT_CONVERSION].items:
                if not getattr(item, "success", False):
                    continue
                for key, field in (("pages", "pages_total"), ("pages_ocr", "pages_ocr"),
                                   ("pages_ocr_incomplete", "pages_ocr_incomplete"),
                                   ("pages_unreadable", "pages_unreadable")):
                    metrics_dict[key] = metrics_dict.get(key, 0) + getattr(item, field, 0)
                # Not a failure -- the document is indexed -- but text is missing from it, and a person relying
                # on the index has to know which file to check by hand
                if item.pages_ocr_incomplete or item.pages_unreadable:
                    error_dict[f"document_pages_{item.source_id}"] = (
                        f"Source {item.source_id}: {item.pages_unreadable} scanned page(s) could not be read and "
                        f"are not indexed; {item.pages_ocr_incomplete} page(s) lost lines to OCR."
                    )

        if PipelinePhase.IMAGE_DESCRIPTION in pipeline_results:
            for item in pipeline_results[PipelinePhase.IMAGE_DESCRIPTION].items:
                if not getattr(item, "success", False):
                    continue
                metrics_dict["images"] = metrics_dict.get("images", 0) + 1
                if not item.ocr_checked:
                    metrics_dict["images_unverified"] = metrics_dict.get("images_unverified", 0) + 1
                dropped = item.text_items_dropped
                metrics_dict["image_items_dropped"] = metrics_dict.get("image_items_dropped", 0) + dropped
                withheld = item.ocr_lines_withheld
                metrics_dict["image_ocr_withheld"] = metrics_dict.get("image_ocr_withheld", 0) + withheld
                # Not a failure -- the image is indexed -- but some of its text is not, and a person relying on the
                # index should know it was left out rather than trusted. Only what actually happened is listed.
                gaps = []
                if dropped:
                    gaps.append(f"{dropped} text item(s) the vision model read were not found by OCR")
                if withheld:
                    gaps.append(f"{withheld} line(s) OCR read with low confidence were not confirmed by the vision "
                                f"model (handwriting?)")
                if item.ocr_incomplete:
                    gaps.append("OCR detected lines it could not read")
                if gaps:
                    error_dict[f"image_text_{item.source_id}"] = (
                        f"Source {item.source_id}: " + "; ".join(gaps) + ". That text is not indexed."
                    )

        for phase, result in pipeline_results.items():
            if result.status in [PhaseStatus.FAILED, PhaseStatus.PARTIAL_SUCCESS]:
                for idx, err_str in enumerate(result.errors):
                    error_dict[f"{phase.value}_error_{idx}"] = err_str

    @staticmethod
    def _build_report(target_path: str, metrics: dict, errors: dict, duration: float) -> IngestionReport:
        return IngestionReport(
            target_path=target_path, total_files_scanned=metrics["scanned"],
            successful_ingestions=metrics["success"], failed_ingestions=metrics["failed"],
            errors=errors, total_duration_seconds=round(duration, 2),
            vectors_created=metrics["vectors"], clips_extracted=metrics["clips"],
            pages_converted=metrics.get("pages", 0), pages_ocr=metrics.get("pages_ocr", 0),
            pages_ocr_incomplete=metrics.get("pages_ocr_incomplete", 0),
            pages_unreadable=metrics.get("pages_unreadable", 0),
            images_described=metrics.get("images", 0),
            images_text_unverified=metrics.get("images_unverified", 0),
            image_text_items_dropped=metrics.get("image_items_dropped", 0),
            image_ocr_lines_withheld=metrics.get("image_ocr_withheld", 0)
        )
