import logging
import uuid
from typing import Any, Dict, List

from app.core.config import settings
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType
from app.core.jobs.service.manager import JobManager
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.media_compaction.service.compaction_api import CompactionAPI
from app.orchestration.domain.entities import PhaseBatchResult, RagIndexingJobResult, StorageCompactionJobResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase

logger = logging.getLogger(__name__)


class StorageCompactionPhase:
    """
    The last step of a recording's ingestion: once a source is indexed, drop what the vault no longer needs (its
    WAV) and swap its original for a compact playback copy, as far as the settings ask. CPU/IO bound, no model.
    Only indexed sources are touched: a source that failed earlier keeps everything a retry reads. A failure
    here never fails the batch; the files simply stay as they were.
    """

    def __init__(self, job_manager: JobManager, audio_extractor_api: AudioExtractorAPI,
                 compaction_api: CompactionAPI):
        self._job_manager = job_manager
        self._audio = audio_extractor_api
        self._compaction = compaction_api

    @staticmethod
    def enabled() -> bool:
        """Off on the appliance (its defaults keep everything); the pipelines then skip the phase entirely."""
        return not settings.KEEP_EXTRACTED_AUDIO or settings.COMPACT_MEDIA_ORIGINALS

    def execute(self, indexing_results: List[RagIndexingJobResult]) -> PhaseBatchResult:
        indexed = [item for item in indexing_results if item.success]
        logger.info(f"Executing Storage Compaction Phase for {len(indexed)} indexed sources...")
        results: List[StorageCompactionJobResult] = []
        errors: List[str] = []

        for item in indexed:
            request = JobRequest(job_type=JobType.STORAGE_COMPACTION, source_id=item.source_id,
                                 input_params={"source_id": str(item.source_id)})
            try:
                summary = self._job_manager.run_job(request, self.compact_source, source_id=item.source_id)
                results.append(StorageCompactionJobResult(source_id=item.source_id, success=True,
                                                          bytes_saved=summary["bytes_saved"]))
            except Exception as e:
                logger.error(f"Storage compaction failed for Source {item.source_id}: {e}")
                results.append(StorageCompactionJobResult(source_id=item.source_id, success=False,
                                                          error_message=str(e)))
                errors.append(f"Failed Storage Compaction for Source {item.source_id}: {e}")

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        status = PhaseStatus.COMPLETED
        if succeeded == 0 and failed > 0:
            status = PhaseStatus.FAILED
        elif succeeded > 0 and failed > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.STORAGE_COMPACTION,
            status=status,
            total_processed=len(indexed),
            successful_count=succeeded,
            failed_count=failed,
            items=results,
            errors=errors,
        )

    def compact_source(self, source_id: uuid.UUID, dry_run: bool = False) -> Dict[str, Any]:
        """
        One source's compaction, as the job row records it. `dry_run` (the backfill's preview) measures the WAV
        and says whether the original would be re-encoded, changing nothing.
        """
        summary: Dict[str, Any] = {"bytes_saved": 0}
        if not settings.KEEP_EXTRACTED_AUDIO:
            freed = self._audio.discard_extracted_audio(source_id, dry_run=dry_run)
            summary["audio_bytes_freed"] = freed
            summary["bytes_saved"] += freed
        if settings.COMPACT_MEDIA_ORIGINALS:
            outcome = self._compaction.compact_source(source_id, dry_run=dry_run)
            summary["original"] = outcome.status.value
            summary["original_bytes"] = outcome.bytes_before
            summary["original_bytes_saved"] = outcome.bytes_saved
            summary["bytes_saved"] += outcome.bytes_saved
            if outcome.reason:
                summary["reason"] = outcome.reason
        return summary
