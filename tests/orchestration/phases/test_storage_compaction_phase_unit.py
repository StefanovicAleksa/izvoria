import uuid
from unittest.mock import ANY, Mock

import pytest

from app.core.jobs.domain.enums import JobType
from app.features.media_compaction.domain.entities import CompactionOutcome, CompactionStatus
from app.orchestration.domain.entities import RagIndexingJobResult, StorageCompactionJobResult
from app.orchestration.domain.enums import PhaseStatus, PipelinePhase
from app.orchestration.service.phases import storage_compaction_phase as module
from app.orchestration.service.phases.storage_compaction_phase import StorageCompactionPhase


@pytest.fixture
def web_storage(monkeypatch):
    """The hosted web version's settings: drop the WAV, compact the original."""
    monkeypatch.setattr(module.settings, "KEEP_EXTRACTED_AUDIO", False)
    monkeypatch.setattr(module.settings, "COMPACT_MEDIA_ORIGINALS", True)


class TestStorageCompactionPhase:
    @pytest.fixture
    def job_manager(self):
        manager = Mock()
        manager.run_job.side_effect = lambda request, func, **kwargs: func(**kwargs)
        return manager

    @pytest.fixture
    def audio_api(self):
        api = Mock()
        api.discard_extracted_audio.return_value = 100
        return api

    @pytest.fixture
    def compaction_api(self):
        api = Mock()
        api.compact_source.side_effect = lambda source_id, dry_run=False: CompactionOutcome(
            source_id=source_id, status=CompactionStatus.REPLACED, bytes_before=1000, bytes_after=300)
        return api

    @pytest.fixture
    def phase(self, job_manager, audio_api, compaction_api):
        return StorageCompactionPhase(job_manager=job_manager, audio_extractor_api=audio_api,
                                      compaction_api=compaction_api)

    @pytest.mark.parametrize("keep_audio, compact, enabled", [
        (True, False, False), (False, False, True), (True, True, True), (False, True, True),
    ])
    def test_enabled_only_when_a_setting_asks_for_something(self, monkeypatch, keep_audio, compact, enabled):
        monkeypatch.setattr(module.settings, "KEEP_EXTRACTED_AUDIO", keep_audio)
        monkeypatch.setattr(module.settings, "COMPACT_MEDIA_ORIGINALS", compact)

        assert StorageCompactionPhase.enabled() is enabled

    def test_each_indexed_source_loses_its_wav_and_gets_a_compact_original(self, web_storage, phase, job_manager,
                                                                           audio_api, compaction_api):
        indexed, failed = uuid.uuid4(), uuid.uuid4()

        result = phase.execute([RagIndexingJobResult(source_id=indexed, success=True),
                                RagIndexingJobResult(source_id=failed, success=False)])

        request = job_manager.run_job.call_args.args[0]
        assert (request.job_type, request.source_id) == (JobType.STORAGE_COMPACTION, indexed)
        job_manager.run_job.assert_called_once_with(ANY, phase.compact_source, source_id=indexed)
        audio_api.discard_extracted_audio.assert_called_once_with(indexed, dry_run=False)
        compaction_api.compact_source.assert_called_once_with(indexed, dry_run=False)
        assert result.phase is PipelinePhase.STORAGE_COMPACTION
        assert result.status is PhaseStatus.COMPLETED
        assert (result.total_processed, result.successful_count) == (1, 1)
        assert result.items == [StorageCompactionJobResult(source_id=indexed, success=True, bytes_saved=800)]

    def test_a_failure_is_recorded_without_stopping_the_others(self, web_storage, phase, compaction_api):
        first, second = uuid.uuid4(), uuid.uuid4()
        compaction_api.compact_source.side_effect = [
            RuntimeError("disk full"),
            CompactionOutcome(source_id=second, status=CompactionStatus.ALREADY_COMPACT),
        ]

        result = phase.execute([RagIndexingJobResult(source_id=first, success=True),
                                RagIndexingJobResult(source_id=second, success=True)])

        assert result.status is PhaseStatus.PARTIAL_SUCCESS
        assert result.items[0].success is False and "disk full" in result.items[0].error_message
        assert result.items[1] == StorageCompactionJobResult(source_id=second, success=True, bytes_saved=100)
        assert len(result.errors) == 1

    def test_only_what_the_settings_ask_for_is_done(self, monkeypatch, phase, audio_api, compaction_api):
        monkeypatch.setattr(module.settings, "KEEP_EXTRACTED_AUDIO", True)
        monkeypatch.setattr(module.settings, "COMPACT_MEDIA_ORIGINALS", True)

        summary = phase.compact_source(uuid.uuid4())

        audio_api.discard_extracted_audio.assert_not_called()
        assert summary == {"bytes_saved": 700, "original": "replaced", "original_bytes": 1000,
                           "original_bytes_saved": 700}

    def test_a_dry_run_is_passed_through_and_records_the_pending_original(self, web_storage, phase, audio_api,
                                                                          compaction_api):
        source_id = uuid.uuid4()
        compaction_api.compact_source.side_effect = None
        compaction_api.compact_source.return_value = CompactionOutcome(
            source_id=source_id, status=CompactionStatus.PENDING, bytes_before=5000, bytes_after=5000)

        summary = phase.compact_source(source_id, dry_run=True)

        audio_api.discard_extracted_audio.assert_called_once_with(source_id, dry_run=True)
        compaction_api.compact_source.assert_called_once_with(source_id, dry_run=True)
        assert summary == {"bytes_saved": 100, "audio_bytes_freed": 100, "original": "pending",
                           "original_bytes": 5000, "original_bytes_saved": 0}
