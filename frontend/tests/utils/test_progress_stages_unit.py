# Canvas: desktop_test_progress_stages_v1
# Path: frontend/tests/utils/test_progress_stages_unit.py

import pytest

from onyx_desktop.domain.enums import IngestMode
from onyx_desktop.utils.progress_stages import (
    JOBS_PER_STAGE,
    STAGES_BY_MODE,
    report_step,
    stage_for,
)

# Every job type the batch pipelines report (app/core/jobs/domain/enums.py), by pipeline.
PIPELINE_JOBS = {
    IngestMode.VIDEO: [
        "directory_scan", "file_ingest", "audio_extraction", "vad", "domain_adaptation", "transcribe",
        "hallucination_correction", "diarize", "sentence_reconstruction", "phonetic_correction",
        "contextual_retrieval", "visual_cue_detection", "frame_extraction", "visual_analysis",
        "visual_hallucination_correction", "visual_context_synthesis", "context_unification",
        "rag_indexing", "clip_extraction",
    ],
    IngestMode.AUDIO: [
        "directory_scan", "file_ingest", "audio_extraction", "vad", "domain_adaptation", "transcribe",
        "hallucination_correction", "diarize", "sentence_reconstruction", "phonetic_correction",
        "contextual_retrieval", "context_unification", "rag_indexing",
    ],
    IngestMode.DOCUMENT: [
        "directory_scan", "file_ingest", "document_conversion", "document_contextualization",
        "context_unification", "rag_indexing",
    ],
    IngestMode.IMAGE: ["directory_scan", "file_ingest", "image_description", "context_unification", "rag_indexing"],
}


class TestStages:
    @pytest.mark.parametrize("mode", list(IngestMode))
    def test_every_job_of_a_pipeline_lands_in_one_of_its_steps(self, mode):
        keys = [stage.key for stage in STAGES_BY_MODE[mode]]
        for job in PIPELINE_JOBS[mode]:
            stage = stage_for(job)
            assert stage is not None, job
            assert stage.key in keys, (mode, job)

    @pytest.mark.parametrize("mode", list(IngestMode))
    def test_the_steps_only_move_forward_in_pipeline_order(self, mode):
        """The pipeline runs its jobs in order, so the step of each job must never go back to an earlier one."""
        keys = [stage.key for stage in STAGES_BY_MODE[mode]]
        positions = [keys.index(stage_for(job).key) for job in PIPELINE_JOBS[mode]]
        # clip_extraction runs beside rag_indexing, both in the last step
        assert positions == sorted(positions)

    @pytest.mark.parametrize("mode", list(IngestMode))
    def test_every_step_has_a_size(self, mode):
        for stage in STAGES_BY_MODE[mode]:
            assert (mode, stage.key) in JOBS_PER_STAGE

    def test_an_unknown_job_is_not_shown(self):
        assert stage_for("some_future_job") is None

    def test_step_labels_do_not_name_internal_jobs(self):
        labels = " ".join(stage.label.lower() for stages in STAGES_BY_MODE.values() for stage in stages)
        for word in ("transcri", "diariz", "vad", "rag", "whisper", "hallucination", "vector", "chunk"):
            assert word not in labels

    def test_every_step_label_is_general(self):
        assert [s.label for s in STAGES_BY_MODE[IngestMode.VIDEO]] == [
            "Preparing files", "Analyzing audio", "Contextualizing", "Analyzing visuals", "Indexing"]


class TestReportSteps:
    @pytest.mark.parametrize("job,detail,expected", [
        ("rag_retrieval", "Searching the corpus (12 queries)", "Gathering sources…"),
        ("research_outline", "Assigning evidence to the outline", "Organizing the findings…"),
        ("research_section", "Writing section 2/5: Background", "Writing section 2 of 5: Background"),
        ("research_summary", "Writing the executive summary", "Finishing the report…"),
        ("something_else", "internal", None),
    ])
    def test_a_report_shows_general_steps(self, job, detail, expected):
        assert report_step(job, detail) == expected
