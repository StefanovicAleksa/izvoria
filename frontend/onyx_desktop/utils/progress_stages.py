# Canvas: desktop_progress_stages_v1
# Path: frontend/onyx_desktop/utils/progress_stages.py

"""
Folds the engine's internal job names into a handful of general steps a person can follow.

The pipeline reports progress per internal job (dozens of them), and naming those on screen describes how the
appliance works in more detail than anyone using it needs. The UI shows only these steps; this module is the one
place that knows which job belongs to which, so the mapping can change without touching a widget.

A job with no step here (a future one, say) is ignored rather than shown under its internal name.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from onyx_desktop.domain.enums import IngestMode


@dataclass(frozen=True)
class Stage:
    key: str
    label: str


PREPARING = Stage("preparing", "Preparing files")
AUDIO = Stage("audio", "Analyzing audio")
DOCUMENTS = Stage("documents", "Reading documents")
IMAGES = Stage("images", "Analyzing images")
CONTEXT = Stage("context", "Contextualizing")
VISUALS = Stage("visuals", "Analyzing visuals")
INDEXING = Stage("indexing", "Indexing")

# The steps each batch goes through, in the order the pipeline runs them.
STAGES_BY_MODE: Dict[IngestMode, Tuple[Stage, ...]] = {
    IngestMode.VIDEO: (PREPARING, AUDIO, CONTEXT, VISUALS, INDEXING),
    IngestMode.AUDIO: (PREPARING, AUDIO, CONTEXT, INDEXING),
    IngestMode.DOCUMENT: (PREPARING, DOCUMENTS, CONTEXT, INDEXING),
    IngestMode.IMAGE: (PREPARING, IMAGES, INDEXING),
}

_STAGE_OF_JOB: Dict[str, Stage] = {
    "directory_scan": PREPARING,
    "file_ingest": PREPARING,
    "audio_extraction": PREPARING,
    "vad": AUDIO,
    "domain_adaptation": AUDIO,
    "transcribe": AUDIO,
    "hallucination_correction": AUDIO,
    "diarize": AUDIO,
    "sentence_reconstruction": AUDIO,
    "phonetic_correction": AUDIO,
    "document_conversion": DOCUMENTS,
    "image_description": IMAGES,
    "contextual_retrieval": CONTEXT,
    "document_contextualization": CONTEXT,
    "visual_cue_detection": VISUALS,
    "frame_extraction": VISUALS,
    "visual_analysis": VISUALS,
    "visual_hallucination_correction": VISUALS,
    "visual_context_synthesis": VISUALS,
    # Merging everything into one record per file runs after the visuals, as the first part of indexing
    "context_unification": INDEXING,
    "clip_extraction": INDEXING,
    "rag_indexing": INDEXING,
}

# The number of per-file jobs in each step, so a step's bar can move while it runs: with N files a step of K
# jobs completes roughly N*K of them. Not every job sees every file (only files with visual cues reach the
# visual jobs), so this is an estimate; a step is marked done outright once the next one starts.
JOBS_PER_STAGE: Dict[Tuple[IngestMode, str], int] = {
    (IngestMode.VIDEO, "preparing"): 2,
    (IngestMode.VIDEO, "audio"): 7,
    (IngestMode.VIDEO, "context"): 1,
    (IngestMode.VIDEO, "visuals"): 5,
    (IngestMode.VIDEO, "indexing"): 3,
    (IngestMode.AUDIO, "preparing"): 2,
    (IngestMode.AUDIO, "audio"): 7,
    (IngestMode.AUDIO, "context"): 1,
    (IngestMode.AUDIO, "indexing"): 2,
    (IngestMode.DOCUMENT, "preparing"): 1,
    (IngestMode.DOCUMENT, "documents"): 1,
    (IngestMode.DOCUMENT, "context"): 1,
    (IngestMode.DOCUMENT, "indexing"): 2,
    (IngestMode.IMAGE, "preparing"): 1,
    (IngestMode.IMAGE, "images"): 1,
    (IngestMode.IMAGE, "indexing"): 2,
}

# Report runs: what a person sees while a report is planned and written.
_REPORT_STEP: Dict[str, str] = {
    "research_scoping": "Planning the report…",
    "rag_retrieval": "Gathering sources…",
    "rag_reranking": "Gathering sources…",
    "retrieval_scrubbing": "Gathering sources…",
    "research_outline": "Organizing the findings…",
    "research_summary": "Finishing the report…",
}


def stage_for(job: str) -> Optional[Stage]:
    return _STAGE_OF_JOB.get(job)


def report_step(job: str, detail: Optional[str]) -> Optional[str]:
    """
    The line shown under a running report's bar. Section writing keeps its own detail ("Writing section 3/8:
    Title"): the titles are the user's own outline, so they say nothing about how the appliance works.
    """
    if job == "research_section":
        return detail or "Writing…"
    return _REPORT_STEP.get(job)
