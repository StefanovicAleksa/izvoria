# Canvas: desktop_progress_stages_v1
# Path: frontend/izvoria_desktop/utils/progress_stages.py

"""
Folds the engine's internal job names into a handful of general steps a person can follow.

The pipeline reports progress per internal job (dozens of them), and naming those on screen describes how the
appliance works in more detail than anyone using it needs. The UI shows only these steps; this module is the one
place that knows which job belongs to which, so the mapping can change without touching a widget.

A job with no step here (a future one, say) is ignored rather than shown under its internal name.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from izvoria_desktop.domain.enums import IngestMode
from izvoria_desktop.utils.i18n import N_, tr


@dataclass(frozen=True)
class Stage:
    key: str
    label: str


# Labels are English keys, translated where they are shown
DOWNLOADING = Stage("downloading", N_("Downloading videos"))
PREPARING = Stage("preparing", N_("Preparing files"))
AUDIO = Stage("audio", N_("Analyzing audio"))
DOCUMENTS = Stage("documents", N_("Reading documents"))
IMAGES = Stage("images", N_("Analyzing images"))
CONTEXT = Stage("context", N_("Contextualizing"))
VISUALS = Stage("visuals", N_("Analyzing visuals"))
INDEXING = Stage("indexing", N_("Indexing"))

# The steps each batch goes through, in the order the pipeline runs them.
STAGES_BY_MODE: Dict[IngestMode, Tuple[Stage, ...]] = {
    IngestMode.VIDEO: (PREPARING, AUDIO, CONTEXT, VISUALS, INDEXING),
    IngestMode.AUDIO: (PREPARING, AUDIO, CONTEXT, INDEXING),
    IngestMode.DOCUMENT: (PREPARING, DOCUMENTS, CONTEXT, INDEXING),
    IngestMode.IMAGE: (PREPARING, IMAGES, INDEXING),
}

_STAGE_OF_JOB: Dict[str, Stage] = {
    # A YouTube batch's first step, before the pipeline starts (data/youtube_links.py)
    "youtube_download": DOWNLOADING,
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
    # Off on the appliance; the hosted web version's last step
    "storage_compaction": INDEXING,
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
    "research_scoping": N_("Planning the report…"),
    "rag_retrieval": N_("Gathering sources…"),
    "rag_reranking": N_("Gathering sources…"),
    "retrieval_scrubbing": N_("Gathering sources…"),
    "research_outline": N_("Organizing the findings…"),
    "research_summary": N_("Finishing the report…"),
}
# The engine's own line for a section, "Writing section 3/8: Background", re-said in the display language
_SECTION_DETAIL = re.compile(r"^Writing section (\d+)/(\d+):\s*(.*)$", re.S)


def stages_for(mode: IngestMode, from_links: bool = False) -> Tuple[Stage, ...]:
    """A batch's steps; one from YouTube links downloads its videos first."""
    return (DOWNLOADING, *STAGES_BY_MODE[mode]) if from_links else STAGES_BY_MODE[mode]


def stage_for(job: str) -> Optional[Stage]:
    return _STAGE_OF_JOB.get(job)


def report_step(job: str, detail: Optional[str]) -> Optional[str]:
    """
    The line shown under a running report's bar, in the display language. Section writing names the section
    ("Writing section 3 of 8: Title"): the titles are the user's own outline, so they say nothing about how the
    appliance works.
    """
    if job == "research_section":
        match = _SECTION_DETAIL.match(detail or "")
        if match is None:
            return tr("Writing…")
        return tr("Writing section {current} of {total}: {title}",
                  current=int(match.group(1)), total=int(match.group(2)), title=match.group(3))
    step = _REPORT_STEP.get(job)
    return tr(step) if step else None
