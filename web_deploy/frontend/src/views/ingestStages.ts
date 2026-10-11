// Folds the engine's internal job names into the handful of general steps the desktop app shows while ingesting.
// A copy of frontend/izvoria_desktop/utils/progress_stages.py: keep the two in step. A job with no step here (a future
// one, say) is ignored rather than shown under its internal name.

import type { IngestJob } from "../api/types";

type Mode = IngestJob["mode"];

export interface Stage {
  key: string;
  label: string;
}

// A YouTube batch's first step, run before the GPU is taken (web_deploy/ingest_worker/fetch_links.py)
const DOWNLOADING: Stage = { key: "downloading", label: "Downloading videos" };
const PREPARING: Stage = { key: "preparing", label: "Preparing files" };
const AUDIO: Stage = { key: "audio", label: "Analyzing audio" };
const DOCUMENTS: Stage = { key: "documents", label: "Reading documents" };
const IMAGES: Stage = { key: "images", label: "Analyzing images" };
const CONTEXT: Stage = { key: "context", label: "Contextualizing" };
const VISUALS: Stage = { key: "visuals", label: "Analyzing visuals" };
const INDEXING: Stage = { key: "indexing", label: "Indexing" };

// The steps each batch goes through, in the order the pipeline runs them
export const STAGES_BY_MODE: Record<Mode, Stage[]> = {
  video: [PREPARING, AUDIO, CONTEXT, VISUALS, INDEXING],
  audio: [PREPARING, AUDIO, CONTEXT, INDEXING],
  document: [PREPARING, DOCUMENTS, CONTEXT, INDEXING],
  image: [PREPARING, IMAGES, INDEXING],
};

const STAGE_OF_JOB: Record<string, Stage> = {
  downloading: DOWNLOADING,
  // Written by the ingestion worker while the live models hand over the GPU, before the engine starts
  waiting_for_gpu: PREPARING,
  directory_scan: PREPARING,
  file_ingest: PREPARING,
  audio_extraction: PREPARING,
  vad: AUDIO,
  domain_adaptation: AUDIO,
  transcribe: AUDIO,
  hallucination_correction: AUDIO,
  diarize: AUDIO,
  sentence_reconstruction: AUDIO,
  phonetic_correction: AUDIO,
  document_conversion: DOCUMENTS,
  image_description: IMAGES,
  contextual_retrieval: CONTEXT,
  document_contextualization: CONTEXT,
  visual_cue_detection: VISUALS,
  frame_extraction: VISUALS,
  visual_analysis: VISUALS,
  visual_hallucination_correction: VISUALS,
  visual_context_synthesis: VISUALS,
  context_unification: INDEXING,
  clip_extraction: INDEXING,
  rag_indexing: INDEXING,
  storage_compaction: INDEXING,
};

// Per-file jobs in each step, so a step's bar can move while it runs: with N files a step of K jobs completes
// roughly N*K of them. An estimate (only files with visual cues reach the visual jobs); a step is marked done
// outright once the next one starts.
const JOBS_PER_STAGE: Record<Mode, Record<string, number>> = {
  // Indexing: unification, the index and storage compaction (the web version cuts no clips)
  video: { preparing: 2, audio: 7, context: 1, visuals: 5, indexing: 3 },
  audio: { preparing: 2, audio: 7, context: 1, indexing: 3 },
  document: { preparing: 1, documents: 1, context: 1, indexing: 2 },
  image: { preparing: 1, images: 1, indexing: 2 },
};

export type StageState = "waiting" | "active" | "done" | "failed";

export interface StageRow {
  stage: Stage;
  state: StageState;
  /** 0..1, or null for a busy bar (the step's size is not known yet) */
  fraction: number | null;
}

/** Every step of the job's mode, with how far the batch has got: the desktop's StageList, from the job row. */
export function stagesOf(job: IngestJob): Stage[] {
  return job.origin === "youtube" ? [DOWNLOADING, ...STAGES_BY_MODE[job.mode]] : STAGES_BY_MODE[job.mode];
}

export function stageRows(job: IngestJob): StageRow[] {
  const stages = stagesOf(job);
  const keys = stages.map((s) => s.key);
  const progress = job.progress;
  const done = progress?.done_by_phase ?? {};
  const files = progress?.files ?? 0;

  if (job.status === "done") return stages.map((stage) => ({ stage, state: "done", fraction: 1 }));
  if (job.status !== "running" && job.status !== "failed") {
    return stages.map((stage) => ({ stage, state: "waiting", fraction: 0 }));
  }

  // The pipeline only moves forward: the furthest step any job has reached is the running one
  let current = 0;
  for (const phase of [...Object.keys(done), progress?.phase ?? ""]) {
    const stage = STAGE_OF_JOB[phase];
    const index = stage ? keys.indexOf(stage.key) : -1;
    if (index > current) current = index;
  }

  const doneIn = (key: string) =>
    Object.entries(done).reduce((n, [phase, count]) => (STAGE_OF_JOB[phase]?.key === key ? n + count : n), 0);

  return stages.map((stage, index) => {
    if (index < current) return { stage, state: "done", fraction: 1 };
    if (index > current) return { stage, state: "waiting", fraction: 0 };
    if (job.status === "failed") return { stage, state: "failed", fraction: 0 };
    if (stage.key === DOWNLOADING.key) {
      const total = progress?.total ?? 0;
      return { stage, state: "active", fraction: total ? Math.min((progress?.done ?? 0) / total, 0.99) : null };
    }
    if (stage.key === PREPARING.key || files === 0) return { stage, state: "active", fraction: null };
    const expected = files * (JOBS_PER_STAGE[job.mode][stage.key] ?? 1);
    // Never show a running step as full: the estimate can run ahead of the real work
    return { stage, state: "active", fraction: Math.min(Math.max(doneIn(stage.key) / expected, 0), 0.97) };
  });
}

/** `4m 12s`, `1h 03m`: the desktop's formatting.duration. */
export function duration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  if (total < 60) return `${total}s`;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours ? `${hours}h ${String(minutes).padStart(2, "0")}m` : `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

/** Epoch seconds of a server time. One without a zone is UTC (the local queue is SQLite, which drops it). */
export function epochSeconds(iso: string): number {
  const zoned = /([zZ]|[+-]\d\d:?\d\d)$/.test(iso) ? iso : `${iso.replace(" ", "T")}Z`;
  return Date.parse(zoned) / 1000;
}
