// Shapes the gateway and the box API return (web_deploy/api/mapping.py, web_deploy/gateway/app.py).

export type Lang = "en" | "sr";

export interface User {
  id: string;
  firm: string;
  email: string;
  name: string;
  role: "member" | "admin" | "owner";
  disabled?: boolean;
}

/** The firm a browser has signed in as (the owner's console is the firm "admin", with no language of its own). */
export interface FirmRef {
  slug: string;
  name: string;
  default_language?: Lang;
}

/** Where this browser is: no firm yet, a firm but no account, or signed in. */
export type Stage = "firm" | "account" | "signed_in";

export interface Me {
  stage: Stage;
  console: "firm" | "admin" | null;
  firm: FirmRef | null;
  user: User | null;
  languages: Lang[];
}

export interface AdminFirm {
  slug: string;
  name: string;
  default_language: Lang;
  is_demo: boolean;
  disabled: boolean;
  has_password: boolean;
  users: number;
}

export interface InboundRequest {
  id: string;
  kind: "signup" | "support";
  created_at: number;
  email: string;
  name: string;
  firm: string;
  subject: string;
  payload: Record<string, any>;
  language: Lang;
  status: "new" | "handled" | "spam";
  mail_status: "pending" | "sent" | "failed" | "off";
  mail_error: string;
}

export type BoxStateName = "stopped" | "starting" | "gpu_unavailable" | "booting" | "ready" | "stopping" | "error";

export interface BoxStatus {
  state: BoxStateName;
  message: string;
  error: string;
  queued: number;
  ingesting: number;
  reports_running: number;
  /** Dev hardware: ingestion has the whole GPU; chat, search and reports wait until the queue is empty */
  paused?: boolean;
}

export interface Evidence {
  doc_number: number;
  chunk_id: string;
  source_id: string;
  start_sec: number;
  end_sec: number;
  excerpt: string;
  speakers: string[];
  media_kind: "clip" | "full_source" | "document" | "image" | "unresolved";
  has_source: boolean;
  location: string;
  anchor_kind: string;
  page_start: number | null;
  page_end: number | null;
  rank?: number;
  relevance_score?: number;
  rerank_score?: number | null;
}

export interface ReasoningStep {
  iteration: number;
  decision: string;
  interim_analysis: string;
  refined_query: string | null;
  chunks_added: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  outcome?: "answered" | "needs_clarification";
  evidence?: Evidence[];
  reasoning?: ReasoningStep[];
  latency_ms?: number;
  critique_unresolved?: boolean;
  created_at?: string | null;
  pending?: boolean;
  error?: string;
}

export interface ConversationRef {
  session_id: string;
  title: string;
  started_at: string;
  last_at: string;
  turn_count: number;
}

export interface SearchOutcome {
  query: string;
  mode: "standard" | "verified";
  execution_time_ms: number;
  hits: Evidence[];
  saved_at?: string;
  missing_hits?: number;
}

export interface SearchRecordRef {
  record_id: string;
  query: string;
  mode: "standard" | "verified";
  hit_count: number;
  created_at: string;
}

export interface OutlineSection {
  key: string;
  title: string;
  brief: string;
  questions: string[];
}

export interface PlanDraft {
  report_id: string;
  topic: string;
  /** The report's name; "" for a report from before titles, which is shown by its request (topic) */
  title: string;
  scope_note: string;
  sections: OutlineSection[];
  indexed_source_count: number;
  indexed_chunk_count: number;
  expected_job_count: number;
  estimated_minutes: number;
}

export interface ReportDocument {
  report_id: string;
  topic: string;
  /** The report's name; "" for a report from before titles, which is shown by its request (topic) */
  title: string;
  markdown: string;
  status: string;
  mode: string;
  language: Lang | null;
  evidence: Evidence[];
  section_count: number;
  uncovered_count: number;
  rounds_used: number;
  indexed_source_count: number;
  swept_source_count: number;
}

export interface ReportRef {
  report_id: string;
  topic: string;
  /** The report's name; "" for a report from before titles, which is shown by its request (topic) */
  title: string;
  status: string;
  created_at: string | null;
  completed_at: string | null;
  section_count: number;
  evidence_count: number;
}

export interface IngestJob {
  id: string;
  mode: "video" | "audio" | "document" | "image";
  language: Lang;
  file_count: number;
  status: "preparing" | "queued" | "running" | "done" | "failed" | "cancelled";
  position: number | null;
  owner_id: string;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  progress: {
    phase: string;
    state: string;
    detail: string | null;
    steps_completed?: number;
    steps_failed?: number;
    /** Units of work each engine job has finished (completed or failed); a job appears once it starts */
    done_by_phase?: Record<string, number>;
    /** Distinct files the batch has shown so far */
    files?: number;
    /** Wall time (epoch seconds) of the last tick */
    updated_at?: number;
    /** While a YouTube batch downloads: videos fetched so far (fractional while one is under way), of how many */
    done?: number;
    total?: number;
  } | null;
  /** Uploaded by the user, or downloaded from YouTube before ingestion */
  origin?: "upload" | "youtube";
  report: {
    scanned: number;
    succeeded: number;
    failed: number;
    /** Disk freed by keeping only a compact copy of each recording (absent on older jobs) */
    storage_bytes_saved?: number;
    errors?: Record<string, string>;
  } | null;
  error: string | null;
}

export interface LibrarySource {
  source_id: string;
  name: string;
  source_type: string;
  language: string | null;
  size_bytes: number | null;
  created_at: string | null;
}

export interface SourceInfo {
  kind: "video" | "audio" | "pdf" | "text" | "document" | "image";
  clip: boolean;
  variants: ("media" | "excerpt" | "text" | "original")[];
  start_sec: number;
  end_sec: number;
  page_start: number | null;
  page_end: number | null;
  excerpt_start_sec: number | null;
  size_bytes: number | null;
  download_name: string;
}

export interface ProgressEvent {
  phase: string;
  state: string;
  detail: string | null;
}

export interface RunError {
  kind: string;
  message: string;
}

export interface LinkVideo {
  id: string;
  title: string;
  duration: number | null;
}

export interface LinkPreview {
  videos: LinkVideo[];
  problems: { link: string; reason: string }[];
  playlist_hints: string[];
  capped: boolean;
}
