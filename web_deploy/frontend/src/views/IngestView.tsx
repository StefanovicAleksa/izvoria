import { useEffect, useRef, useState, type DragEvent } from "react";
import { api } from "../api/client";
import { forgetPending, pendingBatchFor, rememberPending, retryable, uploadAll } from "../api/uploads";
import type { IngestJob, Lang, LinkPreview } from "../api/types";
import { ProgressBar } from "../components/Controls";
import { StageList } from "../components/StageList";
import { LANGUAGE_NAMES, useI18n } from "../i18n";
import { useSession } from "../session";
import { fromDrop, fromInput, pick, type Picked } from "./folderFiles";
import { duration, epochSeconds, stageRows } from "./ingestStages";

type Mode = IngestJob["mode"];

const ACCEPT: Record<Mode, string> = {
  video: ".mp4,.mkv,.mov,.avi,.webm",
  audio: ".mp3,.wav,.flac,.m4a,.aac,.ogg,.wma,.alac,.aiff,.amr,.opus,.mka",
  document: ".pdf,.docx,.txt,.md,.markdown,.epub,.html,.htm,.rtf,.csv",
  image: ".jpg,.jpeg,.png,.webp,.tif,.tiff,.bmp",
};

const MODE_NAMES: Record<Mode, string> = { video: "Video", audio: "Audio", document: "Documents", image: "Images" };

// As the desktop: after this long without a tick, say the step is still working
const SILENCE_WARN_SECONDS = 90;

const NOTHING: Picked = { files: [], skipped: 0, folder: null };

function megabytes(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

/** 5:07, 1:02:41 */
function clock(seconds: number | null): string {
  if (seconds == null) return "";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rest = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${rest}` : `${m}:${rest}`;
}

interface LinkSettings {
  enabled: boolean;
  max_videos: number;
}

export function IngestView() {
  const { t } = useI18n();
  const { me } = useSession();
  const [mode, setMode] = useState<Mode>("document");
  const [language, setLanguage] = useState<Lang>(me.firm?.default_language ?? "en");
  const [picked, setPicked] = useState<Picked>(NOTHING);
  const [uploaded, setUploaded] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [links, setLinks] = useState<LinkSettings>({ enabled: false, max_videos: 0 });
  const folderInput = useRef<HTMLInputElement>(null);

  const refresh = () => api.get<{ jobs: IngestJob[]; links?: LinkSettings }>("/api/ingest/jobs").then((r) => {
    setJobs(r.jobs);
    if (r.links) setLinks(r.links);
  });

  useEffect(() => {
    refresh().catch(() => {});
    const timer = window.setInterval(() => refresh().catch(() => {}), 5000);
    return () => window.clearInterval(timer);
  }, []);

  // React has no prop for a folder picker; the browser attribute is set directly
  useEffect(() => {
    folderInput.current?.setAttribute("webkitdirectory", "");
    folderInput.current?.setAttribute("directory", "");
  }, []);

  const choose = (entries: { file: File; path: string }[]) => setPicked(pick(entries, ACCEPT[mode]));

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (busy) return;
    choose(await fromDrop(e.dataTransfer));
  };

  const submit = async () => {
    if (!picked.files.length || busy) return;
    setBusy(true);
    setError(null);
    try {
      // The same files picked again after a broken-off upload continue it, from the pieces already sent
      const batch = pendingBatchFor(mode, language, picked.files)
        ?? (await api.post<{ batch: string }>("/api/uploads")).batch;
      rememberPending(batch, mode, language, picked.files);
      await uploadAll(batch, picked.files, (name, fraction) => setUploaded((u) => ({ ...u, [name]: fraction })));
      await api.post("/api/ingest", { batch, mode, language });
      forgetPending();
      setPicked(NOTHING);
      setUploaded({});
      await refresh();
    } catch (e: any) {
      // A dropped connection can be resumed; a refused file can't
      setError(retryable(e) ? `${e.message} ${t("Choose the same files again to continue where it stopped.")}` : e.message);
    } finally {
      setBusy(false);
    }
  };

  const files = picked.files;
  const totalBytes = files.reduce((n, f) => n + f.file.size, 0);
  const sentBytes = files.reduce((n, f) => n + (uploaded[f.name] ?? 0) * f.file.size, 0);

  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Ingest")}</h2></div>
      <p className="page-intro">
        {t("Add recordings, documents and images to your firm's material. Once ingested they can be searched, asked about and cited.")}
      </p>
      <div className="card">
        <div className="section-label">{t("Upload files")}</div>
        <div className="row ingest-options">
          <label className="inline-field">
            <span>{t("Kind of files")}</span>
            <select value={mode} disabled={busy} onChange={(e) => { setMode(e.target.value as Mode); setPicked(NOTHING); }}>
              {(Object.keys(MODE_NAMES) as Mode[]).map((m) => (
                <option key={m} value={m}>{t(MODE_NAMES[m])}</option>
              ))}
            </select>
          </label>
          <label className="inline-field">
            <span>{t("Language of these files")}</span>
            <select value={language} disabled={busy} onChange={(e) => setLanguage(e.target.value as Lang)}>
              {(me.languages as Lang[]).map((code) => (
                <option key={code} value={code}>{LANGUAGE_NAMES[code]}</option>
              ))}
            </select>
          </label>
        </div>
        <div
          className={`file-drop${dragging ? " dragging" : ""}${busy ? " disabled" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <div className="row file-drop-buttons">
            {/* The browser's own controls print a chosen file's name at full length, so they are hidden */}
            <label className={`button${busy ? " disabled" : ""}`}>
              <input type="file" multiple accept={ACCEPT[mode]} disabled={busy} className="visually-hidden"
                     onChange={(e) => { choose(fromInput(e.target.files)); e.target.value = ""; }} />
              {t("Choose files")}
            </label>
            <label className={`button${busy ? " disabled" : ""}`}>
              <input ref={folderInput} type="file" multiple disabled={busy} className="visually-hidden"
                     data-testid="folder-input"
                     onChange={(e) => { choose(fromInput(e.target.files)); e.target.value = ""; }} />
              {t("Choose folder")}
            </label>
          </div>
          <span className="muted">
            {files.length
              ? `${files.length} ${t("files")}${picked.folder ? ` ${t("from")} ${picked.folder}` : ""} · ${megabytes(totalBytes)}`
              : t("or drop files or a folder here")}
          </span>
          {picked.skipped > 0 && (
            <span className="subtle">{picked.skipped} {t("left out: not this kind of file")}</span>
          )}
        </div>
        {files.length > 0 && (
          <ul className="file-list">
            {files.map((f) => (
              <li key={f.name}>
                <span className="file-name" title={f.name}>{f.name}</span>
                <span className="muted">{megabytes(f.file.size)}</span>
                {!busy && (
                  <button type="button" className="icon" aria-label={t("Remove")} title={t("Remove")}
                          onClick={() => setPicked((p) => ({ ...p, files: p.files.filter((x) => x !== f) }))}>×</button>
                )}
              </li>
            ))}
          </ul>
        )}
        {busy && <><p className="muted">{t("Uploading")}…</p><ProgressBar done={sentBytes} total={totalBytes} /></>}
        {error && <p className="banner danger">{error}</p>}
        <button className="primary" disabled={!files.length || busy} onClick={submit}>{t("Upload and queue")}</button>
      </div>

      {mode === "video" && links.enabled && (
        <YouTubeLinks language={language} maxVideos={links.max_videos} onQueued={() => refresh().catch(() => {})} />
      )}

      <div className="card">
        <div className="section-label">{t("Ingestion queue")}</div>
        <p className="notice">
          {t("Ingestion runs one batch at a time for the whole appliance. Yours starts when it reaches the front of the line.")}
        </p>
        {jobs.length === 0 && <p>{t("Nothing queued.")}</p>}
        <ul className="jobs">
          {jobs.map((job) => (
            <JobItem key={job.id} job={job}
                     onCancel={() => api.del(`/api/ingest/jobs/${job.id}`).then(refresh).catch((e) => setError(e.message))} />
          ))}
        </ul>
      </div>
    </section>
  );
}

function YouTubeLinks({ language, maxVideos, onQueued }: {
  language: Lang;
  maxVideos: number;
  onQueued: () => void;
}) {
  const { t } = useI18n();
  const [text, setText] = useState("");
  const [preview, setPreview] = useState<LinkPreview | null>(null);
  const [dropped, setDropped] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState<"checking" | "queueing" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const typed = () => text.split("\n").map((line) => line.trim()).filter(Boolean);
  const kept = preview ? preview.videos.filter((v) => !dropped.has(v.id)) : [];
  const seconds = kept.reduce((n, v) => n + (v.duration ?? 0), 0);

  const check = async () => {
    setBusy("checking");
    setError(null);
    try {
      setPreview(await api.post<LinkPreview>("/api/ingest/links/preview", { links: typed(), mode: "video" }));
      setDropped(new Set());
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const queue = async () => {
    setBusy("queueing");
    setError(null);
    try {
      await api.post("/api/ingest/links", { links: typed(), mode: "video", language, video_ids: kept.map((v) => v.id) });
      setText("");
      setPreview(null);
      onQueued();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="card youtube-links">
      <div className="section-label">{t("Or add from YouTube")}</div>
      <label className="field">
        <span>{t("Paste YouTube links, one per line: videos or whole playlists.")}</span>
        <textarea rows={3} value={text} disabled={busy !== null} aria-label={t("YouTube links")}
                  placeholder="https://www.youtube.com/playlist?list=…"
                  onChange={(e) => { setText(e.target.value); setPreview(null); }} />
      </label>
      <p className="subtle">
        {t("The videos are downloaded (up to 720p) before ingestion starts. Only add videos you have the right to use.")}
      </p>
      {error && <p className="banner danger">{error}</p>}
      {!preview && (
        <button disabled={!typed().length || busy !== null} onClick={check}>
          {busy === "checking" ? `${t("Checking")}…` : t("Check links")}
        </button>
      )}
      {preview && (
        <>
          {preview.problems.map((p) => (
            <p key={p.link} className="banner warning">
              <span className="file-name" title={p.link}>{p.link}</span> — {p.reason}
            </p>
          ))}
          {preview.playlist_hints.length > 0 && (
            <p className="subtle">{t("A link to a video inside a playlist adds only that video. Paste the playlist's own link to add all of it.")}</p>
          )}
          {preview.capped && <p className="subtle">{t("Only the first {n} videos are added at once.").replace("{n}", String(maxVideos))}</p>}
          {preview.videos.length > 0 && (
            <ul className="file-list">
              {preview.videos.map((v) => (
                <li key={v.id} className={dropped.has(v.id) ? "dropped" : undefined}>
                  <span className="file-name" title={v.title}>{v.title}</span>
                  <span className="muted">{clock(v.duration)}</span>
                  <button type="button" className="icon" disabled={busy !== null}
                          aria-label={dropped.has(v.id) ? t("Add back") : t("Remove")}
                          title={dropped.has(v.id) ? t("Add back") : t("Remove")}
                          onClick={() => setDropped((d) => {
                            const next = new Set(d);
                            if (next.has(v.id)) next.delete(v.id); else next.add(v.id);
                            return next;
                          })}>
                    {dropped.has(v.id) ? "+" : "×"}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <div className="row">
            <button className="primary" disabled={!kept.length || busy !== null} onClick={queue}>
              {busy === "queueing" ? `${t("Adding to the queue")}…`
                : `${t("Queue videos")} (${kept.length}${seconds ? ` · ${clock(seconds)}` : ""})`}
            </button>
            <button className="link" disabled={busy !== null} onClick={() => setPreview(null)}>{t("Edit links")}</button>
          </div>
        </>
      )}
    </div>
  );
}

function JobItem({ job, onCancel }: { job: IngestJob; onCancel: () => void }) {
  const { t } = useI18n();
  const [now, setNow] = useState(() => Date.now() / 1000);
  const running = job.status === "running";

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  const rows = stageRows(job);
  const active = rows.findIndex((r) => r.state === "active" || r.state === "failed");
  const files = job.progress?.files ?? 0;
  const startedAt = job.started_at ? epochSeconds(job.started_at) : null;
  const quietFor = job.progress?.updated_at ? now - job.progress.updated_at : 0;
  const waitingForGpu = running && job.progress?.phase === "waiting_for_gpu";
  const downloading = running && job.progress?.phase === "downloading";

  const step = active >= 0
    ? t("Step {current} of {total}").replace("{current}", String(active + 1)).replace("{total}", String(rows.length))
    : null;

  return (
    <li className={`job ${job.status}`}>
      <div className="job-head">
        <span className="job-what">
          {job.origin === "youtube"
            ? <>{t("YouTube")} · {job.file_count} {t("videos")}</>
            : <>{t(MODE_NAMES[job.mode])} · {job.file_count} {t("files")}</>} · {LANGUAGE_NAMES[job.language]}
        </span>
        <span className={`status ${job.status}`}>{t(job.status)}</span>
        {job.status === "queued" && job.position != null && (
          <span className="muted">{t("Position in line")}: {job.position}</span>
        )}
        {job.status === "preparing" && (
          <span className="muted">
            {t("Making the 720p copies")}
            {job.progress?.total ? ` · ${job.progress.done ?? 0}/${job.progress.total}` : ""}
            {job.progress?.detail ? ` · ${job.progress.detail}` : ""}
          </span>
        )}
        {job.status === "done" && job.report && (
          <span className="muted">
            {t("{succeeded} of {scanned} added")
              .replace("{succeeded}", String(job.report.succeeded)).replace("{scanned}", String(job.report.scanned))}
            {(job.report.storage_bytes_saved ?? 0) > 0 && (
              <> · {t("{size} of storage freed").replace("{size}", megabytes(job.report.storage_bytes_saved ?? 0))}</>
            )}
          </span>
        )}
        <span className="spacer" />
        {(job.status === "queued" || job.status === "preparing") &&
          <button className="link" onClick={onCancel}>{t("Cancel")}</button>}
      </div>

      {(running || job.status === "failed") && (
        <>
          {step && (
            <p className="subtle job-step">
              {waitingForGpu ? t("Waiting for the GPU…")
                : downloading ? `${step} · ${Math.floor(job.progress?.done ?? 0)}/${job.progress?.total ?? job.file_count}`
                  + (job.progress?.detail ? ` · ${job.progress.detail}` : "")
                : files ? `${step} · ${files} ${t("files")}` : step}
              {startedAt && running && <> · {duration(now - startedAt)}</>}
            </p>
          )}
          <StageList rows={rows} />
          {running && (
            <p className="notice">
              {t("Some steps take considerably longer than others, especially with long recordings or large documents, and a bar may stay in place for several minutes while work continues in the background. This is expected: there is no need to restart the appliance or close the app. Progress will move on as each file is completed.")}
            </p>
          )}
          {running && quietFor >= SILENCE_WARN_SECONDS && startedAt && (
            <p className="subtle">
              {t("Still working — {elapsed} so far. The current step is taking a while, which is normal for larger files.")
                .replace("{elapsed}", duration(now - startedAt))}
            </p>
          )}
        </>
      )}

      {job.status === "failed" && job.error && <p className="banner danger">{job.error}</p>}
      {job.status === "done" && job.report?.errors && Object.keys(job.report.errors).length > 0 && (
        <details className="job-errors">
          <summary>{t("Files that were not added")}</summary>
          <ul>
            {Object.entries(job.report.errors).map(([name, reason]) => (
              <li key={name}><span className="file-name" title={name}>{name}</span> <span className="muted">{reason}</span></li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}
