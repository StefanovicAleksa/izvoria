import { Fragment, useEffect, useState } from "react";
import { api, sourceUrl } from "../api/client";
import type { Evidence, SourceInfo } from "../api/types";
import { useI18n } from "../i18n";

const DOC_TAG = /\[Doc (\d+)\]/g;

export function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rest = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${rest}` : `${m}:${rest}`;
}

export function whereLabel(e: Evidence): string {
  if (e.location) return e.location;
  if (e.anchor_kind === "image") return "image";
  return `${formatTime(e.start_sec)}–${formatTime(e.end_sec)}`;
}

/** Answer text with every [Doc N] as a button that shows the excerpt it cites. */
export function CitedText({ text, onCite }: { text: string; onCite: (n: number) => void }) {
  return (
    <div className="cited-text">
      {text.split(/\n{2,}/).map((paragraph, p) => {
        const parts: (string | number)[] = [];
        let last = 0;
        for (const match of paragraph.matchAll(DOC_TAG)) {
          parts.push(paragraph.slice(last, match.index));
          parts.push(Number(match[1]));
          last = (match.index ?? 0) + match[0].length;
        }
        parts.push(paragraph.slice(last));
        return (
          <p key={p}>
            {parts.map((part, i) =>
              typeof part === "number" ? (
                <button key={i} className="cite" onClick={() => onCite(part)} title={`Doc ${part}`}>
                  Doc {part}
                </button>
              ) : (
                <Fragment key={i}>{part}</Fragment>
              ),
            )}
          </p>
        );
      })}
    </div>
  );
}

export function EvidenceCard({ evidence, highlighted, selected, onOpen }: {
  evidence: Evidence;
  highlighted?: boolean;
  selected?: boolean;
  onOpen?: (e: Evidence) => void;
}) {
  const { t } = useI18n();
  const classes = ["evidence", highlighted && "highlighted", selected && "selected"].filter(Boolean).join(" ");
  return (
    <article className={classes} id={`doc-${evidence.chunk_id}`}>
      <header>
        <span className="chip">{evidence.rank ? `#${evidence.rank}` : `Doc ${evidence.doc_number}`}</span>
        <span className="where">{whereLabel(evidence)}</span>
        {evidence.speakers.length > 0 && <span className="speakers">{evidence.speakers.join(", ")}</span>}
        {evidence.rerank_score != null && <span className="score">{evidence.rerank_score.toFixed(2)}</span>}
      </header>
      <blockquote>{evidence.excerpt}</blockquote>
      {evidence.has_source && onOpen && (
        <footer>
          <button onClick={() => onOpen(evidence)}>{t("Open source")}</button>
        </footer>
      )}
    </article>
  );
}

/**
 * The right-hand "Source review" pane of the desktop app: the cited passage of a recording, the page of a PDF, the
 * lines of a text, an image. On a narrower screen it slides over the view while a source is open.
 */
export function SourcePanel({ evidence, onClose }: { evidence: Evidence | null; onClose: () => void }) {
  const { t } = useI18n();
  return (
    <>
      <aside className={`pane pane-source source-panel${evidence ? " open" : ""}`} aria-label={t("Source review")}>
        <h2>{t("Source review")}</h2>
        <p className="subtle">{t("The passage an answer rests on, with the recording or page it comes from.")}</p>
        {evidence ? (
          <SourceCard key={evidence.chunk_id + evidence.doc_number} evidence={evidence} onClose={onClose} />
        ) : (
          <p className="placeholder">{t("Open a source to review it here.")}</p>
        )}
      </aside>
      <div className={`drawer-backdrop${evidence ? " open" : ""}`} onClick={onClose} />
    </>
  );
}

function SourceCard({ evidence, onClose }: { evidence: Evidence; onClose: () => void }) {
  const { t } = useI18n();
  const [info, setInfo] = useState<SourceInfo | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const label = evidence.rank ? `Result ${evidence.rank}` : `Doc ${evidence.doc_number}`;
  const url = (variant: string) => sourceUrl(evidence.chunk_id, variant, label, evidence.location);

  useEffect(() => {
    if (!evidence.has_source) return;
    const query = new URLSearchParams({ label, location: evidence.location }).toString();
    api
      .get<SourceInfo>(`/api/sources/${evidence.chunk_id}?${query}`)
      .then((loaded) => {
        setInfo(loaded);
        if (loaded.kind === "text" && loaded.variants?.includes("text")) {
          fetch(url("text"), { credentials: "same-origin" })
            .then((r) => r.text())
            .then(setText)
            .catch(() => setText(null));
        }
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evidence.chunk_id]);

  const playable = info && (info.kind === "video" || info.kind === "audio");
  const excerpt = info?.variants?.includes("excerpt");
  const seekTo = info ? (excerpt ? info.start_sec - (info.excerpt_start_sec ?? 0) : info.clip ? 0 : info.start_sec) : 0;

  return (
    <div className="source-card">
      <header>
        <span className="chip">{evidence.rank ? `#${evidence.rank}` : `Doc ${evidence.doc_number}`}</span>
        <h3>{whereLabel(evidence)}</h3>
        <button className="icon" onClick={onClose} aria-label={t("Close")}>×</button>
      </header>
      {evidence.speakers.length > 0 && <p className="subtle">{evidence.speakers.join(", ")}</p>}
      {error && <p className="banner warning">{error}</p>}
      {evidence.has_source && !info && !error && <p className="muted">{t("Loading…")}</p>}
      {playable &&
        (info!.kind === "video" ? (
          <video controls src={url(excerpt ? "excerpt" : "media")}
                 onLoadedMetadata={(e) => (e.currentTarget.currentTime = Math.max(0, seekTo))} />
        ) : (
          <audio controls src={url(excerpt ? "excerpt" : "media")}
                 onLoadedMetadata={(e) => (e.currentTarget.currentTime = Math.max(0, seekTo))} />
        ))}
      {info?.kind === "pdf" && <iframe title={label} className="pdf" src={`${url("media")}#page=${info.page_start ?? 1}`} />}
      {info?.kind === "image" && <img className="source-image" alt={label} src={url("media")} />}
      {info?.kind === "text" && text !== null && (
        <TextLines text={text} first={info.page_start ?? 0} last={info.page_end ?? 0} />
      )}
      <blockquote className="excerpt">{evidence.excerpt}</blockquote>
      {info?.variants?.includes("original") && (
        <a className="button primary" href={url("original")} download={info.download_name}>
          {t("Download")}
        </a>
      )}
    </div>
  );
}

function TextLines({ text, first, last }: { text: string; first: number; last: number }) {
  const lines = text.split("\n");
  useEffect(() => {
    document.getElementById(`line-${first}`)?.scrollIntoView({ block: "center" });
  }, [first]);
  return (
    <pre className="text-lines">
      {lines.map((line, i) => {
        const n = i + 1;
        const cited = n >= first && n <= last;
        return (
          <div key={n} id={`line-${n}`} className={cited ? "cited" : undefined}>
            <span className="line-no">{n}</span>
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}
