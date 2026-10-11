import { useEffect, useRef, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { useNavigate, useParams } from "react-router-dom";
import { api, followRun, runToCompletion } from "../api/client";
import type { Evidence, OutlineSection, PlanDraft, ProgressEvent, ReportDocument, ReportRef } from "../api/types";
import { ProgressBar, ProgressLine } from "../components/Controls";
import { EvidenceCard, SourcePanel } from "../components/Evidence";
import { LANGUAGE_NAMES, useI18n } from "../i18n";
import { useSession } from "../session";

type Stage =
  | { kind: "topic" }
  | { kind: "scoping" }
  | { kind: "plan"; plan: PlanDraft }
  | { kind: "writing"; reportId: string; total: number }
  | { kind: "document"; document: ReportDocument };

const DOC_LINK = /^izvoria:\/\/doc\/(\d+)$/;

/** What a report is called: its title, or (from before titles) the request it was asked with. */
export const reportName = (r: { title: string; topic: string }) => r.title || r.topic;

export function ReportView() {
  const { t } = useI18n();
  const { answerLanguage } = useSession();
  const { reportId } = useParams();
  const navigate = useNavigate();
  const [stage, setStage] = useState<Stage>({ kind: "topic" });
  const [topic, setTopic] = useState("");
  const [mode, setMode] = useState<"deep" | "exhaustive">("deep");
  const [reports, setReports] = useState<ReportRef[]>([]);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Evidence | null>(null);
  const stopFollowing = useRef<(() => void) | null>(null);

  const refresh = () => api.get<{ reports: ReportRef[] }>("/api/reports").then((r) => setReports(r.reports));
  useEffect(() => {
    refresh().catch(() => {});
    return () => stopFollowing.current?.();
  }, []);

  const follow = (runId: string, id: string, total: number) => {
    stopFollowing.current?.();
    setProgress([]);
    setStage({ kind: "writing", reportId: id, total });
    stopFollowing.current = followRun<ReportDocument>(runId, {
      onProgress: (e) => setProgress((p) => [...p, e]),
      onResult: (document) => {
        setStage({ kind: "document", document });
        refresh().catch(() => {});
      },
      onError: (err) => {
        setError(err.message);
        load(id);
      },
    });
  };

  const load = async (id: string) => {
    setError(null);
    try {
      const body = await api.get<{ stage: "plan" | "document"; plan?: PlanDraft; document?: ReportDocument; running?: boolean }>(
        `/api/reports/${id}`,
      );
      if (body.stage === "plan") return setStage({ kind: "plan", plan: body.plan! });
      if (body.running) {
        const { runs } = await api.get<{ runs: { id: string; subject: string | null }[] }>("/api/runs");
        const run = runs.find((r) => r.subject === id);
        if (run) return follow(run.id, id, 0);
      }
      setStage({ kind: "document", document: body.document! });
    } catch (e: any) {
      setError(e.message);
      setStage({ kind: "topic" });
    }
  };

  useEffect(() => {
    setOpen(null);
    if (reportId) load(reportId);
    else setStage({ kind: "topic" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reportId]);

  const propose = async () => {
    if (!topic.trim()) return;
    setError(null);
    setProgress([]);
    setStage({ kind: "scoping" });
    try {
      const plan = await runToCompletion<PlanDraft>("/api/reports/plan", { topic, language: answerLanguage },
                                                    (e) => setProgress((p) => [...p, e]));
      navigate(`/reports/${plan.report_id}`, { replace: true });
      setStage({ kind: "plan", plan });
      refresh().catch(() => {});
    } catch (e: any) {
      setError(e.message);
      setStage({ kind: "topic" });
    }
  };

  const write = async (plan: PlanDraft) => {
    setError(null);
    try {
      const { run_id } = await api.post<{ run_id: string }>(`/api/reports/${plan.report_id}/run`, { mode, plan });
      follow(run_id, plan.report_id, plan.expected_job_count);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const resume = async (id: string) => {
    setError(null);
    try {
      const { run_id } = await api.post<{ run_id: string }>(`/api/reports/${id}/resume`);
      follow(run_id, id, 0);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const stop = (id: string) => api.post(`/api/reports/${id}/cancel`).catch((e) => setError(e.message));

  return (
    <div className="panes">
      <aside className="pane pane-list">
        <h3>{t("Reports")}</h3>
        <button className="primary wide" onClick={() => navigate("/reports")}>{t("New report")}</button>
        <ul className="history">
          {reports.length === 0 && <li className="empty-note">{t("No reports yet.")}</li>}
          {reports.map((r) => (
            <li key={r.report_id} className={r.report_id === reportId ? "active" : undefined}>
              <button onClick={() => navigate(`/reports/${r.report_id}`)} title={r.topic}>
                <span className="title">{reportName(r)}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="pane pane-main">
        <div className="page-header"><h2>{t("Research report")}</h2></div>
        {error && <p className="banner danger">{error}</p>}

        {stage.kind === "topic" && (
          <form onSubmit={(e) => { e.preventDefault(); propose(); }}>
            <p className="page-intro">
              {t("A multi-page report on a topic, written section by section from your material, with every claim cited. You approve the outline before it is written.")}
            </p>
            <label className="field">
              <span>{t("Topic")}</span>
              <textarea value={topic} onChange={(e) => setTopic(e.target.value)} rows={4} maxLength={4000}
                        placeholder={t("What should the report cover?")} />
            </label>
            <div className="row">
              <div className="segmented" title={t("Exhaustive keeps searching while sections lack evidence. Slower.")}>
                {(["deep", "exhaustive"] as const).map((m) => (
                  <button type="button" key={m} className={mode === m ? "active" : undefined} onClick={() => setMode(m)}>
                    {t(m === "deep" ? "Deep" : "Exhaustive")}
                  </button>
                ))}
              </div>
              <span className="subtle">{t("Written in")} {LANGUAGE_NAMES[answerLanguage]}</span>
              <span className="spacer" />
              <button className="primary" disabled={!topic.trim()}>{t("Propose an outline")}</button>
            </div>
          </form>
        )}

        {stage.kind === "scoping" && <ProgressLine events={progress} fallback="Scoping the topic…" />}

        {stage.kind === "plan" && (
          <OutlineEditor plan={stage.plan} mode={mode} onMode={setMode} onWrite={write} />
        )}

        {stage.kind === "writing" && (
          <div className="card">
            <h3>{t("Writing the report…")}</h3>
            {stage.total > 0 && (
              <ProgressBar done={progress.filter((p) => p.state === "completed" || p.state === "failed").length}
                           total={stage.total} />
            )}
            <ProgressLine events={progress} fallback="Writing the report…" />
            <button className="danger-outline" onClick={() => stop(stage.reportId)}>{t("Stop")}</button>
          </div>
        )}

        {stage.kind === "document" && (
          <ReportPage document={stage.document} onOpen={setOpen} onResume={resume}
                      onRenamed={(id) => { load(id); refresh().catch(() => {}); }} onError={setError} />
        )}
      </section>

      <SourcePanel evidence={open} onClose={() => setOpen(null)} />
    </div>
  );
}

function OutlineEditor({ plan, mode, onMode, onWrite }: {
  plan: PlanDraft;
  mode: "deep" | "exhaustive";
  onMode: (m: "deep" | "exhaustive") => void;
  onWrite: (plan: PlanDraft) => void;
}) {
  const { t } = useI18n();
  const [title, setTitle] = useState(plan.title);
  const [sections, setSections] = useState<OutlineSection[]>(plan.sections);
  const update = (i: number, change: Partial<OutlineSection>) =>
    setSections((all) => all.map((s, j) => (j === i ? { ...s, ...change } : s)));

  return (
    <div className="outline">
      <label className="field report-title">
        <span>{t("Title")}</span>
        <input value={title} maxLength={120} onChange={(e) => setTitle(e.target.value)} placeholder={plan.topic} />
      </label>
      <p className="subtle">{t("Request")}: {plan.topic}</p>
      <p className="muted">{plan.scope_note}</p>
      <p className="muted">
        {plan.indexed_source_count} {t("sources indexed")} · {t("Estimated time")}: ~{plan.estimated_minutes} {t("minutes")}
      </p>
      <div className="section-label">{t("Review the outline")}</div>
      <p className="muted">{t("Edit titles, briefs and questions, remove what you don't need, then write the report.")}</p>
      {sections.map((s, i) => (
        <fieldset key={s.key || i} className="section-editor">
          <input value={s.title} onChange={(e) => update(i, { title: e.target.value })} aria-label={t("Name")} />
          <textarea value={s.brief} rows={2} onChange={(e) => update(i, { brief: e.target.value })} />
          <label className="field">
            <span>{t("Questions for this section")}</span>
            <textarea rows={Math.max(2, s.questions.length)} value={s.questions.join("\n")}
                      onChange={(e) => update(i, { questions: e.target.value.split("\n") })} />
          </label>
          <button type="button" className="link danger" onClick={() => setSections((all) => all.filter((_, j) => j !== i))}>
            {t("Remove")}
          </button>
        </fieldset>
      ))}
      <button type="button" onClick={() => setSections((all) => [...all, { key: "", title: "", brief: "", questions: [] }])}>
        + {t("Add section")}
      </button>
      <div className="row" style={{ marginTop: 14 }}>
        <div className="segmented">
          {(["deep", "exhaustive"] as const).map((m) => (
            <button type="button" key={m} className={mode === m ? "active" : undefined} onClick={() => onMode(m)}>
              {t(m === "deep" ? "Deep" : "Exhaustive")}
            </button>
          ))}
        </div>
        <span className="spacer" />
        <button className="primary" disabled={!sections.some((s) => s.title.trim())}
                onClick={() => onWrite({ ...plan, title: title.trim(), sections: sections.filter((s) => s.title.trim()) })}>
          {t("Write the report")}
        </button>
      </div>
    </div>
  );
}

function ReportPage({ document, onOpen, onResume, onRenamed, onError }: {
  document: ReportDocument;
  onOpen: (e: Evidence) => void;
  onResume: (id: string) => void;
  onRenamed: (id: string) => void;
  onError: (message: string) => void;
}) {
  const { t } = useI18n();
  const [focus, setFocus] = useState<number | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const byNumber = new Map(document.evidence.map((e) => [e.doc_number, e]));
  // A run that stopped for any reason (cancelled, failed, or the box went to sleep mid-run) keeps what it wrote
  const resumable = document.status !== "completed";

  const cite = (n: number) => {
    setFocus(n);
    const card = byNumber.get(n);
    if (card?.has_source) onOpen(card);
    if (card) window.document.getElementById(`doc-${card.chunk_id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="report-page">
      {document.status !== "completed" && (
        <p className={`banner ${document.status === "failed" ? "danger" : "warning"}`}>
          {t("This report is incomplete:")} {document.status.replace(/_/g, " ")}.
        </p>
      )}
      {renaming !== null && (
        <form className="rename-row" onSubmit={(e) => {
          e.preventDefault();
          api.post(`/api/reports/${document.report_id}/rename`, { title: renaming })
            .then(() => { setRenaming(null); onRenamed(document.report_id); })
            .catch((err) => onError(err.message));
        }}>
          <input value={renaming} maxLength={120} autoFocus aria-label={t("Title")} placeholder={document.topic}
                 onChange={(e) => setRenaming(e.target.value)} />
          <button className="primary">{t("Save")}</button>
          <button type="button" onClick={() => setRenaming(null)}>{t("Cancel")}</button>
        </form>
      )}
      <div className="report-toolbar">
        <span className="subtle">
          {document.language ? LANGUAGE_NAMES[document.language] : ""} · {document.mode} ·{" "}
          {document.swept_source_count}/{document.indexed_source_count} {t("sources indexed")}
        </span>
        <span className="spacer" />
        <a className="button primary" href={`/api/reports/${document.report_id}/pdf`}>{t("Save as PDF")}</a>
        <a className="button" href={`/api/reports/${document.report_id}/markdown`}>{t("Save as Markdown")}</a>
        {renaming === null && <button onClick={() => setRenaming(reportName(document))}>{t("Rename")}</button>}
        {resumable && <button onClick={() => onResume(document.report_id)}>{t("Resume")}</button>}
      </div>
      <article className="markdown">
        <ReactMarkdown
          // Formulas arrive as $$…$$ (the API's math rule); single dollars stay text, so prices are never math
          remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
          rehypePlugins={[[rehypeKatex, { throwOnError: false, output: "htmlAndMathml" }]]}
          urlTransform={(url) => (DOC_LINK.test(url) ? url : defaultUrlTransform(url))}
          components={{
            a: ({ href, children }) => {
              const match = href ? DOC_LINK.exec(href) : null;
              if (match) {
                return (
                  <button className="cite" onClick={() => cite(Number(match[1]))}>
                    {children}
                  </button>
                );
              }
              return <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>;
            },
          }}
        >
          {document.markdown}
        </ReactMarkdown>
      </article>
      <details open className="evidence-list">
        <summary>{t("Evidence")} ({document.evidence.length})</summary>
        {document.evidence.map((e) => (
          <EvidenceCard key={e.doc_number} evidence={e} highlighted={focus === e.doc_number} onOpen={onOpen} />
        ))}
      </details>
    </div>
  );
}
