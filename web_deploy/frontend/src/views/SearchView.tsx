import { useEffect, useState } from "react";
import { api, runToCompletion } from "../api/client";
import type { Evidence, ProgressEvent, SearchOutcome, SearchRecordRef } from "../api/types";
import { ProgressLine } from "../components/Controls";
import { EvidenceCard, SourcePanel } from "../components/Evidence";
import { useI18n } from "../i18n";
import { useSession } from "../session";

export function SearchView() {
  const { t } = useI18n();
  const { answerLanguage } = useSession();
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<"standard" | "verified">("standard");
  const [outcome, setOutcome] = useState<SearchOutcome | null>(null);
  const [history, setHistory] = useState<SearchRecordRef[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<Evidence | null>(null);

  const refresh = () => api.get<{ searches: SearchRecordRef[] }>("/api/searches").then((r) => setHistory(r.searches));
  useEffect(() => {
    refresh().catch(() => {});
  }, []);

  const run = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    setProgress([]);
    setOpen(null);
    try {
      setOutcome(
        await runToCompletion<SearchOutcome>("/api/search", { prompt, mode, language: answerLanguage, limit: 10 },
                                             (e) => setProgress((p) => [...p, e])),
      );
      refresh().catch(() => {});
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const reopen = async (id: string) => {
    setError(null);
    setOpen(null);
    try {
      setOutcome(await api.get<SearchOutcome>(`/api/searches/${id}`));
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="panes">
      <aside className="pane pane-list">
        <h3>{t("Past searches")}</h3>
        <ul className="history">
          {history.length === 0 && <li className="empty-note">{t("No searches yet.")}</li>}
          {history.map((h) => (
            <li key={h.record_id}>
              <button onClick={() => reopen(h.record_id)} title={h.query}>
                <span className="title">{h.query}</span>&nbsp;<span className="faint">({h.hit_count})</span>
              </button>
              <button className="icon" aria-label={t("Delete")}
                      onClick={() => api.del(`/api/searches/${h.record_id}`).then(refresh)}>×</button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="pane pane-main">
        <div className="page-header"><h2>{t("Search")}</h2></div>
        <p className="page-intro">{t("Find the passages in your material that match a question, ranked by relevance.")}</p>
        <form className="search-bar" onSubmit={(e) => { e.preventDefault(); run(); }}>
          <input value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder={t("Search your material…")}
                 maxLength={4000} />
          <div className="segmented" title={t("Verified search checks every hit against your question and drops the ones that don't fit.")}>
            {(["standard", "verified"] as const).map((m) => (
              <button type="button" key={m} className={mode === m ? "active" : undefined} onClick={() => setMode(m)}>
                {t(m === "standard" ? "Standard" : "Verified")}
              </button>
            ))}
          </div>
          <button className="primary" disabled={busy || !prompt.trim()}>{t("Search")}</button>
        </form>
        {busy && <ProgressLine events={progress} fallback="Search" />}
        {error && <p className="banner danger">{error}</p>}
        {outcome && !busy && (
          <div className="results">
            <p className="subtle">
              {outcome.hits.length} {t("results in")} “{outcome.query}” · {outcome.execution_time_ms} ms
            </p>
            {outcome.hits.length === 0 && <p>{t("No results.")}</p>}
            {outcome.hits.map((hit) => (
              <EvidenceCard key={hit.chunk_id} evidence={hit} selected={open?.chunk_id === hit.chunk_id} onOpen={setOpen} />
            ))}
          </div>
        )}
      </section>

      <SourcePanel evidence={open} onClose={() => setOpen(null)} />
    </div>
  );
}
