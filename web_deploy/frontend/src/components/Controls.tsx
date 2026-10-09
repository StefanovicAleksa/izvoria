import type { Lang, ProgressEvent } from "../api/types";
import { LANGUAGE_NAMES, useI18n } from "../i18n";

export function AnswerLanguage({ value, onChange, languages }: {
  value: Lang;
  onChange: (lang: Lang) => void;
  languages: Lang[];
}) {
  const { t } = useI18n();
  return (
    <label className="inline-field" title={t("Answer in")}>
      <span>{t("Answer in")}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as Lang)}>
        {languages.map((code) => (
          <option key={code} value={code}>
            {LANGUAGE_NAMES[code]}
          </option>
        ))}
      </select>
    </label>
  );
}

// The engine's PipelinePhase values (app/orchestration/domain/enums.py) for the live and report phases
const PHASE_NAMES: Record<string, string> = {
  rag_retrieval: "Searching",
  rag_reranking: "Ranking the evidence",
  retrieval_scrubbing: "Checking the evidence",
  qa_generation: "Writing the answer",
  reasoning_turn: "Reasoning",
  research_scoping: "Scoping",
  research_sweep: "Gathering evidence",
  report_outline: "Planning the outline",
  report_section: "Writing a section",
};

export function ProgressLine({ events, fallback }: { events: ProgressEvent[]; fallback: string }) {
  const { t } = useI18n();
  const last = events[events.length - 1];
  const phase = last ? PHASE_NAMES[last.phase] ?? last.phase.replace(/_/g, " ") : null;
  return (
    <div className="progress-line" role="status">
      <span className="spinner" aria-hidden />
      <span>{phase ? `${t(phase)}…` : t(fallback)}</span>
    </div>
  );
}

export function ProgressBar({ done, total }: { done: number; total: number }) {
  const fraction = total > 0 ? Math.min(1, done / total) : 0;
  return (
    <div className="progress-bar" role="progressbar" aria-valuenow={Math.round(fraction * 100)} aria-valuemin={0}
         aria-valuemax={100}>
      <div style={{ width: `${fraction * 100}%` }} />
    </div>
  );
}
