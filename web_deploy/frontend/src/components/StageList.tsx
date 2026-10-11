import type { StageRow, StageState } from "../views/ingestStages";
import { useI18n } from "../i18n";

// The desktop's StageList (frontend/izvoria_desktop/ui/widgets/stage_list.py): a mark, the step, its bar, its state
const MARKS: Record<StageState, string> = { waiting: "○", active: "●", done: "✓", failed: "✕" };
const CAPTIONS: Record<StageState, string> = { waiting: "Waiting", active: "In progress", done: "Done", failed: "Stopped" };

export function StageList({ rows }: { rows: StageRow[] }) {
  const { t } = useI18n();
  return (
    <ol className="stage-list">
      {rows.map(({ stage, state, fraction }) => (
        <li key={stage.key} className={`stage ${state}`}>
          <span className="stage-mark" aria-hidden>{MARKS[state]}</span>
          <span className="stage-label">{t(stage.label)}</span>
          <span
            className={`stage-bar${fraction === null ? " busy" : ""}`}
            role="progressbar"
            aria-label={t(stage.label)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}
          >
            <span style={fraction === null ? undefined : { width: `${fraction * 100}%` }} />
          </span>
          <span className="stage-caption">{t(CAPTIONS[state])}</span>
        </li>
      ))}
    </ol>
  );
}
