import type { Lang } from "../api/types";
import { LANGUAGE_NAMES, useI18n } from "../i18n";
import { useTheme } from "../theme";

// The desktop's Feather-style strokes (frontend/izvoria_desktop/ui/icons.py)
const SUN = (
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </>
);
const MOON = <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />;

/** The display language and light/dark switches, beside the brand on every page. */
export function Toggles() {
  const { t, lang, setLang } = useI18n();
  const { mode, toggle } = useTheme();
  const next: Lang = lang === "en" ? "sr" : "en";
  return (
    <div className="toggles">
      <button type="button" className="toggle" onClick={() => setLang(next)}
              title={`${t("Display language")}: ${LANGUAGE_NAMES[next]}`} aria-label={t("Display language")}>
        {lang.toUpperCase()}
      </button>
      <button type="button" className="toggle" onClick={toggle}
              title={t(mode === "dark" ? "Light mode" : "Dark mode")} aria-label={t(mode === "dark" ? "Light mode" : "Dark mode")}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"
             strokeLinejoin="round" aria-hidden>
          {mode === "dark" ? SUN : MOON}
        </svg>
      </button>
    </div>
  );
}
