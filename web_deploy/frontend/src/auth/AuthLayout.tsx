import type { ReactNode } from "react";
import { useI18n } from "../i18n";
import { Toggles } from "../layout/Toggles";

/** Every signed-out page: the ink brand panel of the desktop's sidebar beside a single form. */
export function AuthLayout({ children, footer, wide = false }: { children: ReactNode; footer?: ReactNode; wide?: boolean }) {
  const { t } = useI18n();
  return (
    <div className="auth">
      <aside className="auth-brand">
        <div className="brand">Onyx</div>
        <p className="tagline">{t("Answers from your firm's own material, with a source for every claim.")}</p>
        <ul>
          <li>{t("Answers only from your material: when it isn't there, Onyx says so.")}</li>
          <li>{t("Every answer cites the passage it rests on, so it can be checked in a click.")}</li>
          <li>{t("Recordings, documents and images, in English and Serbian.")}</li>
        </ul>
        <Toggles />
      </aside>
      <main className="auth-main">
        <div className={`auth-card${wide ? " wide" : ""}`}>
          {children}
          {footer && <footer className="auth-footer">{footer}</footer>}
        </div>
      </main>
    </div>
  );
}

/** The hidden field only a bot fills in, and how long the form has been open: both checked by the gateway. */
export function Honeypot({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="honeypot" aria-hidden>
      <label>
        Website
        <input name="website" tabIndex={-1} autoComplete="off" value={value} onChange={(e) => onChange(e.target.value)} />
      </label>
    </div>
  );
}
