import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { LibrarySource } from "../api/types";
import { useI18n } from "../i18n";
import { useSession } from "../session";

export function LibraryView() {
  const { t } = useI18n();
  const [sources, setSources] = useState<LibrarySource[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.get<{ sources: LibrarySource[] }>("/api/library").then((r) => setSources(r.sources)).catch((e) => setError(e.message));
  }, []);
  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Library")}</h2></div>
      <p className="page-intro">{t("Everything your firm has ingested, which every answer is drawn from.")}</p>
      {error && <p className="banner danger">{error}</p>}
      <div className="card">
        {sources && sources.length === 0 && <p className="muted">{t("Nothing ingested yet.")}</p>}
        {sources && sources.length > 0 && (
          <table className="table compact-rows">
            <thead>
              <tr><th>{t("Name")}</th><th>{t("Type")}</th><th>{t("Language")}</th><th>{t("Added")}</th></tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.source_id}>
                  <td>{s.name}</td>
                  <td>{s.source_type}</td>
                  <td>{s.language ?? ""}</td>
                  <td>{s.created_at ? new Date(s.created_at).toLocaleDateString() : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

export function AccountView() {
  const { t } = useI18n();
  const { me } = useSession();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const save = async () => {
    try {
      await api.post("/auth/password", { current, new: next });
      setMessage({ ok: true, text: t("Password changed.") });
      setCurrent("");
      setNext("");
    } catch (e: any) {
      setMessage({ ok: false, text: e.message });
    }
  };
  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Account")}</h2></div>
      <p className="page-intro">{me.user?.email}{me.firm && me.console === "firm" ? ` · ${me.firm.name}` : ""}</p>
      <form className="card" style={{ maxWidth: 460 }} onSubmit={(e) => { e.preventDefault(); save(); }}>
        <div className="section-label">{t("Change password")}</div>
        <label className="field"><span>{t("Current password")}</span>
          <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
        </label>
        <label className="field"><span>{t("New password")} <span className="hint">{t("At least 10 characters.")}</span></span>
          <input type="password" autoComplete="new-password" minLength={10} value={next} onChange={(e) => setNext(e.target.value)} />
        </label>
        {message && <p className={`banner ${message.ok ? "info" : "danger"}`}>{message.text}</p>}
        <button className="primary" disabled={!current || next.length < 10}>{t("Save")}</button>
      </form>
    </section>
  );
}
