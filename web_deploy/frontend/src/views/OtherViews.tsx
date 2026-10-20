import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { LibrarySource } from "../api/types";
import { useI18n } from "../i18n";
import { useSession } from "../session";

const DELETE_CONFIRM = "Delete “{name}” and everything made from it (transcript, search index, frames)? Answers and reports that cited it keep their text, but its citations can no longer be opened. This cannot be undone.";

interface ProfileEdit { source_id: string; title: string; topic: string; saving: boolean; }

export function LibraryView() {
  const { t } = useI18n();
  const [sources, setSources] = useState<LibrarySource[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<ProfileEdit | null>(null);
  const { me } = useSession();
  const canDelete = me.user?.role === "admin" || me.user?.role === "owner";
  const save = (edit: ProfileEdit) => {
    setEditing({ ...edit, saving: true });
    api.put<Pick<LibrarySource, "source_id" | "title" | "topic" | "title_edited">>(
      `/api/library/${edit.source_id}/profile`, { title: edit.title, topic: edit.topic })
      .then((saved) => {
        setSources((all) => (all ?? []).map((s) => (s.source_id === saved.source_id ? { ...s, ...saved } : s)));
        setEditing(null);
      })
      .catch((e) => { setError(e.message); setEditing({ ...edit, saving: false }); });
  };
  const remove = (source: LibrarySource) => {
    if (!window.confirm(t(DELETE_CONFIRM).replace("{name}", source.name))) return;
    api.del(`/api/library/${source.source_id}`)
      .then(() => setSources((all) => (all ?? []).filter((s) => s.source_id !== source.source_id)))
      .catch((e) => setError(e.message));
  };
  useEffect(() => {
    api.get<{ sources: LibrarySource[] }>("/api/library").then((r) => setSources(r.sources)).catch((e) => setError(e.message));
  }, []);
  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Library")}</h2></div>
      <p className="page-intro">{t("Everything your firm has ingested, which every answer is drawn from.")}</p>
      <p className="subtle">{t("Each file has a title and a line on what it covers, written when it was ingested. Ask about one by name (\u201cin lecture 4\u201d) and the answer is read from that file alone.")}</p>
      {error && <p className="banner danger">{error}</p>}
      <div className="card">
        {sources && sources.length === 0 && <p className="muted">{t("Nothing ingested yet.")}</p>}
        {sources && sources.length > 0 && (
          <table className="table compact-rows">
            <thead>
              <tr>
                <th>{t("Name")}</th><th>{t("Type")}</th><th>{t("Language")}</th><th>{t("Added")}</th>
                {canDelete && <th />}
              </tr>
            </thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.source_id}>
                  <td>
                    {editing?.source_id === s.source_id ? (
                      <form className="profile-edit" onSubmit={(e) => { e.preventDefault(); if (editing.title.trim()) save(editing); }}>
                        <input value={editing.title} maxLength={120} autoFocus aria-label={t("Title")}
                               onChange={(e) => setEditing({ ...editing, title: e.target.value })} />
                        <textarea value={editing.topic} maxLength={400} rows={2} aria-label={t("What it covers")}
                                  placeholder={t("What it covers")}
                                  onChange={(e) => setEditing({ ...editing, topic: e.target.value })} />
                        <div className="actions">
                          <button className="primary" disabled={editing.saving || !editing.title.trim()}>{t("Save")}</button>
                          <button type="button" onClick={() => setEditing(null)}>{t("Cancel")}</button>
                        </div>
                      </form>
                    ) : (
                      <>
                        <div className="source-title">{s.title || s.name}</div>
                        {s.title && s.title !== s.name && <div className="subtle">{s.name}</div>}
                        {s.topic && <div className="subtle source-topic">{s.topic}</div>}
                      </>
                    )}
                  </td>
                  <td>{s.source_type}</td>
                  <td>{s.language ?? ""}</td>
                  <td>{s.created_at ? new Date(s.created_at).toLocaleDateString() : ""}</td>
                  {canDelete && (
                    <td className="row-actions">
                      {editing?.source_id !== s.source_id && (
                        <button className="icon" aria-label={t("Rename")} title={t("Rename")}
                                onClick={() => setEditing({ source_id: s.source_id, title: s.title || s.name,
                                                            topic: s.topic ?? "", saving: false })}>✎</button>
                      )}
                      <button className="icon" aria-label={t("Delete")} title={t("Delete")} onClick={() => remove(s)}>×</button>
                    </td>
                  )}
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
