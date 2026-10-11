import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AdminFirm, BoxStatus, InboundRequest, User } from "../api/types";
import { useI18n } from "../i18n";

export type AdminSection = "firms" | "requests" | "server";

/** The owner's console: onboarding firms, the requests that come in from the public pages, the GPU server. */
export function AdminConsole({ section }: { section: AdminSection }) {
  if (section === "requests") return <RequestsSection />;
  if (section === "server") return <ServerSection />;
  return <FirmsSection />;
}

// ---------- firms ----------

const EMPTY_FIRM = { slug: "", name: "", default_language: "en", is_demo: false, firm_password: "" };

function FirmsSection() {
  const { t } = useI18n();
  const [firms, setFirms] = useState<AdminFirm[]>([]);
  const [consoleHasPassword, setConsoleHasPassword] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [settingPassword, setSettingPassword] = useState<string | null>(null);
  const [firmPassword, setFirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [form, setForm] = useState(EMPTY_FIRM);

  const load = () =>
    api.get<{ firms: AdminFirm[]; console_has_password: boolean }>("/admin/api/firms")
      .then((r) => {
        setFirms(r.firms);
        setConsoleHasPassword(r.console_has_password);
      })
      .catch((e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  const act = (promise: Promise<unknown>, done: string | null = null) => {
    setError(null);
    setNotice(null);
    return promise.then(() => {
      if (done) setNotice(done);
      return load();
    }).catch((e) => {
      setError(e.message);
      throw e;
    });
  };

  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Firms")}</h2></div>
      <p className="page-intro">
        {t("Each firm signs in with its sign-in name and a password its people share, then with their own accounts. Give a new firm its first administrator; they manage the rest.")}
      </p>
      {!consoleHasPassword && (
        <p className="banner warning">{t("The console has no firm password yet: set one below, or nobody can sign in to it after signing out.")}</p>
      )}
      {error && <p className="banner danger">{error}</p>}
      {notice && <p className="banner info">{notice}</p>}

      <div className="card">
        <table className="table">
          <thead>
            <tr><th>{t("Firm")}</th><th>{t("Sign-in name")}</th><th>{t("Language")}</th><th>{t("Users")}</th><th /></tr>
          </thead>
          <tbody>
            {firms.map((f) => (
              <tr key={f.slug} className={selected === f.slug ? "active" : undefined}>
                <td>
                  <strong>{f.name}</strong>{" "}
                  {f.is_demo && <span className="badge">{t("Demo")}</span>}{" "}
                  {f.disabled && <span className="badge danger">{t("Disabled")}</span>}{" "}
                  {!f.has_password && <span className="badge warning">{t("No firm password")}</span>}
                </td>
                <td><code>{f.slug}</code></td>
                <td>{f.default_language}</td>
                <td>{f.users}</td>
                <td>
                  {settingPassword === f.slug ? (
                    <form className="actions" onSubmit={(e) => {
                      e.preventDefault();
                      act(api.post(`/admin/api/firms/${f.slug}/password`, { password: firmPassword }), t("Firm password set."))
                        .then(() => setSettingPassword(null)).catch(() => {});
                    }}>
                      <input type="password" autoComplete="new-password" placeholder={t("New firm password")} minLength={10}
                             value={firmPassword} onChange={(e) => setFirmPassword(e.target.value)} autoFocus />
                      <button className="primary" disabled={firmPassword.length < 10}>{t("Save")}</button>
                      <button type="button" onClick={() => setSettingPassword(null)}>{t("Cancel")}</button>
                    </form>
                  ) : (
                    <div className="actions">
                      <button className="link" onClick={() => setSelected(selected === f.slug ? null : f.slug)}>{t("Users")}</button>
                      <button className="link" onClick={() => { setSettingPassword(f.slug); setFirmPassword(""); }}>{t("Firm password")}</button>
                      <button className={`link${f.disabled ? "" : " danger"}`}
                              onClick={() => act(api.post(`/admin/api/firms/${f.slug}/${f.disabled ? "enable" : "disable"}`)).catch(() => {})}>
                        {t(f.disabled ? "Enable" : "Disable")}
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && <FirmUsers slug={selected} onChange={load} />}

      <form className="card" onSubmit={(e) => {
        e.preventDefault();
        act(api.post("/admin/api/firms", form), t("Firm created. Now add its first administrator.")).then(() => {
          setSelected(form.slug.trim().toLowerCase());
          setForm(EMPTY_FIRM);
        }).catch(() => {});
      }}>
        <div className="section-label">{t("New firm")}</div>
        <div className="grid-2">
          <label className="field"><span>{t("Firm name")}</span>
            <input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </label>
          <label className="field"><span>{t("Sign-in name")} <span className="hint">{t("Lowercase letters, digits and hyphens, e.g. petrovic-law.")}</span></span>
            <input required autoCapitalize="none" value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} />
          </label>
          <label className="field"><span>{t("Firm password")} <span className="hint">{t("Shared by everyone at the firm.")}</span></span>
            <input type="password" autoComplete="new-password" required minLength={10} value={form.firm_password}
                   onChange={(e) => setForm({ ...form, firm_password: e.target.value })} />
          </label>
          <label className="field"><span>{t("Default answer language")}</span>
            <select value={form.default_language} onChange={(e) => setForm({ ...form, default_language: e.target.value })}>
              <option value="en">English</option>
              <option value="sr">Srpski</option>
            </select>
          </label>
        </div>
        <label className="check">
          <input type="checkbox" checked={form.is_demo} onChange={(e) => setForm({ ...form, is_demo: e.target.checked })} />
          <span>{t("Demo firm")}</span>
        </label>
        <button className="primary" disabled={!form.slug || !form.name || form.firm_password.length < 10}>{t("Create firm")}</button>
      </form>

      <ConsolePasswordCard onDone={load} />
    </section>
  );
}

const EMPTY_USER = { email: "", name: "", password: "", role: "admin" };

function FirmUsers({ slug, onChange }: { slug: string; onChange: () => void }) {
  const { t } = useI18n();
  const [users, setUsers] = useState<User[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [resetting, setResetting] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [form, setForm] = useState(EMPTY_USER);

  const load = () => api.get<{ users: User[] }>(`/admin/api/firms/${slug}/users`).then((r) => setUsers(r.users)).catch((e) => setError(e.message));
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);
  const act = (promise: Promise<unknown>) => {
    setError(null);
    return promise.then(() => {
      load();
      onChange();
    }).catch((e) => {
      setError(e.message);
      throw e;
    });
  };

  return (
    <div className="card">
      <div className="section-label">{t("Users at")} {slug}</div>
      {error && <p className="banner danger">{error}</p>}
      {users.length === 0 && <p className="muted">{t("No accounts yet. Add the firm's first administrator below.")}</p>}
      {users.length > 0 && (
        <table className="table">
          <tbody>
            {users.map((u) => (
              <tr key={u.id}>
                <td>{u.email} {u.disabled && <span className="badge danger">{t("Disabled")}</span>}</td>
                <td>{u.name}</td>
                <td>
                  <select value={u.role} aria-label={t("Role")}
                          onChange={(e) => act(api.post(`/admin/api/users/${u.id}/role`, { role: e.target.value })).catch(() => {})}>
                    <option value="member">{t("Member")}</option>
                    <option value="admin">{t("Administrator")}</option>
                  </select>
                </td>
                <td>
                  {resetting === u.id ? (
                    <form className="actions" onSubmit={(e) => {
                      e.preventDefault();
                      act(api.post(`/admin/api/users/${u.id}/password`, { password })).then(() => setResetting(null)).catch(() => {});
                    }}>
                      <input type="password" autoComplete="new-password" placeholder={t("New password")} minLength={10}
                             value={password} onChange={(e) => setPassword(e.target.value)} autoFocus />
                      <button className="primary" disabled={password.length < 10}>{t("Save")}</button>
                      <button type="button" onClick={() => setResetting(null)}>{t("Cancel")}</button>
                    </form>
                  ) : (
                    <div className="actions">
                      <button className="link" onClick={() => { setResetting(u.id); setPassword(""); }}>{t("Reset password")}</button>
                      <button className={`link${u.disabled ? "" : " danger"}`}
                              onClick={() => act(api.post(`/admin/api/users/${u.id}/${u.disabled ? "enable" : "disable"}`)).catch(() => {})}>
                        {t(u.disabled ? "Enable" : "Disable")}
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <form className="row" style={{ marginTop: 14 }} onSubmit={(e) => {
        e.preventDefault();
        act(api.post(`/admin/api/firms/${slug}/users`, form)).then(() => setForm(EMPTY_USER)).catch(() => {});
      }}>
        <input type="email" required placeholder={t("Email")} value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
        <input placeholder={t("Name")} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        <input type="password" autoComplete="new-password" required placeholder={t("Initial password")} minLength={10}
               value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
        <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })} aria-label={t("Role")}>
          <option value="admin">{t("Administrator")}</option>
          <option value="member">{t("Member")}</option>
        </select>
        <button className="primary" disabled={!form.email || form.password.length < 10}>{t("Add account")}</button>
      </form>
    </div>
  );
}

function ConsolePasswordCard({ onDone }: { onDone: () => void }) {
  const { t } = useI18n();
  const [mine, setMine] = useState("");
  const [next, setNext] = useState("");
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const save = async () => {
    try {
      await api.post("/admin/api/console-password", { account_password: mine, password: next });
      setMessage({ ok: true, text: t("Console password changed.") });
      setMine("");
      setNext("");
      onDone();
    } catch (e: any) {
      setMessage({ ok: false, text: e.message });
    }
  };
  return (
    <form className="card" onSubmit={(e) => { e.preventDefault(); save(); }}>
      <div className="section-label">{t("Console password")}</div>
      <p className="muted">{t("The firm password of this console: the first sign-in step, as the firm \"admin\".")}</p>
      <div className="grid-2">
        <label className="field"><span>{t("Your own password")}</span>
          <input type="password" autoComplete="current-password" value={mine} onChange={(e) => setMine(e.target.value)} />
        </label>
        <label className="field"><span>{t("New console password")}</span>
          <input type="password" autoComplete="new-password" minLength={10} value={next} onChange={(e) => setNext(e.target.value)} />
        </label>
      </div>
      {message && <p className={`banner ${message.ok ? "info" : "danger"}`}>{message.text}</p>}
      <button className="primary" disabled={!mine || next.length < 10}>{t("Save")}</button>
    </form>
  );
}

// ---------- requests from the public pages ----------

// Labels for a sign-up's answers, in the firm profile template's order
const PROFILE_LABELS: [string, string][] = [
  ["industry", "Field of work"],
  ["description", "What your firm does"],
  ["sub_domains", "Practice areas or specialities"],
  ["data_kinds", "What you will bring to Izvoria"],
  ["mishearings", "Terms transcription already gets wrong"],
];
const LANGUAGE_LABELS: [string, string][] = [
  ["share", "Languages of your material"],
  ["english_terms", "English terms in Serbian speech"],
  ["latin", "Latin"],
];
const NOUN_LABELS: [string, string][] = [
  ["people", "People"],
  ["clients", "Clients, counterparties and institutions"],
  ["products", "Products, funds, drugs, devices or software"],
  ["places", "Places"],
  ["acronyms", "Internal acronyms and in-house terms"],
  ["laws", "Laws, regulations, standards and cases"],
];

function RequestsSection() {
  const { t } = useI18n();
  const [requests, setRequests] = useState<InboundRequest[]>([]);
  const [mailEnabled, setMailEnabled] = useState(true);
  const [kind, setKind] = useState("");
  const [status, setStatus] = useState("new");
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    const query = new URLSearchParams({ kind, status }).toString();
    return api.get<{ requests: InboundRequest[]; mail_enabled: boolean }>(`/admin/api/requests?${query}`)
      .then((r) => {
        setRequests(r.requests);
        setMailEnabled(r.mail_enabled);
      })
      .catch((e) => setError(e.message));
  };
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, status]);
  const act = (promise: Promise<unknown>) => {
    setError(null);
    promise.then(load).catch((e) => setError(e.message));
  };

  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Requests")}</h2></div>
      <p className="page-intro">{t("Firms asking for Izvoria, and help requests, as sent from the public pages. Each is also emailed when email is set up.")}</p>
      {!mailEnabled && <p className="banner warning">{t("Email is not set up on this gateway (SMTP_HOST): requests are only kept here.")}</p>}
      {error && <p className="banner danger">{error}</p>}
      <div className="row" style={{ marginBottom: 14 }}>
        <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label={t("Kind")}>
          <option value="">{t("All requests")}</option>
          <option value="signup">{t("Access requests")}</option>
          <option value="support">{t("Help requests")}</option>
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label={t("Status")}>
          <option value="new">{t("New")}</option>
          <option value="handled">{t("Handled")}</option>
          <option value="spam">{t("Spam")}</option>
          <option value="">{t("Any status")}</option>
        </select>
      </div>
      {requests.length === 0 && <p className="muted">{t("Nothing here.")}</p>}
      {requests.map((r) => (
        <article className="card" key={r.id}>
          <div className="row">
            <span className={`badge${r.kind === "signup" ? " ok" : ""}`}>{t(r.kind === "signup" ? "Access request" : "Help request")}</span>
            <strong>{r.kind === "signup" ? r.firm : r.subject}</strong>
            <span className="spacer" />
            <span className="subtle">{new Date(r.created_at * 1000).toLocaleString()}</span>
          </div>
          <p className="muted" style={{ marginTop: 6 }}>
            {r.name ? `${r.name} · ` : ""}<a href={`mailto:${r.email}`}>{r.email}</a>
            {r.kind === "signup" && r.payload.phone ? ` · ${r.payload.phone}` : ""}
            {r.kind === "signup" && r.payload.contact_role ? ` · ${r.payload.contact_role}` : ""}
            {r.kind === "support" && r.firm ? ` · ${r.firm}` : ""}
          </p>
          <details>
            <summary className="subtle" style={{ cursor: "pointer" }}>{t("Details")}</summary>
            {r.kind === "support" ? (
              <p style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{r.payload.message}</p>
            ) : (
              <SignupDetails payload={r.payload} />
            )}
          </details>
          <div className="row" style={{ marginTop: 10 }}>
            <MailStatus request={r} />
            <span className="spacer" />
            {mailEnabled && r.mail_status !== "sent" && (
              <button onClick={() => act(api.post(`/admin/api/requests/${r.id}/resend`))}>{t("Send email again")}</button>
            )}
            {r.status !== "handled" && (
              <button className="primary" onClick={() => act(api.post(`/admin/api/requests/${r.id}/status`, { status: "handled" }))}>
                {t("Mark handled")}
              </button>
            )}
            {r.status === "new" && (
              <button className="danger-outline" onClick={() => act(api.post(`/admin/api/requests/${r.id}/status`, { status: "spam" }))}>
                {t("Spam")}
              </button>
            )}
            {r.status !== "new" && (
              <button onClick={() => act(api.post(`/admin/api/requests/${r.id}/status`, { status: "new" }))}>{t("Reopen")}</button>
            )}
          </div>
        </article>
      ))}
    </section>
  );
}

function MailStatus({ request }: { request: InboundRequest }) {
  const { t } = useI18n();
  const labels: Record<InboundRequest["mail_status"], [string, string]> = {
    sent: ["ok", "Emailed"],
    pending: ["", "Sending"],
    failed: ["danger", "Email failed"],
    off: ["warning", "Not emailed"],
  };
  const [tone, label] = labels[request.mail_status];
  return (
    <span className={`badge ${tone}`} title={request.mail_error || undefined}>{t(label)}</span>
  );
}

function SignupDetails({ payload }: { payload: Record<string, any> }) {
  const { t } = useI18n();
  const profile = payload.profile ?? {};
  const row = (label: string, value?: string) =>
    value ? (
      <div key={label} style={{ marginTop: 10 }}>
        <div className="section-label" style={{ marginBottom: 2 }}>{t(label)}</div>
        <div style={{ whiteSpace: "pre-wrap" }}>{value}</div>
      </div>
    ) : null;
  const nouns = NOUN_LABELS.map(([key, label]) => row(label, profile.proper_nouns?.[key])).filter(Boolean);
  return (
    <div>
      {PROFILE_LABELS.map(([key, label]) => row(label, profile[key]))}
      {LANGUAGE_LABELS.map(([key, label]) => row(label, profile.languages?.[key]))}
      {nouns.length > 0 ? nouns : <p className="subtle" style={{ marginTop: 10 }}>{t("No names given: collect them during setup.")}</p>}
    </div>
  );
}

// ---------- the GPU server ----------

function ServerSection() {
  const { t } = useI18n();
  const [box, setBox] = useState<BoxStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = () => api.get<BoxStatus>("/admin/api/box").then(setBox).catch((e) => setError(e.message));
  useEffect(() => {
    load();
    const timer = window.setInterval(load, 10000);
    return () => window.clearInterval(timer);
  }, []);
  const act = (action: string) => api.post<BoxStatus>(`/admin/api/box/${action}`).then(setBox).catch((e) => setError(e.message));

  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("GPU server")}</h2></div>
      <p className="page-intro">{t("The machine every firm's questions and ingestion run on. It starts when someone signs in and stops when nobody needs it.")}</p>
      {error && <p className="banner danger">{error}</p>}
      <div className="card">
        {box && (
          <>
            <div className="row">
              <span className={`dot ${box.state === "ready" ? "ready" : box.state === "error" ? "error" : box.state === "stopped" ? "" : "waking"}`} />
              <strong>{box.state}</strong>
              <span className="muted">{box.message}</span>
            </div>
            {box.error && <p className="banner danger" style={{ marginTop: 10 }}>{box.error}</p>}
            <p className="subtle" style={{ marginTop: 10 }}>
              {t("Queued")} {box.queued} · {t("Ingesting")} {box.ingesting} · {t("Reports running")} {box.reports_running}
            </p>
          </>
        )}
        <div className="row">
          <button className="primary" onClick={() => act("start")}>{t("Start")}</button>
          <button className="danger-outline" onClick={() => act("stop")}>{t("Stop")}</button>
        </div>
      </div>
    </section>
  );
}
