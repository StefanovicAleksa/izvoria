import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { User } from "../api/types";
import { useI18n } from "../i18n";
import { useSession } from "../session";

/**
 * A firm administrator's page: the firm's accounts, and the password everyone at the firm shares for the first
 * sign-in step. The gateway keeps an admin from locking themselves or the firm out.
 */
export function AccountsView() {
  const { t } = useI18n();
  const { me } = useSession();
  const [users, setUsers] = useState<User[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [resetting, setResetting] = useState<string | null>(null);
  const [resetPassword, setResetPassword] = useState("");
  const [form, setForm] = useState({ email: "", name: "", password: "", role: "member" });

  const load = () => api.get<{ users: User[] }>("/firm/api/users").then((r) => setUsers(r.users)).catch((e) => setError(e.message));
  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Whether it worked: a form is cleared only then, so a refused one keeps what was typed. */
  const act = async (promise: Promise<unknown>, done: string | null = null): Promise<boolean> => {
    setError(null);
    setNotice(null);
    try {
      await promise;
    } catch (e: any) {
      setError(e.message);
      return false;
    }
    if (done) setNotice(done);
    await load();
    return true;
  };

  const add = async () => {
    if (await act(api.post("/firm/api/users", form), t("Account created. Give the person their password and the firm's sign-in details."))) {
      setForm({ email: "", name: "", password: "", role: "member" });
    }
  };

  const reset = async (id: string) => {
    if (await act(api.post(`/firm/api/users/${id}/password`, { password: resetPassword }), t("Password reset. They are signed out everywhere."))) {
      setResetting(null);
      setResetPassword("");
    }
  };

  return (
    <section className="page narrow">
      <div className="page-header"><h2>{t("Accounts")}</h2></div>
      <p className="page-intro">
        {t("Everyone at your firm signs in with the firm's shared password first, then with their own account. Administrators add and manage the accounts.")}
      </p>
      {error && <p className="banner danger">{error}</p>}
      {notice && <p className="banner info">{notice}</p>}

      <div className="card">
        <div className="section-label">{t("People at")} {me.firm?.name}</div>
        <table className="table">
          <thead>
            <tr><th>{t("Email")}</th><th>{t("Name")}</th><th>{t("Role")}</th><th>{t("Status")}</th><th /></tr>
          </thead>
          <tbody>
            {users.map((u) => {
              const self = u.id === me.user?.id;
              return (
                <tr key={u.id}>
                  <td>{u.email} {self && <span className="faint">({t("you")})</span>}</td>
                  <td>{u.name}</td>
                  <td>
                    <select value={u.role} disabled={self} aria-label={t("Role")}
                            onChange={(e) => act(api.post(`/firm/api/users/${u.id}/role`, { role: e.target.value }))}>
                      <option value="member">{t("Member")}</option>
                      <option value="admin">{t("Administrator")}</option>
                    </select>
                  </td>
                  <td>{u.disabled ? <span className="badge danger">{t("Disabled")}</span> : <span className="badge ok">{t("Active")}</span>}</td>
                  <td>
                    {resetting === u.id ? (
                      <form className="actions" onSubmit={(e) => { e.preventDefault(); reset(u.id); }}>
                        <input type="password" autoComplete="new-password" placeholder={t("New password")} minLength={10}
                               value={resetPassword} onChange={(e) => setResetPassword(e.target.value)} autoFocus />
                        <button className="primary" disabled={resetPassword.length < 10}>{t("Save")}</button>
                        <button type="button" onClick={() => setResetting(null)}>{t("Cancel")}</button>
                      </form>
                    ) : (
                      <div className="actions">
                        {self ? (
                          <span className="subtle">{t("Change your own password under Account.")}</span>
                        ) : (
                          <>
                            <button className="link" onClick={() => { setResetting(u.id); setResetPassword(""); }}>{t("Reset password")}</button>
                            <button className={`link${u.disabled ? "" : " danger"}`}
                                    onClick={() => act(api.post(`/firm/api/users/${u.id}/${u.disabled ? "enable" : "disable"}`))}>
                              {t(u.disabled ? "Enable" : "Disable")}
                            </button>
                          </>
                        )}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <form className="card" onSubmit={(e) => { e.preventDefault(); add(); }}>
        <div className="section-label">{t("Add an account")}</div>
        <div className="grid-2">
          <label className="field"><span>{t("Email")}</span>
            <input type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </label>
          <label className="field"><span>{t("Name")}</span>
            <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </label>
          <label className="field"><span>{t("Initial password")} <span className="hint">{t("At least 10 characters. They can change it under Account.")}</span></span>
            <input type="password" autoComplete="new-password" required minLength={10} value={form.password}
                   onChange={(e) => setForm({ ...form, password: e.target.value })} />
          </label>
          <label className="field"><span>{t("Role")}</span>
            <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
              <option value="member">{t("Member")}</option>
              <option value="admin">{t("Administrator")}</option>
            </select>
          </label>
        </div>
        <button className="primary" disabled={!form.email || form.password.length < 10}>{t("Add account")}</button>
      </form>

      <FirmPasswordCard />
    </section>
  );
}

function FirmPasswordCard() {
  const { t } = useI18n();
  const [mine, setMine] = useState("");
  const [next, setNext] = useState("");
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);

  const save = async () => {
    try {
      await api.post("/firm/api/password", { account_password: mine, password: next });
      setMessage({ ok: true, text: t("Firm password changed. Let your colleagues know the new one.") });
      setMine("");
      setNext("");
    } catch (e: any) {
      setMessage({ ok: false, text: e.message });
    }
  };

  return (
    <form className="card" onSubmit={(e) => { e.preventDefault(); save(); }}>
      <div className="section-label">{t("Firm password")}</div>
      <p className="muted">
        {t("The password everyone at your firm uses for the first sign-in step. Changing it asks every device to enter the new one; people already signed in stay signed in.")}
      </p>
      <div className="grid-2">
        <label className="field"><span>{t("Your own password")}</span>
          <input type="password" autoComplete="current-password" value={mine} onChange={(e) => setMine(e.target.value)} />
        </label>
        <label className="field"><span>{t("New firm password")}</span>
          <input type="password" autoComplete="new-password" minLength={10} value={next} onChange={(e) => setNext(e.target.value)} />
        </label>
      </div>
      {message && <p className={`banner ${message.ok ? "info" : "danger"}`}>{message.text}</p>}
      <button className="primary" disabled={!mine || next.length < 10}>{t("Change firm password")}</button>
    </form>
  );
}
