import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Me } from "../api/types";
import { useI18n } from "../i18n";
import { AuthLayout } from "./AuthLayout";

/** Step two: a person's own email and password, within the firm signed in to at step one. */
export function AccountSignIn({ me, onDone }: { me: Me; onDone: () => void }) {
  const { t } = useI18n();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const owner = me.console === "admin";

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.post("/auth/login", { email, password });
      onDone();
    } catch (e: any) {
      setError(e.message);
      setBusy(false);
    }
  };

  const switchFirm = () => api.post("/auth/firm/forget").finally(onDone);

  return (
    <AuthLayout
      footer={
        <>
          <span>{t("Forgot your password? Your firm's administrator can reset it.")}</span>
          <span>
            {t("Trouble signing in?")} <Link to="/help">{t("Help & support")}</Link>
          </span>
        </>
      }
    >
      <div className="firm-pill">
        <span className="dot ready" aria-hidden />
        <strong>{owner ? t("Admin console") : me.firm?.name}</strong>
        <button type="button" className="link" onClick={switchFirm}>{t("Switch firm")}</button>
      </div>
      <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <h1>{t("Sign in")}</h1>
        <p className="lead">{owner ? t("Sign in with your owner account.") : t("Sign in with your own Izvoria account.")}</p>
        <label className="field">
          <span>{t("Email")}</span>
          <input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoFocus />
        </label>
        <label className="field">
          <span>{t("Password")}</span>
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)}
                 required />
        </label>
        {error && <p className="banner danger">{t(error)}</p>}
        <button className="primary wide" disabled={busy || !email || !password}>
          {busy ? t("Signing in…") : t("Sign in")}
        </button>
      </form>
    </AuthLayout>
  );
}
