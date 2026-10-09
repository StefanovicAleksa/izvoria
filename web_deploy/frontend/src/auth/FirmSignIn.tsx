import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { useI18n } from "../i18n";
import { AuthLayout } from "./AuthLayout";

const LAST_FIRM = "onyx.firm";

function lastFirm(): string {
  try {
    return localStorage.getItem(LAST_FIRM) ?? "";
  } catch {
    return "";
  }
}

/** Step one: the firm's sign-in name and the password everyone at the firm shares. */
export function FirmSignIn({ onDone }: { onDone: () => void }) {
  const { t } = useI18n();
  const [params] = useSearchParams();
  const [firm, setFirm] = useState(() => params.get("firm") ?? lastFirm());
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.post("/auth/firm", { firm: firm.trim().toLowerCase(), password, remember });
      try {
        localStorage.setItem(LAST_FIRM, firm.trim().toLowerCase());
      } catch {
        /* private window */
      }
      onDone();
    } catch (e: any) {
      setError(e.message);
      setBusy(false);
    }
  };

  return (
    <AuthLayout
      footer={
        <>
          <span>
            {t("Is your firm not using Onyx yet?")} <Link to="/request-access">{t("Request access for your firm")}</Link>
          </span>
          <span>
            {t("Trouble signing in?")} <Link to="/help">{t("Help & support")}</Link>
          </span>
        </>
      }
    >
      <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <h1>{t("Sign in to your firm")}</h1>
        <p className="lead">
          {t("Enter your firm's sign-in name and the password your firm shares. You will then sign in with your own account.")}
        </p>
        <label className="field">
          <span>{t("Firm sign-in name")}</span>
          <input value={firm} onChange={(e) => setFirm(e.target.value)} required autoCapitalize="none" autoCorrect="off"
                 spellCheck={false} autoComplete="username" placeholder="legal-demo" autoFocus={!firm} />
        </label>
        <label className="field">
          <span>{t("Firm password")}</span>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required
                 autoComplete="current-password" autoFocus={!!firm} />
        </label>
        <label className="check">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          <span>{t("Remember this firm on this device for 30 days")}</span>
        </label>
        {error && <p className="banner danger">{t(error)}</p>}
        <button className="primary wide" disabled={busy || !firm.trim() || !password}>
          {busy ? t("Checking…") : t("Continue")}
        </button>
      </form>
    </AuthLayout>
  );
}
