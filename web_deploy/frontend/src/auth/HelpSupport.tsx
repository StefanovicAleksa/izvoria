import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Me } from "../api/types";
import { useI18n } from "../i18n";
import { AuthLayout, Honeypot } from "./AuthLayout";

/** A problem or a question, from anyone: signed in or not. It reaches the support inbox by email. */
export function HelpSupport({ me }: { me: Me | null }) {
  const { t, lang } = useI18n();
  const opened = useRef(Date.now());
  const signedIn = me?.stage === "signed_in";
  const [form, setForm] = useState({
    email: me?.user?.email ?? "",
    name: me?.user?.name ?? "",
    firm: me?.firm && me.console === "firm" ? me.firm.slug : "",
    subject: "",
    message: "",
  });
  const [website, setWebsite] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.post("/public/support", { ...form, language: lang, website, elapsed_ms: Date.now() - opened.current });
      setSent(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const footer = signedIn ? (
    <span><Link to="/">{t("Back to Onyx")}</Link></span>
  ) : (
    <>
      <span><Link to="/signin">{t("Back to sign in")}</Link></span>
      <span>{t("Is your firm not using Onyx yet?")} <Link to="/request-access">{t("Request access for your firm")}</Link></span>
    </>
  );

  if (sent) {
    return (
      <AuthLayout footer={footer}>
        <div className="done-state">
          <h1>{t("Message sent")}</h1>
          <p>{t("Thank you. We will reply by email as soon as we can.")}</p>
          <p className="muted">{t("A reply will come to")} {form.email}.</p>
        </div>
      </AuthLayout>
    );
  }

  const set = (key: keyof typeof form) => (e: { target: { value: string } }) => setForm({ ...form, [key]: e.target.value });

  return (
    <AuthLayout footer={footer}>
      <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <h1>{t("Help & support")}</h1>
        <p className="lead">
          {t("Describe the problem or question and we will reply by email. For a forgotten password, ask your firm's administrator first: they can reset it straight away.")}
        </p>
        <label className="field"><span>{t("Email")}</span>
          <input type="email" required autoComplete="email" value={form.email} onChange={set("email")} />
        </label>
        <div className="grid-2">
          <label className="field"><span>{t("Your name (optional)")}</span>
            <input autoComplete="name" maxLength={200} value={form.name} onChange={set("name")} />
          </label>
          <label className="field"><span>{t("Firm (optional)")}</span>
            <input maxLength={100} value={form.firm} onChange={set("firm")} />
          </label>
        </div>
        <label className="field"><span>{t("Subject")}</span>
          <input required maxLength={150} value={form.subject} onChange={set("subject")} />
        </label>
        <label className="field">
          <span>{t("What happened?")} <span className="hint">{t("What you did, what you expected and what you saw instead. Please don't include confidential material.")}</span></span>
          <textarea required rows={7} maxLength={8000} value={form.message} onChange={set("message")} />
        </label>
        <Honeypot value={website} onChange={setWebsite} />
        {error && <p className="banner danger">{t(error)}</p>}
        <button className="primary wide" disabled={busy || !form.email.trim() || !form.subject.trim() || !form.message.trim()}>
          {busy ? t("Sending…") : t("Send")}
        </button>
      </form>
    </AuthLayout>
  );
}
