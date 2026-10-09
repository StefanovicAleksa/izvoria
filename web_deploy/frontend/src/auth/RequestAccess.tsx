import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useI18n } from "../i18n";
import { AuthLayout, Honeypot } from "./AuthLayout";

// The firm profile the glossary is built from (docs/glossary_authoring/firm_profile_template.md), asked of the
// firm itself. The gateway emails it in the template's order.
const NOUNS = [
  ["people", "People", "Partners, doctors, analysts, frequent guests."],
  ["clients", "Clients, counterparties and institutions", "Organisations that come up often."],
  ["products", "Products, funds, drugs, devices or software", "What your firm uses or works with."],
  ["places", "Places", "Courts, hospitals, exchanges, local towns."],
  ["acronyms", "Internal acronyms and in-house terms", "With what each one stands for."],
  ["laws", "Laws, regulations, standards and cases", "Those cited often."],
] as const;

type Nouns = Record<(typeof NOUNS)[number][0], string>;

const EMPTY_NOUNS: Nouns = { people: "", clients: "", products: "", places: "", acronyms: "", laws: "" };

export function RequestAccess() {
  const { t, lang } = useI18n();
  const opened = useRef(Date.now());
  const [contact, setContact] = useState({ firm_name: "", contact_name: "", contact_role: "", email: "", phone: "" });
  const [profile, setProfile] = useState({ industry: "", sub_domains: "", description: "", data_kinds: "", mishearings: "" });
  const [languages, setLanguages] = useState({ share: "", english_terms: "", latin: "" });
  const [nouns, setNouns] = useState<Nouns>(EMPTY_NOUNS);
  const [website, setWebsite] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.post("/public/signup", {
        ...contact, language: lang, website, elapsed_ms: Date.now() - opened.current,
        profile: { ...profile, languages, proper_nouns: nouns },
      });
      setSent(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const footer = (
    <>
      <span><Link to="/signin">{t("Back to sign in")}</Link></span>
      <span>{t("Questions first?")} <Link to="/help">{t("Help & support")}</Link></span>
    </>
  );

  if (sent) {
    return (
      <AuthLayout footer={footer}>
        <div className="done-state">
          <h1>{t("Thank you")}</h1>
          <p>{t("We have received your request and will contact you to arrange the next steps.")}</p>
          <p className="muted">{t("A reply will come to")} {contact.email}.</p>
        </div>
      </AuthLayout>
    );
  }

  const contactField = (key: keyof typeof contact, label: string, extra: object = {}) => (
    <label className="field">
      <span>{t(label)}</span>
      <input value={contact[key]} onChange={(e) => setContact({ ...contact, [key]: e.target.value })} maxLength={200} {...extra} />
    </label>
  );
  const profileField = (key: keyof typeof profile, label: string, hint: string, rows = 3) => (
    <label className="field">
      <span>{t(label)} <span className="hint">{t(hint)}</span></span>
      <textarea rows={rows} maxLength={4000} value={profile[key]} onChange={(e) => setProfile({ ...profile, [key]: e.target.value })} />
    </label>
  );
  const languageField = (key: keyof typeof languages, label: string, hint: string) => (
    <label className="field">
      <span>{t(label)} <span className="hint">{t(hint)}</span></span>
      <textarea rows={2} maxLength={2000} value={languages[key]} onChange={(e) => setLanguages({ ...languages, [key]: e.target.value })} />
    </label>
  );

  return (
    <AuthLayout wide footer={footer}>
      <form onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <h1>{t("Bring Onyx to your firm")}</h1>
        <p className="lead">
          {t("Tell us about your firm and we will be in touch to arrange a demonstration and the next steps. Onyx is set up for each firm individually.")}
        </p>

        <div className="form-section">
          <h3>{t("Contact")}</h3>
          <div className="grid-2">
            {contactField("firm_name", "Firm name", { required: true, autoComplete: "organization" })}
            {contactField("contact_name", "Your name", { required: true, autoComplete: "name" })}
            {contactField("email", "Email", { required: true, type: "email", autoComplete: "email" })}
            {contactField("phone", "Phone (optional)", { type: "tel", autoComplete: "tel", maxLength: 50 })}
            {contactField("contact_role", "Your role at the firm (optional)", { autoComplete: "organization-title" })}
          </div>
        </div>

        <div className="form-section">
          <h3>{t("Tailoring Onyx to your firm")}</h3>
          <p className="why">
            {t("Onyx is tuned to each firm's vocabulary, so that the names, terms and abbreviations your people use are recognised and spelled correctly in transcripts and answers. The answers below are what we tune it with. Please be as accurate as you can, and leave a field empty rather than guess: you can add or correct anything later.")}
          </p>
          {profileField("industry", "Field of work", "For example law, medicine, finance, consulting.", 1)}
          {profileField("description", "What your firm does", "For whom, and in which jurisdictions or markets.")}
          {profileField("sub_domains", "Practice areas or specialities", "One per line, with a short description, e.g. “M&A: share and asset deals, due diligence”.", 4)}
          {profileField("data_kinds", "What you will bring to Onyx", "Meetings, hearings, consultations, lectures, dictations, contracts? Who speaks, and how formally?")}
        </div>

        <div className="form-section">
          <h3>{t("Languages")}</h3>
          {languageField("share", "Languages of your material", "Roughly what share is in English and what in Serbian.")}
          {languageField("english_terms", "English terms in Serbian speech", "Do Serbian speakers say terms like “due diligence” or “stop loss” in English? Which ones, and where do they use a Serbian word instead?")}
          {languageField("latin", "Latin", "Is Latin used in speech, e.g. diagnoses or legal maxims?")}
        </div>

        <div className="form-section">
          <h3>{t("Names and terms")} <span className="badge">{t("Optional")}</span></h3>
          <p className="why">
            {t("Names are where speech recognition benefits most: it has never heard your clients, colleagues or products. Exact spelling matters, because a name is written into transcripts exactly as given here.")}
          </p>
          <details className="optional">
            <summary>{t("Add names and terms")}</summary>
            <p className="notice">
              {t("Only include names you are permitted to share: this form is sent to us by email. Never include client documents, transcripts or case details. You can leave this section empty and give us the names later, during setup.")}
            </p>
            {NOUNS.map(([key, label, hint]) => (
              <label className="field" key={key}>
                <span>{t(label)} <span className="hint">{t(hint)}</span></span>
                <textarea rows={2} maxLength={4000} value={nouns[key]} onChange={(e) => setNouns({ ...nouns, [key]: e.target.value })} />
              </label>
            ))}
            {profileField("mishearings", "Terms transcription already gets wrong", "If you have used speech recognition before: “heard X, should be Y”.", 2)}
          </details>
        </div>

        <Honeypot value={website} onChange={setWebsite} />
        {error && <p className="banner danger">{t(error)}</p>}
        <div className="row end">
          <button className="primary" disabled={busy || !contact.firm_name.trim() || !contact.contact_name.trim() || !contact.email.trim()}>
            {busy ? t("Sending…") : t("Send request")}
          </button>
        </div>
      </form>
    </AuthLayout>
  );
}
