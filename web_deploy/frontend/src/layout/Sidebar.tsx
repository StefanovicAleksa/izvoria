import { NavLink } from "react-router-dom";
import type { Lang } from "../api/types";
import { describeBox, useBoxStatus } from "../box";
import { AnswerLanguage } from "../components/Controls";
import { useI18n } from "../i18n";
import { useSession } from "../session";
import { Toggles } from "./Toggles";

/** The ink sidebar of the desktop app: brand, the views, administration by role, and the appliance card. */
export function Sidebar({ open }: { open: boolean }) {
  const { t } = useI18n();
  const { me, answerLanguage, setAnswerLanguage, signOut } = useSession();
  const owner = me.console === "admin";
  const firmAdmin = !owner && me.user?.role === "admin";

  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label={t("Navigation")}>
      <div className="brand-row">
        <div className="names">
          <div className="brand">Izvoria</div>
          <div className="brand-sub" title={me.firm?.name}>{owner ? t("Admin console") : me.firm?.name}</div>
        </div>
        <Toggles />
      </div>

      {!owner && (
        <>
          <div className="section-label">{t("Workspace")}</div>
          <nav>
            <NavLink to="/chat">{t("Chat")}</NavLink>
            <NavLink to="/search">{t("Search")}</NavLink>
            <NavLink to="/ingest">{t("Ingest")}</NavLink>
            <NavLink to="/reports">{t("Report")}</NavLink>
            <NavLink to="/library">{t("Library")}</NavLink>
          </nav>
        </>
      )}

      {(owner || firmAdmin) && (
        <>
          <div className="section-label">{t("Administration")}</div>
          <nav>
            {firmAdmin && <NavLink to="/accounts">{t("Accounts")}</NavLink>}
            {owner && <NavLink to="/admin/firms">{t("Firms")}</NavLink>}
            {owner && <NavLink to="/admin/requests">{t("Requests")}</NavLink>}
            {owner && <NavLink to="/admin/server">{t("GPU server")}</NavLink>}
          </nav>
        </>
      )}

      <div className="grow" />
      <div className="appliance-card">
        {!owner && <ApplianceState />}
        {!owner && (
          <AnswerLanguage value={answerLanguage} onChange={setAnswerLanguage} languages={me.languages as Lang[]} />
        )}
        <div className="who">
          <NavLink to="/account" title={t("Account")}>{me.user?.email}</NavLink>
          <span className="role">{t(roleName(me.user?.role))}</span>
          <NavLink to="/help" className="help">{t("Help & support")}</NavLink>
          <button type="button" className="sign-out" onClick={signOut}>{t("Sign out")}</button>
        </div>
      </div>
    </aside>
  );
}

function ApplianceState() {
  const { t } = useI18n();
  const { status } = useBoxStatus();
  const { dot, label } = describeBox(status);
  return (
    <div>
      <div className="state">
        <span className={`dot ${dot}`} aria-hidden />
        <strong>{t(label)}</strong>
      </div>
      {status && status.state !== "ready" && status.message && <div className="state-text">{status.message}</div>}
    </div>
  );
}

function roleName(role?: string): string {
  if (role === "owner") return "Owner";
  if (role === "admin") return "Administrator";
  return "Member";
}
