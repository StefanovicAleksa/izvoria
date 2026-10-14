import { useCallback, useEffect, useState, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { api, SIGNED_OUT_EVENT } from "./api/client";
import type { BoxStatus, Lang, Me } from "./api/types";
import { AccountSignIn } from "./auth/AccountSignIn";
import { FirmSignIn } from "./auth/FirmSignIn";
import { HelpSupport } from "./auth/HelpSupport";
import { RequestAccess } from "./auth/RequestAccess";
import { BoxStatusProvider, useBoxStatus } from "./box";
import { I18nProvider, useI18n } from "./i18n";
import { AppShell } from "./layout/AppShell";
import { SessionContext } from "./session";
import { ThemeProvider } from "./theme";
import { AccountsView } from "./views/AccountsView";
import { AdminConsole } from "./views/AdminConsole";
import { ChatView } from "./views/ChatView";
import { IngestView } from "./views/IngestView";
import { AccountView, LibraryView } from "./views/OtherViews";
import { ReportView } from "./views/ReportView";
import { SearchView } from "./views/SearchView";

export default function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.get<Me>("/auth/me").then((next) => {
      setError(null);
      setMe(next);
    }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    load();
    window.addEventListener(SIGNED_OUT_EVENT, load);
    return () => window.removeEventListener(SIGNED_OUT_EVENT, load);
  }, [load]);

  return (
    <ThemeProvider>
      <I18nProvider initial={me?.firm?.default_language ?? "en"}>
        {error ? (
          <div className="waking"><p className="banner danger">{error}</p></div>
        ) : !me ? (
          <div className="waking"><span className="spinner" /></div>
        ) : (
          <BrowserRouter>
            <AppRoutes me={me} reload={load} />
          </BrowserRouter>
        )}
      </I18nProvider>
    </ThemeProvider>
  );
}

/** The two public pages are always reachable; everything else depends on how far this browser has signed in. */
function AppRoutes({ me, reload }: { me: Me; reload: () => void }) {
  return (
    <Routes>
      <Route path="/request-access" element={<RequestAccess />} />
      <Route path="/help" element={<HelpSupport me={me} />} />
      {me.stage === "firm" && (
        <>
          <Route path="/signin" element={<FirmSignIn onDone={reload} />} />
          <Route path="*" element={<Navigate to="/signin" replace />} />
        </>
      )}
      {me.stage === "account" && (
        <>
          <Route path="/signin/account" element={<AccountSignIn me={me} onDone={reload} />} />
          <Route path="*" element={<Navigate to="/signin/account" replace />} />
        </>
      )}
      {me.stage === "signed_in" && <Route path="*" element={<SignedIn me={me} reload={reload} />} />}
    </Routes>
  );
}

function SignedIn({ me, reload }: { me: Me; reload: () => void }) {
  const [answerLanguage, setAnswerLanguage] = useState<Lang>(me.firm?.default_language ?? "en");
  const signOut = () => api.post("/auth/logout").finally(reload);
  const owner = me.console === "admin";
  const session = { me, answerLanguage, setAnswerLanguage, signOut, reload };

  if (owner) {
    return (
      <SessionContext.Provider value={session}>
        <AppShell>
          <Routes>
            <Route path="/admin/firms" element={<AdminConsole section="firms" />} />
            <Route path="/admin/requests" element={<AdminConsole section="requests" />} />
            <Route path="/admin/server" element={<AdminConsole section="server" />} />
            <Route path="/account" element={<AccountView />} />
            <Route path="*" element={<Navigate to="/admin/firms" replace />} />
          </Routes>
        </AppShell>
      </SessionContext.Provider>
    );
  }

  return (
    <SessionContext.Provider value={session}>
      <BoxStatusProvider>
        <AppShell>
          <Routes>
            <Route path="/chat" element={<BoxGate><ChatView /></BoxGate>} />
            <Route path="/chat/:sessionId" element={<BoxGate><ChatView /></BoxGate>} />
            <Route path="/search" element={<BoxGate><SearchView /></BoxGate>} />
            <Route path="/reports" element={<BoxGate><ReportView /></BoxGate>} />
            <Route path="/reports/:reportId" element={<BoxGate><ReportView /></BoxGate>} />
            <Route path="/ingest" element={<BoxGate><IngestView /></BoxGate>} />
            <Route path="/library" element={<BoxGate><LibraryView /></BoxGate>} />
            {/* The gateway answers these itself: they work while the GPU server sleeps or wakes */}
            <Route path="/account" element={<AccountView />} />
            {me.user?.role === "admin" && <Route path="/accounts" element={<AccountsView />} />}
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </AppShell>
      </BoxStatusProvider>
    </SessionContext.Provider>
  );
}

/** Holds the views back until the GPU server is awake and its models are loaded; it sleeps when nobody needs it. */
export function BoxGate({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  const { status, setStatus } = useBoxStatus();

  if (status?.state === "ready") {
    return (
      <>
        {status.paused && (
          <div className="banner warning" role="status">
            {t("Ingesting files: the GPU is busy with them. Chat, search and reports resume when the queue is empty; history stays readable.")}
          </div>
        )}
        {children}
      </>
    );
  }
  if (status?.state === "stopped") {
    return (
      <div className="waking">
        <div className="card">
          <h2>{t("The appliance is asleep")}</h2>
          <p className="muted">{t("It goes to sleep after half an hour without use, so the GPU isn't paid for while nobody needs it.")}</p>
          <button className="primary" onClick={() => api.post<BoxStatus>("/box/start").then(setStatus)}>{t("Wake it")}</button>
        </div>
      </div>
    );
  }
  return (
    <div className="waking">
      <div className="card">
        <h2>{t("Waking the appliance")}</h2>
        {status?.state !== "error" && <span className="spinner" />}
        <p className="muted">{status?.message ?? t("Loading…")}</p>
        {status?.error && <p className="banner danger">{status.error}</p>}
        {status?.state === "error" && (
          <button className="primary" onClick={() => api.post<BoxStatus>("/box/start").then(setStatus)}>{t("Try again")}</button>
        )}
      </div>
    </div>
  );
}
