import { useEffect, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { useI18n } from "../i18n";
import { Sidebar } from "./Sidebar";

/** Sidebar on the left, the view on the right; on a phone the sidebar is a drawer behind the menu button. */
export function AppShell({ children }: { children: ReactNode }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  return (
    <div className="shell">
      <Sidebar open={open} />
      <div className={`drawer-backdrop${open ? " open" : ""}`} onClick={() => setOpen(false)} />
      <div className="content">
        <div className="mobile-bar">
          <button type="button" onClick={() => setOpen(true)} aria-label={t("Menu")}>☰</button>
          <span className="brand">Onyx</span>
        </div>
        {children}
      </div>
    </div>
  );
}
