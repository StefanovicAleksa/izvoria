// The GPU server's state, polled once for the whole signed-in app: the sidebar's status card shows it and BoxGate
// holds the views back until it is ready. A stopped server is woken; it sleeps again when nobody needs it.

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api/client";
import type { BoxStatus } from "./api/types";

interface Box {
  status: BoxStatus | null;
  setStatus: (status: BoxStatus) => void;
}

const BoxContext = createContext<Box>({ status: null, setStatus: () => {} });

export function BoxStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<BoxStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        let next = await api.get<BoxStatus>("/box/status");
        if (next.state === "stopped") next = await api.post<BoxStatus>("/box/start");
        if (cancelled) return;
        setStatus(next);
        // While ready, keep an eye out for ingestion taking the GPU (dev hardware) so the status is current
        timer = window.setTimeout(poll, next.state === "ready" ? 10000 : 3000);
      } catch {
        if (!cancelled) timer = window.setTimeout(poll, 5000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  return <BoxContext.Provider value={{ status, setStatus }}>{children}</BoxContext.Provider>;
}

export const useBoxStatus = () => useContext(BoxContext);

/** The status dot's class and a short label, as on the desktop's appliance card. */
export function describeBox(status: BoxStatus | null): { dot: string; label: string } {
  if (!status) return { dot: "", label: "Connecting…" };
  switch (status.state) {
    case "ready":
      return status.paused ? { dot: "busy", label: "Ingesting files" } : { dot: "ready", label: "Ready" };
    case "starting":
    case "booting":
    case "gpu_unavailable":
      return { dot: "waking", label: "Starting" };
    case "stopping":
      return { dot: "waking", label: "Stopping" };
    case "error":
      return { dot: "error", label: "Error" };
    default:
      return { dot: "", label: "Offline" };
  }
}
