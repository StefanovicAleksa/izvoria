// Every call the web app makes. Same-origin only: the gateway serves this app and the API at one address,
// so the sign-in cookies go along and no CORS is involved.

import type { ProgressEvent, RunError } from "./types";

/** Fired when the gateway says this browser is no longer signed in, so the app goes back to the sign-in page. */
export const SIGNED_OUT_EVENT = "izvoria:signed-out";

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly kind: string = "error", readonly body: any = null) {
    super(message);
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    credentials: "same-origin",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await response.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: text };
  }
  if (!response.ok) {
    if (response.status === 401 && (data?.kind === "auth" || data?.kind === "firm_stage")) {
      window.dispatchEvent(new Event(SIGNED_OUT_EVENT));
    }
    throw new ApiError(data?.error ?? response.statusText, response.status, data?.kind ?? "error", data);
  }
  return data as T;
}

export const api = {
  get: <T>(url: string) => request<T>("GET", url),
  post: <T>(url: string, body: unknown = {}) => request<T>("POST", url, body),
  put: <T>(url: string, body: unknown = {}) => request<T>("PUT", url, body),
  del: <T>(url: string) => request<T>("DELETE", url),
};

export interface RunHandlers<T> {
  onProgress?: (event: ProgressEvent) => void;
  onResult: (result: T) => void;
  onError: (error: RunError) => void;
}

/**
 * Follows a run's Server-Sent Events until its result or error. EventSource reconnects on its own and sends
 * Last-Event-ID, which the API resumes from, so a dropped connection mid-report loses nothing.
 * Returns a function that stops following (the run itself carries on, on the server).
 */
export function followRun<T>(runId: string, handlers: RunHandlers<T>): () => void {
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  let finished = false;
  const close = () => {
    finished = true;
    source.close();
  };
  source.addEventListener("progress", (e) => handlers.onProgress?.(JSON.parse((e as MessageEvent).data)));
  source.addEventListener("result", (e) => {
    close();
    handlers.onResult(JSON.parse((e as MessageEvent).data));
  });
  source.addEventListener("error", (e) => {
    const data = (e as MessageEvent).data;
    if (data) {
      close();
      handlers.onError(JSON.parse(data));
    } else if (!finished && source.readyState === EventSource.CLOSED) {
      handlers.onError({ kind: "offline", message: "Lost the connection to the appliance." });
    }
  });
  return close;
}

/** Starts a run and resolves with its result: the common case of a call that just takes a while. */
export function runToCompletion<T>(url: string, body: unknown, onProgress?: (e: ProgressEvent) => void): Promise<T> {
  return api.post<{ run_id: string }>(url, body).then(
    ({ run_id }) =>
      new Promise<T>((resolve, reject) => {
        followRun<T>(run_id, {
          onProgress,
          onResult: resolve,
          onError: (err) => reject(new ApiError(err.message, 0, err.kind)),
        });
      }),
  );
}

export function sourceUrl(chunkId: string, variant: string, label: string, location: string): string {
  const query = new URLSearchParams({ label, location }).toString();
  return `/api/sources/${encodeURIComponent(chunkId)}/${variant}?${query}`;
}
