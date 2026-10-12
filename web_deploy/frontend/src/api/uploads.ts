// Resumable, parallel uploads (the gateway's side is web_deploy/common/chunked.py).
//
// One connection to a server far away rarely fills a fast office line, and a 50 GB recording sent in one request
// starts over from zero when the connection drops. So every file goes in 32 MB pieces, several at once (also within
// one large file), a piece that fails is sent again after a pause, and the gateway says which pieces it already has,
// so an upload broken off, even by closing the page, continues where it stopped.

import { ApiError, api } from "./client";

export const CHUNK_BYTES = 32 * 1024 * 1024;
export const PARALLEL = 4;
const MAX_ATTEMPTS = 8;
const PENDING_KEY = "izvoria.pendingUpload";
// The gateway removes a batch nobody queued after a day; an older one is not worth resuming
const PENDING_MAX_AGE_MS = 20 * 3600 * 1000;

export interface UploadItem {
  file: File;
  name: string;
}

export interface ChunkStatus {
  received: number[];
  complete: boolean;
}

export interface Transport {
  status(batch: string, name: string, size: number, chunk: number): Promise<ChunkStatus>;
  put(batch: string, name: string, index: number, size: number, chunk: number, body: Blob,
      onProgress: (loaded: number) => void): Promise<void>;
}

export interface UploadOptions {
  transport?: Transport;
  parallel?: number;
  chunk?: number;
  sleep?: (ms: number) => Promise<void>;
}

const path = (batch: string, name: string) =>
  `/api/uploads/${encodeURIComponent(batch)}/${encodeURIComponent(name)}/chunks`;

export const httpTransport: Transport = {
  status: (batch, name, size, chunk) => api.post<ChunkStatus>(path(batch, name), { size, chunk }),
  put: (batch, name, index, size, chunk, body, onProgress) =>
    new Promise((resolve, reject) => {
      // XHR, not fetch, for upload progress
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `${path(batch, name)}/${index}?size=${size}&chunk=${chunk}`);
      xhr.withCredentials = true;
      xhr.upload.onprogress = (e) => onProgress(e.loaded);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) return resolve();
        let message = xhr.statusText;
        let kind = "error";
        try {
          const data = JSON.parse(xhr.responseText);
          message = data.error ?? message;
          kind = data.kind ?? kind;
        } catch {
          /* not JSON */
        }
        reject(new ApiError(message, xhr.status, kind));
      };
      xhr.onerror = () => reject(new ApiError("The upload was interrupted.", 0, "offline"));
      xhr.send(body);
    }),
};

export function chunkCount(size: number, chunk: number = CHUNK_BYTES): number {
  return Math.max(1, Math.ceil(size / chunk));
}

/** Whether a failed piece is worth sending again: the network, or the server busy or briefly away. */
export function retryable(error: unknown): boolean {
  const status = error instanceof ApiError ? error.status : 0;
  return status === 0 || status === 408 || status === 429 || status >= 500;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

/** Sends every file of the batch, PARALLEL pieces at a time, skipping the pieces the gateway already has. */
export async function uploadAll(batch: string, items: UploadItem[], onProgress: (name: string, fraction: number) => void,
                                options: UploadOptions = {}): Promise<void> {
  const transport = options.transport ?? httpTransport;
  const chunk = options.chunk ?? CHUNK_BYTES;
  const sleep = options.sleep ?? defaultSleep;

  type Task = { item: UploadItem; index: number };
  const tasks: Task[] = [];
  const done = new Map<string, number>();
  const inFlight = new Map<string, Map<number, number>>();
  const report = (item: UploadItem) => {
    const flying = [...(inFlight.get(item.name)?.values() ?? [])].reduce((a, b) => a + b, 0);
    const size = item.file.size;
    onProgress(item.name, size ? Math.min(1, ((done.get(item.name) ?? 0) + flying) / size) : 1);
  };
  const length = (item: UploadItem, index: number) => Math.min(chunk, item.file.size - index * chunk);

  for (const item of items) {
    const status = await withRetries(() => transport.status(batch, item.name, item.file.size, chunk), sleep);
    const have = new Set(status.complete ? range(chunkCount(item.file.size, chunk)) : status.received);
    done.set(item.name, [...have].reduce((n, i) => n + Math.max(0, length(item, i)), 0));
    inFlight.set(item.name, new Map());
    report(item);
    for (const index of range(chunkCount(item.file.size, chunk))) {
      if (!have.has(index)) tasks.push({ item, index });
    }
  }

  let next = 0;
  const worker = async () => {
    while (next < tasks.length) {
      const { item, index } = tasks[next++];
      const start = index * chunk;
      const body = item.file.slice(start, start + Math.max(0, length(item, index)));
      const flying = inFlight.get(item.name)!;
      await withRetries(() => transport.put(batch, item.name, index, item.file.size, chunk, body, (loaded) => {
        flying.set(index, loaded);
        report(item);
      }), sleep, () => flying.set(index, 0));
      flying.delete(index);
      done.set(item.name, (done.get(item.name) ?? 0) + body.size);
      report(item);
    }
  };
  await Promise.all(range(Math.min(options.parallel ?? PARALLEL, Math.max(1, tasks.length))).map(worker));
}

async function withRetries<T>(call: () => Promise<T>, sleep: (ms: number) => Promise<void>,
                              onRetry: () => void = () => {}): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await call();
    } catch (error) {
      if (!retryable(error) || attempt >= MAX_ATTEMPTS) throw error;
      onRetry();
      await sleep(Math.min(30_000, 1000 * 2 ** (attempt - 1)));
    }
  }
}

function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

// ---------- resuming after the page was closed ----------

interface Pending {
  batch: string;
  mode: string;
  language: string;
  files: string[];
  savedAt: number;
}

const signature = (item: UploadItem) => `${item.name}|${item.file.size}|${item.file.lastModified}`;

/** The batch of an unfinished upload of these same files (same kind and language), to continue it. */
export function pendingBatchFor(mode: string, language: string, items: UploadItem[], now: number = Date.now()): string | null {
  try {
    const pending: Pending = JSON.parse(window.localStorage.getItem(PENDING_KEY) ?? "null");
    if (!pending || pending.mode !== mode || pending.language !== language) return null;
    if (now - pending.savedAt > PENDING_MAX_AGE_MS) return null;
    const known = new Set(pending.files);
    return items.length > 0 && items.every((item) => known.has(signature(item))) ? pending.batch : null;
  } catch {
    return null;
  }
}

export function rememberPending(batch: string, mode: string, language: string, items: UploadItem[],
                                now: number = Date.now()): void {
  try {
    const pending: Pending = { batch, mode, language, files: items.map(signature), savedAt: now };
    window.localStorage.setItem(PENDING_KEY, JSON.stringify(pending));
  } catch {
    /* private window: the upload still works, it just can't be resumed after a reload */
  }
}

export function forgetPending(): void {
  try {
    window.localStorage.removeItem(PENDING_KEY);
  } catch {
    /* nothing to forget */
  }
}
