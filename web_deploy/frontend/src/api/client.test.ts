import { followRun } from "./client";

class FakeEventSource {
  static last: FakeEventSource;
  static CLOSED = 2;
  readyState = 1;
  listeners: Record<string, ((e: any) => void)[]> = {};
  closed = false;
  constructor(public url: string) {
    FakeEventSource.last = this;
  }
  addEventListener(type: string, fn: (e: any) => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  emit(type: string, data?: unknown) {
    this.listeners[type]?.forEach((fn) => fn({ data: data === undefined ? undefined : JSON.stringify(data) }));
  }
  close() {
    this.closed = true;
  }
}

describe("followRun", () => {
  beforeEach(() => {
    (globalThis as any).EventSource = FakeEventSource;
  });

  it("delivers progress, then the result, then stops listening", () => {
    const seen: string[] = [];
    followRun<{ ok: boolean }>("run 1", {
      onProgress: (e) => seen.push(e.phase),
      onResult: (r) => seen.push(`result:${r.ok}`),
      onError: () => seen.push("error"),
    });
    const source = FakeEventSource.last;
    expect(source.url).toBe("/api/runs/run%201/events");

    source.emit("progress", { phase: "rag_retrieval", state: "completed", detail: null });
    source.emit("result", { ok: true });

    expect(seen).toEqual(["rag_retrieval", "result:true"]);
    expect(source.closed).toBe(true);
  });

  it("reports the server's error event, and ignores a reconnect blip", () => {
    const errors: string[] = [];
    followRun("r", { onResult: () => {}, onError: (e) => errors.push(e.kind) });
    const source = FakeEventSource.last;

    source.emit("error"); // a dropped connection EventSource will retry: no data, still open
    source.emit("error", { kind: "busy", message: "Two reports are already running." });

    expect(errors).toEqual(["busy"]);
  });
});
