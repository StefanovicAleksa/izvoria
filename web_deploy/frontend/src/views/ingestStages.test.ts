import type { IngestJob } from "../api/types";
import { duration, epochSeconds, stageRows } from "./ingestStages";

function job(overrides: Partial<IngestJob>): IngestJob {
  return {
    id: "j1", mode: "document", language: "en", file_count: 2, status: "running", position: null, owner_id: "u1",
    created_at: null, started_at: null, finished_at: null, progress: null, report: null, error: null,
    ...overrides,
  };
}

const states = (j: IngestJob) => stageRows(j).map((r) => [r.stage.key, r.state, r.fraction]);

describe("stageRows", () => {
  it("shows every step waiting while the batch is queued", () => {
    expect(states(job({ status: "queued" })).map((r) => r[1])).toEqual(["waiting", "waiting", "waiting", "waiting"]);
  });

  it("shows a busy bar while files are being prepared", () => {
    const j = job({ progress: { phase: "waiting_for_gpu", state: "started", detail: null } });
    expect(states(j)[0]).toEqual(["preparing", "active", null]);
  });

  it("folds jobs into the furthest step reached, with earlier steps done", () => {
    const j = job({
      progress: {
        phase: "document_contextualization", state: "completed", detail: null, files: 2,
        done_by_phase: { directory_scan: 1, file_ingest: 2, document_conversion: 2, document_contextualization: 1 },
      },
    });
    expect(states(j)).toEqual([
      ["preparing", "done", 1],
      ["documents", "done", 1],
      ["context", "active", 0.5],
      ["indexing", "waiting", 0],
    ]);
  });

  it("never shows a running step as full, and ignores jobs it does not know", () => {
    const j = job({
      progress: { phase: "some_future_job", state: "completed", detail: null, files: 1,
                  done_by_phase: { document_conversion: 5, some_future_job: 3 } },
    });
    expect(states(j)[1]).toEqual(["documents", "active", 0.97]);
  });

  it("marks the step a failed batch stopped in", () => {
    const j = job({ status: "failed", progress: { phase: "rag_indexing", state: "failed", detail: null, files: 1,
                                                  done_by_phase: { rag_indexing: 1 } } });
    expect(states(j).map((r) => r[1])).toEqual(["done", "done", "done", "failed"]);
  });

  it("shows a finished batch with every step done", () => {
    expect(states(job({ status: "done", mode: "image" })).map((r) => r[1])).toEqual(["done", "done", "done"]);
  });
});

describe("a YouTube batch", () => {
  const youtubeJob = (progress: any) => job({ mode: "video", origin: "youtube", file_count: 4, progress });

  it("starts with downloading, its bar counting the videos fetched", () => {
    const rows = stageRows(youtubeJob({ phase: "downloading", state: "started", detail: "Lecture 2", done: 1.5, total: 4 }));

    expect(rows.map((r) => r.stage.key)).toEqual(["downloading", "preparing", "audio", "context", "visuals", "indexing"]);
    expect(rows[0]).toMatchObject({ state: "active", fraction: 0.375 });
    expect(rows[1].state).toBe("waiting");
  });

  it("has the download done once it waits for the GPU", () => {
    const rows = stageRows(youtubeJob({ phase: "waiting_for_gpu", state: "started", detail: null }));

    expect(rows.slice(0, 2).map((r) => r.state)).toEqual(["done", "active"]);
  });

  it("an uploaded video batch has no download step", () => {
    expect(stageRows(job({ mode: "video", status: "queued" })).map((r) => r.stage.key)[0]).toBe("preparing");
  });
});

describe("duration", () => {
  it("formats like the desktop", () => {
    expect([duration(42), duration(252), duration(3780)]).toEqual(["42s", "4m 12s", "1h 03m"]);
  });
});

describe("epochSeconds", () => {
  it("reads a time without a zone as UTC", () => {
    const utc = Date.UTC(2026, 9, 7, 11, 47, 44) / 1000;
    expect(epochSeconds("2026-10-07T11:47:44")).toBe(utc);
    expect(epochSeconds("2026-10-07 11:47:44")).toBe(utc);
    expect(epochSeconds("2026-10-07T11:47:44+00:00")).toBe(utc);
    expect(epochSeconds("2026-10-07T13:47:44+02:00")).toBe(utc);
  });
});
