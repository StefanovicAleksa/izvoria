import { beforeEach, describe, expect, it } from "vitest";
import { ApiError } from "./client";
import {
  chunkCount, forgetPending, pendingBatchFor, rememberPending, retryable, uploadAll, type ChunkStatus, type Transport,
} from "./uploads";

const CHUNK = 10;

function file(name: string, size: number, lastModified = 1): File {
  return new File([new Uint8Array(size)], name, { lastModified });
}

/** The gateway as the uploader sees it: pieces it holds per file, and every piece sent, in order. */
function fakeGateway(have: Record<string, number[]> = {}, failures: Record<string, number> = {}) {
  const sent: string[] = [];
  let active = 0;
  let peak = 0;
  const transport: Transport = {
    status: async (_batch, name): Promise<ChunkStatus> => ({ received: have[name] ?? [], complete: false }),
    put: async (_batch, name, index, _size, _chunk, body, onProgress) => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((r) => setTimeout(r, 1));
      active--;
      const key = `${name}#${index}`;
      if ((failures[key] ?? 0) > 0) {
        failures[key]--;
        throw new ApiError("The upload was interrupted.", 0, "offline");
      }
      onProgress(body.size);
      sent.push(`${key}:${body.size}`);
    },
  };
  return { transport, sent, peak: () => peak };
}

const noSleep = async () => {};

describe("uploadAll", () => {
  it("sends every missing piece, several at once, with the right lengths", async () => {
    const gateway = fakeGateway();
    const progress: Record<string, number> = {};

    await uploadAll("b", [{ file: file("a.mp4", 25), name: "a.mp4" }], (n, f) => (progress[n] = f),
      { transport: gateway.transport, chunk: CHUNK, parallel: 3, sleep: noSleep });

    expect(gateway.sent.sort()).toEqual(["a.mp4#0:10", "a.mp4#1:10", "a.mp4#2:5"]);
    expect(gateway.peak()).toBe(3);
    expect(progress["a.mp4"]).toBe(1);
  });

  it("skips what the gateway already has: an upload continues where it stopped", async () => {
    const gateway = fakeGateway({ "a.mp4": [0, 2] });
    const progress: number[] = [];

    await uploadAll("b", [{ file: file("a.mp4", 25), name: "a.mp4" }], (_n, f) => progress.push(f),
      { transport: gateway.transport, chunk: CHUNK, sleep: noSleep });

    expect(gateway.sent).toEqual(["a.mp4#1:10"]);
    expect(progress[0]).toBeCloseTo(15 / 25);
  });

  it("sends a piece again after the connection drops", async () => {
    const gateway = fakeGateway({}, { "a.mp4#0": 2 });
    const pauses: number[] = [];

    await uploadAll("b", [{ file: file("a.mp4", 5), name: "a.mp4" }], () => {},
      { transport: gateway.transport, chunk: CHUNK, sleep: async (ms) => { pauses.push(ms); } });

    expect(gateway.sent).toEqual(["a.mp4#0:5"]);
    expect(pauses).toEqual([1000, 2000]);
  });

  it("gives up on a refused piece at once", async () => {
    const transport: Transport = {
      status: async () => ({ received: [], complete: false }),
      put: async () => { throw new ApiError("The file is larger than the appliance accepts.", 400, "invalid"); },
    };
    await expect(uploadAll("b", [{ file: file("a.mp4", 5), name: "a.mp4" }], () => {},
      { transport, chunk: CHUNK, sleep: noSleep })).rejects.toThrow("larger");
  });

  it("an empty file is one empty piece", async () => {
    const gateway = fakeGateway();
    await uploadAll("b", [{ file: file("e.txt", 0), name: "e.txt" }], () => {},
      { transport: gateway.transport, chunk: CHUNK, sleep: noSleep });
    expect(gateway.sent).toEqual(["e.txt#0:0"]);
    expect(chunkCount(0, CHUNK)).toBe(1);
  });
});

describe("retryable", () => {
  it("retries the network and a busy server, not a refusal", () => {
    expect(retryable(new ApiError("x", 0))).toBe(true);
    expect(retryable(new ApiError("x", 503))).toBe(true);
    expect(retryable(new ApiError("x", 400))).toBe(false);
  });
});

describe("resuming after the page was closed", () => {
  beforeEach(() => window.localStorage.clear());
  const items = [{ file: file("a.mp4", 25, 7), name: "a.mp4" }];

  it("the same files of the same kind and language continue their batch", () => {
    rememberPending("batch-1", "video", "en", items, 1000);
    expect(pendingBatchFor("video", "en", items, 2000)).toBe("batch-1");
    expect(pendingBatchFor("audio", "en", items, 2000)).toBeNull();
    expect(pendingBatchFor("video", "en", [{ file: file("a.mp4", 26, 7), name: "a.mp4" }], 2000)).toBeNull();
  });

  it("an old or finished one is not resumed", () => {
    rememberPending("batch-1", "video", "en", items, 0);
    expect(pendingBatchFor("video", "en", items, 21 * 3600 * 1000)).toBeNull();
    rememberPending("batch-1", "video", "en", items, 0);
    forgetPending();
    expect(pendingBatchFor("video", "en", items, 1)).toBeNull();
  });
});
