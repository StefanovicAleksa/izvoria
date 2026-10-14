import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { IngestView, withoutRetriedFailures } from "./IngestView";
import { SessionContext } from "../session";
import { fakeGateway } from "../test-utils";

const session = {
  me: { console: "firm", firm: { slug: "legal-demo", name: "Legal demo", default_language: "en" }, user: null,
        languages: ["en", "sr"] },
  answerLanguage: "en", setAnswerLanguage: () => {}, signOut: () => {},
} as any;

const PREVIEW = {
  videos: [{ id: "lecture0001", title: "Lecture 1", duration: 600 }, { id: "lecture0002", title: "Lecture 2", duration: 905 }],
  problems: [{ link: "https://vimeo.com/1", reason: "Only YouTube links can be added." }],
  playlist_hints: [],
  capped: false,
};

function show(links = { enabled: true, max_videos: 200 }) {
  const calls = fakeGateway({
    "GET /api/ingest/jobs": { body: { jobs: [], links } },
    "POST /api/ingest/links/preview": { body: PREVIEW },
    "POST /api/ingest/links": { status: 201, body: {} },
  });
  render(<SessionContext.Provider value={session}><IngestView /></SessionContext.Provider>);
  return calls;
}

const chooseKind = (kind: string) => fireEvent.change(screen.getByLabelText("Kind of files"), { target: { value: kind } });

describe("adding from YouTube", () => {
  it("checks the links, lets one video be dropped, and queues the rest", async () => {
    const calls = show();
    chooseKind("video");

    fireEvent.change(await screen.findByLabelText("YouTube links"),
                     { target: { value: "https://www.youtube.com/playlist?list=PLx\n\nhttps://vimeo.com/1\n" } });
    fireEvent.click(screen.getByRole("button", { name: "Check links" }));

    expect(await screen.findByText("Lecture 2")).toBeInTheDocument();
    expect(screen.getByText(/Only YouTube links can be added/)).toBeInTheDocument();
    expect(screen.getByText("15:05")).toBeInTheDocument();
    expect(calls.find((c) => c.url === "/api/ingest/links/preview")!.body.links).toEqual(
      ["https://www.youtube.com/playlist?list=PLx", "https://vimeo.com/1"]);

    fireEvent.click(screen.getAllByRole("button", { name: "Remove" })[0]);
    fireEvent.click(screen.getByRole("button", { name: /Queue videos \(1 · 15:05\)/ }));

    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/api/ingest/links")).toBe(true));
    const queued = calls.find((c) => c.method === "POST" && c.url === "/api/ingest/links")!;
    expect(queued.body).toMatchObject({ mode: "video", language: "en", video_ids: ["lecture0002"] });
  });

  it("is offered for video only", async () => {
    show();
    await screen.findByLabelText("Kind of files");

    expect(screen.queryByText("Or add from YouTube")).not.toBeInTheDocument();
    chooseKind("video");
    expect(await screen.findByText("Or add from YouTube")).toBeInTheDocument();
    chooseKind("audio");
    expect(screen.queryByText("Or add from YouTube")).not.toBeInTheDocument();
  });

  it("is hidden where links are turned off", async () => {
    const calls = show({ enabled: false, max_videos: 0 });
    chooseKind("video");

    await waitFor(() => expect(calls.length).toBeGreaterThan(0));
    expect(screen.queryByText("Or add from YouTube")).not.toBeInTheDocument();
  });
});

describe("choosing a folder", () => {
  it("lists the folder's files of the chosen kind under their path inside it", async () => {
    show();
    chooseKind("video");
    const input = await screen.findByTestId("folder-input");
    expect(input).toHaveAttribute("webkitdirectory");

    const inFolder = (name: string, path: string) => {
      const file = new File(["x"], name);
      Object.defineProperty(file, "webkitRelativePath", { value: path });
      return file;
    };
    fireEvent.change(input, { target: { files: [
      inFolder("Part I.mp4", "Lectures/Unit 3/Part I.mp4"),
      inFolder("Part I.mp4", "Lectures/Unit 4/Part I.mp4"),
      inFolder("notes.pdf", "Lectures/notes.pdf"),
    ] } });

    expect(await screen.findByText("Unit 3 - Part I.mp4")).toBeInTheDocument();
    expect(screen.getByText("Unit 4 - Part I.mp4")).toBeInTheDocument();
    expect(screen.getByText(/2 files from Lectures/)).toBeInTheDocument();
    expect(screen.getByText(/1 left out/)).toBeInTheDocument();
  });
});

describe("the queue", () => {
  const job = (id: string, mode: string, status: string, created_at: string) =>
    ({ id, mode, status, created_at, language: "en", file_count: 1, position: null, owner_id: "u1",
       started_at: null, finished_at: null, progress: null }) as any;

  it("hides a failure once a newer batch of the same kind is under way, and keeps one nobody retried", () => {
    const jobs = [
      job("retry", "video", "running", "2026-10-14T14:34:00Z"),
      job("first", "video", "failed", "2026-10-14T12:58:00Z"),
      job("doc", "document", "failed", "2026-10-14T13:00:00Z"),
      job("older-cancel", "video", "cancelled", "2026-10-14T10:00:00Z"),
    ];
    expect(withoutRetriedFailures(jobs).map((j) => j.id)).toEqual(["retry", "doc"]);
    expect(withoutRetriedFailures([jobs[1]]).map((j) => j.id)).toEqual(["first"]);
  });
});
