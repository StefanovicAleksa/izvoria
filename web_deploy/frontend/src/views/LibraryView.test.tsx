import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LibraryView } from "./OtherViews";
import { SessionContext } from "../session";
import { fakeGateway } from "../test-utils";

const SOURCE = { source_id: "s1", name: "wrong language.mp4", title: null, topic: null, title_edited: false,
                 source_type: "video", language: "sr", size_bytes: 1, created_at: "2026-10-20T10:00:00Z" };
const LECTURE = { ...SOURCE, source_id: "s2", name: "module_04 [x1].mp4", title: "Lecture 4 Conditional Probability",
                  topic: "Bayes' rule and independence." };

function show(role: string) {
  const calls = fakeGateway({
    "GET /api/library": { body: { sources: [SOURCE, LECTURE] } },
    "DELETE /api/library/s1": { body: { deleted: true, files_removed: 3 } },
    "PUT /api/library/s2/profile": { body: { source_id: "s2", title: "Stat 110, Lecture 4", topic: "Bayes.",
                                             title_edited: true } },
  });
  const session = { me: { console: "firm", firm: { slug: "aass", name: "AASS", default_language: "sr" },
                          user: { id: "u1", role }, languages: ["en", "sr"] },
                    answerLanguage: "en", setAnswerLanguage: () => {}, signOut: () => {} } as any;
  render(<SessionContext.Provider value={session}><LibraryView /></SessionContext.Provider>);
  return calls;
}

describe("deleting from the Library", () => {
  it("lets an administrator delete a file after confirming", async () => {
    const calls = show("admin");
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);

    fireEvent.click((await screen.findAllByRole("button", { name: "Delete" }))[0]);

    expect(confirm.mock.calls[0][0]).toContain("wrong language.mp4");
    await waitFor(() => expect(screen.queryByText("wrong language.mp4")).not.toBeInTheDocument());
    expect(calls.some((c) => c.method === "DELETE" && c.url === "/api/library/s1")).toBe(true);
  });

  it("keeps the file when the confirmation is declined", async () => {
    const calls = show("admin");
    vi.spyOn(window, "confirm").mockReturnValue(false);
    fireEvent.click((await screen.findAllByRole("button", { name: "Delete" }))[0]);
    expect(screen.getByText("wrong language.mp4")).toBeInTheDocument();
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
  });

  it("is not offered to members", async () => {
    show("member");
    expect(await screen.findByText("wrong language.mp4")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
  });
});

describe("titles in the Library", () => {
  it("shows a file by its title, with its file name and what it covers beneath", async () => {
    show("member");
    expect(await screen.findByText("Lecture 4 Conditional Probability")).toBeInTheDocument();
    expect(screen.getByText("module_04 [x1].mp4")).toBeInTheDocument();
    expect(screen.getByText("Bayes' rule and independence.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Rename" })).not.toBeInTheDocument();
  });

  it("lets an administrator retitle a file", async () => {
    const calls = show("admin");
    fireEvent.click((await screen.findAllByRole("button", { name: "Rename" }))[1]);

    fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Stat 110, Lecture 4" } });
    fireEvent.change(screen.getByLabelText("What it covers"), { target: { value: "Bayes." } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Stat 110, Lecture 4")).toBeInTheDocument();
    const put = calls.find((c) => c.method === "PUT");
    expect(put?.url).toBe("/api/library/s2/profile");
    expect(put?.body).toEqual({ title: "Stat 110, Lecture 4", topic: "Bayes." });
  });
});
