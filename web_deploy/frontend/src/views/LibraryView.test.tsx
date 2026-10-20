import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { LibraryView } from "./OtherViews";
import { SessionContext } from "../session";
import { fakeGateway } from "../test-utils";

const SOURCE = { source_id: "s1", name: "wrong language.mp4", source_type: "video", language: "sr",
                 size_bytes: 1, created_at: "2026-10-20T10:00:00Z" };

function show(role: string) {
  const calls = fakeGateway({
    "GET /api/library": { body: { sources: [SOURCE] } },
    "DELETE /api/library/s1": { body: { deleted: true, files_removed: 3 } },
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

    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));

    expect(confirm.mock.calls[0][0]).toContain("wrong language.mp4");
    await waitFor(() => expect(screen.queryByText("wrong language.mp4")).not.toBeInTheDocument());
    expect(calls.some((c) => c.method === "DELETE" && c.url === "/api/library/s1")).toBe(true);
  });

  it("keeps the file when the confirmation is declined", async () => {
    const calls = show("admin");
    vi.spyOn(window, "confirm").mockReturnValue(false);
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    expect(screen.getByText("wrong language.mp4")).toBeInTheDocument();
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
  });

  it("is not offered to members", async () => {
    show("member");
    expect(await screen.findByText("wrong language.mp4")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
  });
});
