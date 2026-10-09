import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { ReportView } from "./ReportView";
import { SessionContext } from "../session";

const documentBody = {
  stage: "document",
  running: false,
  document: {
    report_id: "r1", topic: "Termination", title: "", markdown: "# Termination\n\nThe notice is 30 days [Doc 4](onyx://doc/4).",
    status: "completed", mode: "deep", language: "en", section_count: 1, uncovered_count: 0, rounds_used: 1,
    indexed_source_count: 3, swept_source_count: 3,
    evidence: [{ doc_number: 4, chunk_id: "c4", source_id: "s", start_sec: 0, end_sec: 0, excerpt: "Thirty days' notice.",
                 speakers: [], media_kind: "document", has_source: false, location: "p. 7", anchor_kind: "page",
                 page_start: 7, page_end: 7 }],
  },
};

const session = { me: { console: "firm", firm: null, user: null, languages: ["en", "sr"] }, answerLanguage: "en",
                  setAnswerLanguage: () => {}, signOut: () => {} } as any;

function show(path: string) {
  render(
    <SessionContext.Provider value={session}>
      <MemoryRouter initialEntries={[path]}>
        <ReportViewRoute />
      </MemoryRouter>
    </SessionContext.Provider>,
  );
}

describe("report titles", () => {
  const planBody = {
    stage: "plan",
    plan: { report_id: "p1", topic: "create a report on notice periods", title: "Notice Periods", scope_note: "",
            sections: [{ key: "s1", title: "Background", brief: "", questions: [] }], indexed_source_count: 3,
            indexed_chunk_count: 9, expected_job_count: 4, estimated_minutes: 2 },
  };
  let posted: { url: string; body: any }[];

  beforeEach(() => {
    posted = [];
    globalThis.fetch = vi.fn(async (url: any, init?: RequestInit) => {
      const path = String(url);
      if (init?.method === "POST") {
        posted.push({ url: path, body: JSON.parse(String(init.body ?? "{}")) });
        return new Response(JSON.stringify({ run_id: "run1", title: "x" }), { status: 200 });
      }
      if (path === "/api/reports") {
        return new Response(JSON.stringify({ reports: [
          { report_id: "r1", topic: "create a report on termination", title: "Termination Rights", status: "completed",
            created_at: null, completed_at: null, section_count: 1, evidence_count: 1 },
          { report_id: "r0", topic: "an old request", title: "", status: "completed",
            created_at: null, completed_at: null, section_count: 1, evidence_count: 1 },
        ] }), { status: 200 });
      }
      if (path.startsWith("/api/reports/p1")) return new Response(JSON.stringify(planBody), { status: 200 });
      if (path.startsWith("/api/runs")) return new Response(JSON.stringify({ runs: [] }), { status: 200 });
      return new Response(JSON.stringify(documentBody), { status: 200 });
    }) as any;
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("lists reports by title, and an untitled one by its request", async () => {
    show("/reports/r1");

    expect(await screen.findByText("Termination Rights")).toBeInTheDocument();
    expect(screen.getByText("an old request")).toBeInTheDocument();
    expect(screen.queryByText("create a report on termination")).not.toBeInTheDocument();
  });

  it("proposes the model's title for editing and sends the edited one with the outline", async () => {
    show("/reports/p1");

    const title = await screen.findByLabelText("Title");
    expect(title).toHaveValue("Notice Periods");
    expect(screen.getByText(/create a report on notice periods/)).toBeInTheDocument();
    fireEvent.change(title, { target: { value: "Notice Periods in Leases" } });
    fireEvent.click(screen.getByRole("button", { name: "Write the report" }));

    await waitFor(() => expect(posted[0]?.url).toBe("/api/reports/p1/run"));
    expect(posted[0].body.plan.title).toBe("Notice Periods in Leases");
  });

  it("renames a finished report", async () => {
    show("/reports/r1");

    fireEvent.click(await screen.findByRole("button", { name: "Rename" }));
    fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Thirty Days' Notice" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(posted[0]?.url).toBe("/api/reports/r1/rename"));
    expect(posted[0].body).toEqual({ title: "Thirty Days' Notice" });
  });
});

describe("report math", () => {
  it("typesets $$ formulas and leaves prices as text", async () => {
    const body = { ...documentBody, document: { ...documentBody.document,
      markdown: "# T\n\nElasticity $$E_p$$ is defined [Doc 4](onyx://doc/4). It cut the price from $6.40 to $3.00." } };
    globalThis.fetch = vi.fn(async (url: any) =>
      new Response(JSON.stringify(String(url).startsWith("/api/reports/r1") ? body : { reports: [] }), { status: 200 }),
    ) as any;

    show("/reports/r1");

    await screen.findByRole("button", { name: "Doc 4" });
    const article = document.querySelector("article.markdown")!;
    expect(article.querySelectorAll(".katex").length).toBe(1);
    expect(article.textContent).toContain("It cut the price from $6.40 to $3.00.");
  });
});

describe("ReportView", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn(async (url: any) => {
      const body = String(url).startsWith("/api/reports/r1") ? documentBody : { reports: [] };
      return new Response(JSON.stringify(body), { status: 200 });
    }) as any;
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("renders the report with [Doc N] as buttons that bring up their evidence", async () => {
    const session = { me: { console: "firm", firm: null, user: null, languages: ["en", "sr"] }, answerLanguage: "en",
                      setAnswerLanguage: () => {}, signOut: () => {} } as any;
    render(
      <SessionContext.Provider value={session}>
        <MemoryRouter initialEntries={["/reports/r1"]}>
          <ReportViewRoute />
        </MemoryRouter>
      </SessionContext.Provider>,
    );

    const cite = await screen.findByRole("button", { name: "Doc 4" });
    fireEvent.click(cite);

    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
    expect(screen.getByText("Thirty days' notice.").closest("article")).toHaveClass("highlighted");
    expect(screen.getByRole("link", { name: "Save as Markdown" })).toHaveAttribute("href", "/api/reports/r1/markdown");
    expect(screen.getByRole("link", { name: "Save as PDF" })).toHaveAttribute("href", "/api/reports/r1/pdf");
  });
});

function ReportViewRoute() {
  return (
    <Routes>
      <Route path="/reports/:reportId" element={<ReportView />} />
    </Routes>
  );
}
