import { fireEvent, render, screen } from "@testing-library/react";
import { CitedText, EvidenceCard, formatTime, SourcePanel, whereLabel } from "./Evidence";
import { AnswerLanguage } from "./Controls";
import type { Evidence } from "../api/types";

const evidence: Evidence = {
  doc_number: 3, chunk_id: "c3", source_id: "s", start_sec: 75, end_sec: 3700, excerpt: "The notice period is 30 days.",
  speakers: ["SPEAKER_00"], media_kind: "clip", has_source: true, location: "", anchor_kind: "time",
  page_start: null, page_end: null,
};

describe("CitedText", () => {
  it("turns every [Doc N] into a button that reports its number", () => {
    const cited: number[] = [];
    render(<CitedText text={"Thirty days [Doc 3], per the contract [Doc 12]."} onCite={(n) => cited.push(n)} />);

    fireEvent.click(screen.getByRole("button", { name: "Doc 12" }));
    fireEvent.click(screen.getByRole("button", { name: "Doc 3" }));

    expect(cited).toEqual([12, 3]);
    expect(screen.getByText(/per the contract/)).toBeInTheDocument();
  });

  it("keeps paragraphs", () => {
    const { container } = render(<CitedText text={"One.\n\nTwo."} onCite={() => {}} />);
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });
});

describe("EvidenceCard", () => {
  it("shows the excerpt verbatim and opens its source", () => {
    const opened: Evidence[] = [];
    render(<EvidenceCard evidence={evidence} onOpen={(e) => opened.push(e)} />);
    expect(screen.getByText("The notice period is 30 days.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open source" }));
    expect(opened).toEqual([evidence]);
  });

  it("offers no source button when there is no file", () => {
    render(<EvidenceCard evidence={{ ...evidence, has_source: false }} onOpen={() => {}} />);
    expect(screen.queryByRole("button", { name: "Open source" })).toBeNull();
  });

  it("locates documents by page and recordings by time", () => {
    expect(whereLabel(evidence)).toBe("1:15–1:01:40");
    expect(whereLabel({ ...evidence, location: "p. 7 · Article 7" })).toBe("p. 7 · Article 7");
    expect(formatTime(5)).toBe("0:05");
  });
});

describe("AnswerLanguage", () => {
  it("picks the answer language per question", () => {
    const picked: string[] = [];
    render(<AnswerLanguage value="en" languages={["en", "sr"]} onChange={(l) => picked.push(l)} />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "sr" } });
    expect(picked).toEqual(["sr"]);
  });
});

describe("SourcePanel", () => {
  it("invites a pick until a source is open, then shows its excerpt beside the view", () => {
    globalThis.fetch = vi.fn(async () => new Response(JSON.stringify({ kind: "document", variants: [] }))) as any;
    const { rerender } = render(<SourcePanel evidence={null} onClose={() => {}} />);
    expect(screen.getByText("Open a source to review it here.")).toBeInTheDocument();

    const closed: boolean[] = [];
    rerender(<SourcePanel evidence={evidence} onClose={() => closed.push(true)} />);
    expect(screen.getByRole("complementary", { name: "Source review" })).toHaveClass("open");
    expect(screen.getByText("The notice period is 30 days.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(closed).toEqual([true]);
  });
});
