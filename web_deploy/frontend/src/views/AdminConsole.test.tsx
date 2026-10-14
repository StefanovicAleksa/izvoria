import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describeBox } from "../box";
import { fakeGateway } from "../test-utils";
import { AdminConsole } from "./AdminConsole";

const view = (section: "firms" | "requests" | "server") => render(<MemoryRouter><AdminConsole section={section} /></MemoryRouter>);

describe("The owner's firms", () => {
  it("flags a firm nobody can sign in to, and keeps a refused new firm's details", async () => {
    fakeGateway({
      "GET /admin/api/firms": { body: { console_has_password: true, firms: [
        { slug: "legal-demo", name: "Legal demo", default_language: "en", is_demo: true, disabled: false,
          has_password: false, users: 2 }] } },
      "POST /admin/api/firms": { status: 400, body: { error: "A firm called 'legal-demo' already exists." } },
    });
    view("firms");
    expect(await screen.findByText("No firm password")).toBeInTheDocument();

    const form = screen.getByText("New firm").closest("form")!;
    const inputs = form.querySelectorAll("input");
    fireEvent.change(inputs[0], { target: { value: "Legal again" } });
    fireEvent.change(inputs[1], { target: { value: "legal-demo" } });
    fireEvent.change(inputs[2], { target: { value: "shared secret!" } });
    fireEvent.click(screen.getByRole("button", { name: "Create firm" }));

    expect(await screen.findByText(/already exists/)).toBeInTheDocument();
    expect((inputs[1] as HTMLInputElement).value).toBe("legal-demo");
  });
});

describe("A firm's glossaries", () => {
  const FIRMS = { console_has_password: true, firms: [
    { slug: "petrovic", name: "Petrović", default_language: "sr", is_demo: false, disabled: false, has_password: true, users: 1 }] };
  const MEDICAL = { language: "en", industry: "medical", source: "builtin", builtin: "medical", original_name: null,
    updated_at: "2026-10-14T10:00:00+00:00", terms: 120, sub_domains: ["CARDIOLOGY"], warnings: [] };

  it("shows each language's glossary, picks a built-in and shows a refused upload's errors", async () => {
    const calls = fakeGateway({
      "GET /admin/api/firms": { body: FIRMS },
      "GET /admin/api/firms/petrovic/glossaries": { body: {
        glossaries: { en: null, sr: null }, builtins: { en: ["finance", "general", "medical"], sr: [] } } },
      "PUT /admin/api/firms/petrovic/glossaries/en": (call) => call.body.builtin
        ? { body: { glossaries: { en: MEDICAL, sr: null }, builtins: { en: ["finance", "general", "medical"], sr: [] } } }
        : { status: 400, body: { error: "The glossary has 1 error(s).", errors: ["en: sub-domain 'x' is not UPPER_SNAKE_CASE"], kind: "invalid" } },
    });
    view("firms");
    fireEvent.click(await screen.findByRole("button", { name: "Glossary" }));
    expect(await screen.findAllByText("General (shared)", { selector: "div" })).toHaveLength(2);

    fireEvent.change(screen.getAllByLabelText("Built-in glossary")[0], { target: { value: "medical" } });
    expect(await screen.findByText(/Built-in: medical · 1 sub-domains · 120 terms · 2026-10-14/)).toBeInTheDocument();

    const file = new File([JSON.stringify({ industry: "law", base_terms: ["Petrović"] })], "law.json", { type: "application/json" });
    fireEvent.change(screen.getByTestId("glossary-file-en"), { target: { files: [file] } });
    expect(await screen.findByText(/is not UPPER_SNAKE_CASE/)).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "PUT").map((c) => c.body)).toEqual([
      { builtin: "medical" },
      { glossary: { industry: "law", base_terms: ["Petrović"] }, file_name: "law.json" },
    ]);
  });
});

describe("The YouTube account the GPU server downloads with", () => {
  it("shows the cookies' state and uploads a new file", async () => {
    const signed = { present: true, cookies: 12, signed_in: true, expires: 1790000000, expired: false, updated_at: 1760000000, refused_at: null };
    const calls = fakeGateway({
      "GET /admin/api/box": { body: { state: "stopped", message: "", error: "", queued: 0, ingesting: 0, reports_running: 0, paused: false } },
      "GET /admin/api/youtube-cookies": { body: { present: false, cookies: 0, signed_in: false, expires: null, expired: false, updated_at: null, refused_at: 1760000000 } },
      "PUT /admin/api/youtube-cookies": { body: signed },
    });
    view("server");
    expect(await screen.findByText("No cookies: YouTube links fail on the GPU server.")).toBeInTheDocument();
    expect(screen.getByText(/YouTube refused a download as a bot on 2025-10-09/)).toBeInTheDocument();

    const file = new File(["# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t1790000000\tSAPISID\tx\n"], "cookies.txt");
    fireEvent.change(screen.getByTestId("youtube-cookies-file"), { target: { files: [file] } });

    expect(await screen.findByText(/Signed-in cookies · 12 cookies/)).toBeInTheDocument();
    expect(calls.find((c) => c.method === "PUT")!.body.cookies).toContain("SAPISID");
  });
});

describe("The owner's requests", () => {
  it("shows a sign-up's profile and mail status, and marks it handled", async () => {
    const calls = fakeGateway({
      "GET /admin/api/requests": { body: { mail_enabled: true, requests: [{
        id: "r1", kind: "signup", created_at: 1760000000, email: "ana@firm.rs", name: "Ana", firm: "Petrović",
        subject: "", language: "sr", status: "new", mail_status: "failed", mail_error: "smtp is down",
        payload: { phone: "+381", profile: { industry: "law", proper_nouns: {}, languages: { share: "sr 80%" } } },
      }] } },
      "POST /admin/api/requests/r1/status": { body: {} },
      "POST /admin/api/requests/r1/resend": { body: {} },
    });
    view("requests");

    expect(await screen.findByText("Petrović")).toBeInTheDocument();
    expect(screen.getByText("Email failed")).toHaveAttribute("title", "smtp is down");
    expect(screen.getByText("sr 80%")).toBeInTheDocument();
    expect(screen.getByText("No names given: collect them during setup.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Send email again" }));
    fireEvent.click(screen.getByRole("button", { name: "Mark handled" }));
    await screen.findByText("Petrović");
    expect(calls.filter((c) => c.method === "POST").map((c) => [c.url, c.body])).toEqual([
      ["/admin/api/requests/r1/resend", {}],
      ["/admin/api/requests/r1/status", { status: "handled" }],
    ]);
  });
});

describe("describeBox", () => {
  const status = (state: string, paused = false) =>
    ({ state, paused, message: "", error: "", queued: 0, ingesting: 0, reports_running: 0 }) as any;

  it.each([
    [null, "", "Connecting…"],
    [status("ready"), "ready", "Ready"],
    [status("ready", true), "busy", "Ingesting files"],
    [status("booting"), "pending", "Starting"],
    [status("gpu_unavailable"), "pending", "Starting"],
    [status("stopping"), "pending", "Stopping"],
    [status("error"), "error", "Error"],
    [status("stopped"), "", "Offline"],
  ])("%o", (input, dot, label) => {
    expect(describeBox(input)).toEqual({ dot, label });
  });
});
