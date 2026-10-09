import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describeBox } from "../box";
import { fakeGateway } from "../test-utils";
import { AdminConsole } from "./AdminConsole";

const view = (section: "firms" | "requests") => render(<MemoryRouter><AdminConsole section={section} /></MemoryRouter>);

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
    [status("booting"), "waking", "Starting"],
    [status("gpu_unavailable"), "waking", "Starting"],
    [status("stopping"), "waking", "Stopping"],
    [status("error"), "error", "Error"],
    [status("stopped"), "", "Offline"],
  ])("%o", (input, dot, label) => {
    expect(describeBox(input)).toEqual({ dot, label });
  });
});
