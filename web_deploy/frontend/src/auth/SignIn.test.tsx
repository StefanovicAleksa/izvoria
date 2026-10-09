import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "../App";
import { api } from "../api/client";
import { ACCOUNT_STAGE, FIRM_STAGE, fakeGateway } from "../test-utils";

beforeEach(() => {
  localStorage.clear();
  window.history.pushState({}, "", "/");
});

describe("Signing in", () => {
  it("asks for the firm first, then for the person's own account", async () => {
    let me: any = FIRM_STAGE;
    const calls = fakeGateway({
      "/auth/me": () => ({ body: me }),
      "POST /auth/firm": () => {
        me = ACCOUNT_STAGE;
        return { body: { firm: ACCOUNT_STAGE.firm } };
      },
      "POST /auth/login": { status: 401, body: { error: "Wrong email or password." } },
    });
    render(<App />);

    fireEvent.change(await screen.findByLabelText("Firm sign-in name"), { target: { value: "Legal-Demo " } });
    fireEvent.change(screen.getByLabelText("Firm password"), { target: { value: "shared secret!" } });
    fireEvent.click(screen.getByLabelText("Remember this firm on this device for 30 days"));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByText("Legal demo")).toBeInTheDocument();
    expect(window.location.pathname).toBe("/signin/account");
    expect(localStorage.getItem("onyx.firm")).toBe("legal-demo");

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@b.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "secret secret" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByText("Wrong email or password.")).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "POST").map((c) => [c.url, c.body])).toEqual([
      ["/auth/firm", { firm: "legal-demo", password: "shared secret!", remember: true }],
      ["/auth/login", { email: "a@b.com", password: "secret secret" }],
    ]);
  });

  it("explains a wrong firm password and links to access requests and help", async () => {
    fakeGateway({
      "/auth/me": { body: FIRM_STAGE },
      "POST /auth/firm": { status: 401, body: { error: "Wrong firm name or password." } },
    });
    render(<App />);
    fireEvent.change(await screen.findByLabelText("Firm sign-in name"), { target: { value: "x-firm" } });
    fireEvent.change(screen.getByLabelText("Firm password"), { target: { value: "nope nope nope" } });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(await screen.findByText("Wrong firm name or password.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Request access for your firm" })).toHaveAttribute("href", "/request-access");
    expect(screen.getByRole("link", { name: "Help & support" })).toHaveAttribute("href", "/help");
  });

  it("can switch to another firm from the account step", async () => {
    let me: any = ACCOUNT_STAGE;
    const calls = fakeGateway({
      "/auth/me": () => ({ body: me }),
      "POST /auth/firm/forget": () => {
        me = FIRM_STAGE;
        return { body: { ok: true } };
      },
    });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Switch firm" }));
    expect(await screen.findByLabelText("Firm sign-in name")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/auth/firm/forget")).toBe(true);
  });

  it("goes back to the sign-in page when the gateway says the session is gone", async () => {
    let me: any = ACCOUNT_STAGE;
    fakeGateway({ "/auth/me": () => ({ body: me }), "/api/x": { status: 401, body: { error: "Please sign in.", kind: "auth" } } });
    render(<App />);
    await screen.findByText("Legal demo");
    me = FIRM_STAGE;
    await api.get("/api/x").catch(() => {});
    await waitFor(() => expect(screen.getByLabelText("Firm sign-in name")).toBeInTheDocument());
  });
});

const SIGNED_IN_ADMIN = {
  stage: "signed_in", console: "firm", firm: { slug: "legal-demo", name: "Legal demo", default_language: "en" },
  user: { id: "me", firm: "legal-demo", email: "me@firm.com", name: "", role: "admin" }, languages: ["en", "sr"],
};

describe("Signed in while the GPU server wakes", () => {
  it("still lets an administrator manage accounts and anyone reach their account and help", async () => {
    window.history.pushState({}, "", "/accounts");
    fakeGateway({
      "/auth/me": { body: SIGNED_IN_ADMIN },
      "/box/status": { body: { state: "booting", message: "Loading the models", error: "", queued: 0, ingesting: 0, reports_running: 0 } },
      "/firm/api/users": { body: { users: [] } },
    });
    render(<App />);

    expect(await screen.findByText("Add an account")).toBeInTheDocument();
    expect(screen.queryByText("Waking the appliance")).toBeNull();
    expect(screen.getByRole("link", { name: "Help & support" })).toHaveAttribute("href", "/help");

    fireEvent.click(screen.getByRole("link", { name: "Chat" }));
    expect(await screen.findByText("Waking the appliance")).toBeInTheDocument();
  });

  it("sends an owner to the console wherever they land", async () => {
    window.history.pushState({}, "", "/chat");
    fakeGateway({
      "/auth/me": { body: { ...SIGNED_IN_ADMIN, console: "admin", firm: { slug: "admin", name: "Admin console" },
                            user: { ...SIGNED_IN_ADMIN.user, role: "owner", firm: "admin" } } },
      "/admin/api/firms": { body: { firms: [], console_has_password: false } },
    });
    render(<App />);
    expect(await screen.findByText(/The console has no firm password yet/)).toBeInTheDocument();
    expect(window.location.pathname).toBe("/admin/firms");
  });
});

describe("The client", () => {
  it("only treats a lost session as signing out, not a wrong password", async () => {
    const seen: string[] = [];
    window.addEventListener("onyx:signed-out", () => seen.push("out"));
    fakeGateway({
      "POST /auth/login": { status: 401, body: { error: "Wrong email or password." } },
      "/api/a": { status: 401, body: { error: "Please sign in.", kind: "auth" } },
      "POST /auth/firm": { status: 401, body: { error: "Sign in as your firm first.", kind: "firm_stage" } },
    });
    await api.post("/auth/login").catch(() => {});
    expect(seen).toEqual([]);
    await api.get("/api/a").catch(() => {});
    await api.post("/auth/firm").catch(() => {});
    expect(seen).toEqual(["out", "out"]);
  });
});
