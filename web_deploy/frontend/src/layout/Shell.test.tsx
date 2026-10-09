import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SessionContext } from "../session";
import { ThemeProvider } from "../theme";
import { fakeGateway } from "../test-utils";
import { AccountsView } from "../views/AccountsView";
import { Sidebar } from "./Sidebar";

function session(role: "member" | "admin" | "owner", id = "me") {
  return {
    me: {
      stage: "signed_in", console: role === "owner" ? "admin" : "firm", languages: ["en", "sr"],
      firm: role === "owner" ? { slug: "admin", name: "Admin console" } : { slug: "legal-demo", name: "Legal demo" },
      user: { id, firm: "legal-demo", email: "me@firm.com", name: "", role },
    },
    answerLanguage: "en", setAnswerLanguage: () => {}, signOut: () => {}, reload: () => {},
  } as any;
}

const withSession = (role: "member" | "admin" | "owner", element: JSX.Element) =>
  render(
    <ThemeProvider>
      <SessionContext.Provider value={session(role)}>
        <MemoryRouter initialEntries={["/chat"]}>{element}</MemoryRouter>
      </SessionContext.Provider>
    </ThemeProvider>,
  );

describe("Sidebar", () => {
  it("shows the workspace to members, with the current view marked", () => {
    withSession("member", <Sidebar open={false} />);
    expect(screen.getByRole("link", { name: "Chat" })).toHaveClass("active");
    expect(screen.getByRole("link", { name: "Report" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Accounts" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Firms" })).toBeNull();
  });

  it("adds Accounts for a firm's administrators", () => {
    withSession("admin", <Sidebar open={false} />);
    expect(screen.getByRole("link", { name: "Accounts" })).toHaveAttribute("href", "/accounts");
  });

  it("gives the owner the console and no workspace", () => {
    withSession("owner", <Sidebar open={false} />);
    expect(screen.getByRole("link", { name: "Requests" })).toHaveAttribute("href", "/admin/requests");
    expect(screen.queryByRole("link", { name: "Chat" })).toBeNull();
  });
});

describe("Theme", () => {
  beforeEach(() => localStorage.clear());

  it("is dark until switched, and remembers the switch", () => {
    withSession("member", <Sidebar open={false} />);
    expect(document.documentElement.dataset.theme).toBe("dark");
    fireEvent.click(screen.getByRole("button", { name: "Light mode" }));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("onyx.theme")).toBe("light");
  });
});

describe("AccountsView", () => {
  it("lists the firm's people and keeps an administrator from locking themselves out", async () => {
    fakeGateway({
      "/firm/api/users": {
        body: { users: [
          { id: "me", firm: "legal-demo", email: "me@firm.com", name: "", role: "admin", disabled: false },
          { id: "u2", firm: "legal-demo", email: "ana@firm.com", name: "Ana", role: "member", disabled: false },
        ] },
      },
    });
    withSession("admin", <AccountsView />);

    const own = (await screen.findByText("me@firm.com")).closest("tr")!;
    expect(own.querySelector("select")).toBeDisabled();
    expect(own.textContent).not.toContain("Disable");
    const other = screen.getByText("ana@firm.com").closest("tr")!;
    expect(other.querySelector("select")).not.toBeDisabled();
    expect(other.textContent).toContain("Reset password");
  });
});
