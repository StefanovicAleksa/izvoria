import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { SessionContext } from "../session";
import { fakeGateway } from "../test-utils";
import { AccountsView } from "./AccountsView";

const session = {
  me: { stage: "signed_in", console: "firm", languages: ["en"], firm: { slug: "legal-demo", name: "Legal demo" },
        user: { id: "me", firm: "legal-demo", email: "me@firm.com", name: "", role: "admin" } },
  answerLanguage: "en", setAnswerLanguage: () => {}, signOut: () => {}, reload: () => {},
} as any;

const USERS = { users: [
  { id: "me", firm: "legal-demo", email: "me@firm.com", name: "", role: "admin", disabled: false },
  { id: "u2", firm: "legal-demo", email: "ana@firm.com", name: "Ana", role: "member", disabled: false },
] };

const view = () => render(
  <SessionContext.Provider value={session}><MemoryRouter><AccountsView /></MemoryRouter></SessionContext.Provider>,
);

describe("AccountsView", () => {
  it("keeps what was typed when adding an account fails, and says why", async () => {
    fakeGateway({
      "GET /firm/api/users": { body: USERS },
      "POST /firm/api/users": { status: 400, body: { error: "ana@firm.com already has an account at legal-demo." } },
    });
    view();
    await screen.findByText("ana@firm.com");
    const form = screen.getByText("Add an account").closest("form")!;
    const email = form.querySelector("input[type=email]") as HTMLInputElement;
    fireEvent.change(email, { target: { value: "ana@firm.com" } });
    fireEvent.change(form.querySelector("input[type=password]")!, { target: { value: "long enough pw" } });
    fireEvent.click(screen.getByRole("button", { name: "Add account" }));

    expect(await screen.findByText(/already has an account/)).toBeInTheDocument();
    expect(email.value).toBe("ana@firm.com");
  });

  it("clears the form once the account is made", async () => {
    const calls = fakeGateway({
      "GET /firm/api/users": { body: USERS },
      "POST /firm/api/users": { status: 201, body: {} },
    });
    view();
    await screen.findByText("ana@firm.com");
    const form = screen.getByText("Add an account").closest("form")!;
    const email = form.querySelector("input[type=email]") as HTMLInputElement;
    fireEvent.change(email, { target: { value: "new@firm.com" } });
    fireEvent.change(form.querySelector("input[type=password]")!, { target: { value: "long enough pw" } });
    fireEvent.click(screen.getByRole("button", { name: "Add account" }));

    expect(await screen.findByText(/Account created/)).toBeInTheDocument();
    expect(email.value).toBe("");
    expect(calls.find((c) => c.method === "POST")!.body).toEqual(
      { email: "new@firm.com", name: "", password: "long enough pw", role: "member" });
  });

  it("keeps the reset form open when the new password is refused", async () => {
    fakeGateway({
      "GET /firm/api/users": { body: USERS },
      "POST /firm/api/users/u2/password": { status: 400, body: { error: "Passwords need at least 10 characters." } },
    });
    view();
    fireEvent.click(await screen.findByRole("button", { name: "Reset password" }));
    fireEvent.change(screen.getByPlaceholderText("New password"), { target: { value: "0123456789" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText(/at least 10 characters/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("New password")).toBeInTheDocument();
  });
});
