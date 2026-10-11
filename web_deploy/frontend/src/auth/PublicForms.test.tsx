import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { fakeGateway } from "../test-utils";
import { HelpSupport } from "./HelpSupport";
import { RequestAccess } from "./RequestAccess";

const page = (element: JSX.Element) => render(<MemoryRouter>{element}</MemoryRouter>);

describe("Request access", () => {
  it("sends the contact and the firm profile, with the hidden field empty and the time it took", async () => {
    const calls = fakeGateway({ "POST /public/signup": { status: 202, body: { ok: true } } });
    page(<RequestAccess />);

    const send = screen.getByRole("button", { name: "Send request" });
    expect(send).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Firm name"), { target: { value: "Petrović & Partners" } });
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "Ana" } });
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "ana@firm.rs" } });
    fireEvent.change(screen.getByLabelText(/Field of work/), { target: { value: "law" } });
    fireEvent.change(screen.getByLabelText(/^People/), { target: { value: "Ana Petrović" } });
    fireEvent.click(send);

    expect(await screen.findByText("Thank you")).toBeInTheDocument();
    const [call] = calls;
    expect(call.body).toMatchObject({
      firm_name: "Petrović & Partners", contact_name: "Ana", email: "ana@firm.rs", website: "", language: "en",
      profile: { industry: "law", proper_nouns: { people: "Ana Petrović", clients: "" }, languages: { share: "" } },
    });
    expect(typeof call.body.elapsed_ms).toBe("number");
  });

  it("keeps the names optional and says what not to share", () => {
    page(<RequestAccess />);
    expect(screen.getByText("Optional")).toBeInTheDocument();
    expect(screen.getByText(/Never include client documents, transcripts or case details/)).toBeInTheDocument();
  });
});

describe("Help & support", () => {
  it("fills in who is asking when signed in and sends the problem", async () => {
    const calls = fakeGateway({ "POST /public/support": { status: 202, body: { ok: true } } });
    const me: any = {
      stage: "signed_in", console: "firm", firm: { slug: "legal-demo", name: "Legal demo" },
      user: { id: "u", firm: "legal-demo", email: "a@firm.com", name: "Ana", role: "member" }, languages: ["en"],
    };
    page(<HelpSupport me={me} />);

    expect(screen.getByLabelText("Email")).toHaveValue("a@firm.com");
    fireEvent.change(screen.getByLabelText("Subject"), { target: { value: "Upload fails" } });
    fireEvent.change(screen.getByLabelText(/What happened/), { target: { value: "It stops at 40%." } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Message sent")).toBeInTheDocument();
    expect(calls[0].body).toMatchObject({ email: "a@firm.com", firm: "legal-demo", subject: "Upload fails",
                                          message: "It stops at 40%.", website: "" });
    expect(screen.getByRole("link", { name: "Back to Izvoria" })).toBeInTheDocument();
  });
});
