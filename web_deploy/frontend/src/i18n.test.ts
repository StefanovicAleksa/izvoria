import { SERBIAN } from "./i18n";

// Every module's source, as text (Vite's glob import): the app's own files, not its tests
const SOURCES = import.meta.glob(["./**/*.{ts,tsx}", "!./**/*.test.{ts,tsx}"], {
  query: "?raw", import: "default", eager: true,
}) as Record<string, string>;

// Every literal the app passes to t(): each one must have a Serbian entry, or a Serbian user sees English
function literals(): Set<string> {
  const found = new Set<string>();
  for (const source of Object.values(SOURCES)) {
    // t("…"), and both arms of t(cond ? "…" : "…")
    for (const call of source.matchAll(/\bt\(([^()]*)\)/g)) {
      const args = call[1].replace(/[!=]==?\s*"[^"]*"/g, ""); // the operands of a comparison are not shown
      for (const literal of args.matchAll(/"((?:[^"\\]|\\.)*)"/g)) found.add(JSON.parse(`"${literal[1]}"`));
    }
  }
  return found;
}

// Labels kept in arrays and passed to t() through a variable
const INDIRECT = [
  "People", "Clients, counterparties and institutions", "Products, funds, drugs, devices or software", "Places",
  "Internal acronyms and in-house terms", "Laws, regulations, standards and cases",
  "Partners, doctors, analysts, frequent guests.", "Organisations that come up often.",
  "What your firm uses or works with.", "Courts, hospitals, exchanges, local towns.", "With what each one stands for.",
  "Those cited often.", "Ready", "Ingesting files", "Starting", "Stopping", "Error", "Offline", "Connecting…",
  "Owner", "Administrator", "Member", "Emailed", "Sending", "Email failed", "Not emailed",
  "Wrong firm name or password.", "Wrong email or password.", "Too many attempts. Wait a few minutes and try again.",
  "Video", "Audio", "Documents", "Images",
  // Ingestion steps and their states (views/ingestStages.ts, components/StageList.tsx)
  "Preparing files", "Analyzing audio", "Reading documents", "Analyzing images", "Contextualizing",
  "Analyzing visuals", "Indexing", "Waiting", "In progress", "Done", "Stopped", "Downloading videos",
  // RequestAccess's field helpers
  "Firm name", "Your name", "Phone (optional)", "Your role at the firm (optional)",
  "Field of work", "For example law, medicine, finance, consulting.",
  "What your firm does", "For whom, and in which jurisdictions or markets.",
  "Practice areas or specialities",
  "One per line, with a short description, e.g. “M&A: share and asset deals, due diligence”.",
  "What you will bring to Izvoria",
  "Meetings, hearings, consultations, lectures, dictations, contracts? Who speaks, and how formally?",
  "Terms transcription already gets wrong", "If you have used speech recognition before: “heard X, should be Y”.",
  "Languages of your material", "Roughly what share is in English and what in Serbian.",
  "English terms in Serbian speech",
  "Do Serbian speakers say terms like “due diligence” or “stop loss” in English? Which ones, and where do they use a Serbian word instead?",
  "Latin", "Is Latin used in speech, e.g. diagnoses or legal maxims?",
];

describe("the Serbian catalog", () => {
  it("reads the app's sources", () => {
    expect(Object.keys(SOURCES).length).toBeGreaterThan(10);
  });

  it("translates every text the app shows", () => {
    const missing = [...literals(), ...INDIRECT].filter((key) => !(key in SERBIAN));
    expect([...new Set(missing)].sort()).toEqual([]);
  });
});
