# Firm profile — glossary input

Fill this in for one firm and paste it into `glossary_creator_prompt.md` where it says `{{FIRM_PROFILE}}`.
Write in English or Serbian, whichever is easier. Leave a section empty if you don't know it; don't guess.

**Confidentiality.** This profile is sent to Claude, a cloud model, at setup time. It describes the firm, not its
data. Never paste transcripts, documents, case files or patient details here. If the firm won't allow its client
or counterparty names to leave the premises, leave section 6 out and add those names to the glossary by hand.

---

## 1. Firm name

<!-- Only used to label the output. -->

## 2. Industry key

<!-- One lowercase key, which becomes the file name (data/glossaries/<lang>/<industry>.json) and
     APPLIANCE_INDUSTRY. Examples: legal, medical, prop_trading_macro. -->

## 3. Sub-industries / specialised domains

<!-- One per line, with a sentence on what each covers. These become the sub-domains the router picks per
     recording, so split them the way recordings split: one recording is usually about one or two of them.
     Example:
     - M&A — share and asset deals, due diligence, SPAs, warranties and indemnities
     - Employment — dismissals, collective agreements, labour inspections -->

## 4. Firm description

<!-- What the firm does, for whom, in which jurisdictions/markets. Short or long. -->

## 5. What the data will be

<!-- What will be ingested: meetings, lectures, depositions, consultations, ward rounds, trading-desk calls,
     webinars, dictations? Who speaks (lawyers, judges, doctors, patients, traders)? How formal is the speech?
     Any recurring formats (morning meeting, case conference, earnings call)? -->

## 6. Proper nouns the recordings will contain

<!-- The highest-value glossary entries: the speech recogniser has never seen these names, and no general
     vocabulary contains them. Exact spelling matters, because a glossary term is written into transcripts
     exactly as spelled here. -->

- **People** (partners, doctors, analysts, frequent guests):
- **Clients / counterparties / institutions:**
- **Products, funds, drugs, devices, software the firm uses:**
- **Places** (courts, hospitals, exchanges, local towns):
- **Internal acronyms and in-house jargon** (with what they stand for):
- **Laws, regulations, standards, case names cited often:**

## 7. Languages spoken in the recordings

<!-- Which of the served languages (en, sr) the recordings are in, and roughly what share each. -->

- **Languages and share:**
- **Do Serbian speakers use English terms as-is?** (e.g. say "liquidity", "stop loss", "due diligence" instead of
  a Serbian word.) List the ones you know, and the ones where they use a Serbian word instead:
- **Latin in speech?** (medical diagnoses, legal maxims):

## 8. Terms you already know Whisper gets wrong

<!-- Optional. Real mis-hearings from past transcripts: "heard X, should be Y". These are gold: they say exactly
     which terms the glossary must contain. -->
