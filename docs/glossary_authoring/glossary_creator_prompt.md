# Glossary creator prompt

A prompt for Claude that turns a filled-in firm profile (`firm_profile_template.md`) into the per-language
glossaries Izvoria reads at ingestion (`data/glossaries/<lang>/<industry>.json`).

**How to use it**

1. Fill in `firm_profile_template.md` for the firm.
2. Copy everything between the `BEGIN PROMPT` and `END PROMPT` lines below into Claude (the best model you have,
   with extended thinking on). Replace the three placeholders:
   - `{{FIRM_PROFILE}}`: the filled-in profile.
   - `{{LANGUAGES}}`: the served languages, currently `en, sr`.
   - `{{SCOPE}}`: `all`, or a comma-separated list of sub-domain labels when one answer can't hold every
     sub-domain. Run it again per batch; the first run (or `all`) also writes `base_terms`.
3. Save Claude's JSON answer to a file and split it into the glossary files:
   `venv/bin/python -m scripts.validate_glossary --from-claude answer.json --write`
   (merges into existing files, so batches accumulate). Then check everything:
   `venv/bin/python -m scripts.validate_glossary <industry>`.
4. Read the `review` list in the answer. Terms in it are not in the glossary. Add any you confirm by hand.
5. Measure before trusting it: `scripts/eval/glossary_ablation.py` (see its docstring).
6. On the hosted web version, attach it to the firm (it reads no `data/glossaries/` files of its own): the owner
   console's Firms → *Glossary*, per language, or on the VPS
   `admin_cli set-glossary <firm> --from-claude answer.json` for every language at once
   (`docs/web_deploy.md`, "Firm glossaries").

**Why the prompt says what it says.** A glossary drives three stages, and each one rewards precision, not volume:

- The **sub-domain router** (Llama 3.2 3B, Gemma 4 E4B on Serbian appliances) sees only each sub-domain's first ten
  `anchor_words` + `terms`, and rates every sub-domain 0–10 against a preview of the recording.
- **Whisper hotwords** are the anchor words of the routed sub-domains, at most 40 in total. Whisper reads them as
  "text that came before", which pushes it towards those spellings.
- **Domain phonetic correction** looks for spans that sound like *exactly one* glossary term and replaces them
  with the term's spelling, exactly as written. A misspelled term is written into transcripts. A common word
  in the glossary sounds like ordinary speech everywhere. Two terms that sound alike cancel each other out.
  Serbian has no inflection handling yet, so dictionary-form terms are written over inflected speech. The
  `forms` field gathers what a future matcher needs.

---

```text
==================== BEGIN PROMPT ====================

You are building speech-recognition glossaries for Izvoria, an on-premise AI appliance that transcribes a firm's
recordings with Whisper and then answers questions only from those transcripts, with a citation for every claim.
The firm's lawyers, doctors or analysts rely on those answers. A transcript error on a key term (a drug, a
statute, a counterparty, an instrument) becomes a wrong or unfindable answer. Your glossaries are the main tool
for fixing such errors, so their accuracy matters more than anything else in this task.

<firm_profile>
{{FIRM_PROFILE}}
</firm_profile>

Languages to produce glossaries for: {{LANGUAGES}}
Scope of this run: {{SCOPE}}   ("all" = design every sub-domain and write base_terms; otherwise write only the
listed sub-domains, using exactly those labels, and leave base_terms empty)

## How your glossary is used. Optimise for this, not for size.

1. ROUTER. A small language model reads a short preview of each recording and scores every sub-domain 0–10.
   It only sees each sub-domain's first 10 entries of `anchor_words` followed by `terms`. Those entries must
   make the sub-domain unmistakable and must not describe any other sub-domain.
2. HOTWORDS. The anchor words of the chosen sub-domains (at most 40 in total) are given to Whisper as prior
   context, which biases it towards those exact spellings. `base_terms` are used only when no sub-domain is
   chosen.
3. PHONETIC CORRECTION. After transcription, a span of 1–5 words that sounds like exactly one glossary term (and
   is not already that term) becomes a candidate. A small model confirms or rejects it. Confirmed spans are
   replaced by the term exactly as you spelled it. Consequences:
   - A misspelled term is written into every transcript where it is confirmed. Never include a term you are not
     certain is real and correctly spelled.
   - A common word ("market", "risk", "pain", "contract", "zakon") is already transcribed correctly by Whisper.
     As a term it adds nothing and makes ordinary speech look like a candidate. Leave it out.
   - Two terms that sound almost the same make each other ambiguous, so neither gets corrected. Include both
     only when both genuinely occur (ileum / ilium). Never include spelling variants of one term.
   - English plurals ending in -s are handled automatically: list the singular. Irregular plurals
     (metastasis → metastases, criterion → criteria, corpus delicti forms) go in `forms`.

## What makes a good term

Prefer, in this order:
1. Proper nouns from the profile: people, institutions, clients, products, places, laws and cases. Copy their
   spelling exactly from the profile. Do not "correct" it.
2. Proper nouns the domain is known to use: regulators, courts, standard bodies, drug brand and generic names,
   index names, central bankers, landmark cases.
3. Acronyms and initialisms (as written: "EBITDA", "MRI", "ZPP", "SPA"). Spoken letter by letter or as a word,
   Whisper often splits or mangles them.
4. Latin and Greek terminology (diagnoses, anatomy, legal maxims), and foreign loanwords spoken inside the
   language.
5. Multi-word technical terms whose words are ordinary but whose combination is specific ("forward guidance",
   "value area high", "res judicata", "ugovor o građenju").
6. Rare single-word technical terms ("contango", "backwardation", "tamponada", "fibrilacija").

Exclude: everyday words, generic verbs and adjectives, words any general-purpose transcriber writes correctly,
numbers and dates, full sentences, slogans, and anything you would have to invent or guess the spelling of.

Size: be extensive within these rules. A typical sub-domain has 8 anchor words and 40–150 terms. A narrow one
may have fewer. Never pad a list with weak terms to reach a number. One wrong term costs more than ten missing
ones.

## Sub-domains

- Derive them from the profile's sub-industries (section 3). Split or merge only when the recordings would
  plainly split differently. Usually 3–12 sub-domains.
- Label: UPPER_SNAKE_CASE ASCII, in English, identical in every language's file ("EMPLOYMENT_LAW", "CARDIOLOGY").
- `description`: one sentence, in English, on what the sub-domain covers.
- `anchor_words`: 5–8 entries, the most distinctive spoken words of that sub-domain. They are its router
  description and its Whisper hotwords. Each anchor appears in exactly one sub-domain. Put the most telling
  first.
- `terms`: everything else for that sub-domain. A term may appear in more than one sub-domain's `terms` when it
  truly belongs to both.
- `base_terms`: 15–40 terms that run through the whole industry regardless of sub-domain.

## Languages

Produce one glossary per language in {{LANGUAGES}}, with the same sub-domain labels in each. A glossary holds
the words as SPOKEN in that language's recordings, not a translation of the English list.

Classify every term:
- shared: spoken and written identically in every listed language. Examples: Latin diagnoses ("infarctus
  myocardii"), drug INNs, tickers and most acronyms, institution names used untranslated ("Bundesbank"), and
  English jargon that the profile says (or that you know with confidence) speakers use as-is inside Serbian
  speech ("stop loss", "due diligence"). A shared term must have byte-identical spelling in every file.
- localized: the language has its own word, and that is what speakers say. Example: English "liquidity",
  Serbian "likvidnost". Put each language's own spelling in its own file. If Serbian speakers use both the
  native word and the English one, include both in the Serbian file: the native word as localized, the English
  word as shared.
- An English word with an established Serbian adapted spelling ("lizing", "menadžment", "fjučers", "hedž
  fond") is localized under that adapted spelling. Use an adapted spelling only if it is genuinely in use, not
  one you construct.

Serbian specifics:
- Latin script only, with č ć đ š ž. Never Cyrillic.
- Serbian is inflected and the corrector cannot yet produce case endings. For every localized Serbian noun or
  noun phrase, list in `forms` the inflected forms that actually occur in speech: singular and plural cases,
  each spelled out in full. For a phrase, inflect the head ("ugovor o radu", "ugovora o radu", "ugovoru o radu",
  "ugovorom o radu", "ugovori o radu", "ugovora o radu", ...). Remove duplicates. Include only forms you are
  certain of.
- Indeclinable shared terms (acronyms, Latin) get no Serbian forms, except where Serbian genuinely declines
  them in speech ("ECB-a", "ECB-u"). List those only if they are standard.

## Self-checks before you answer

For each term:
1. Is it real, and is this its exact standard spelling and capitalisation? If you are not certain, it goes to
   `review` instead of the glossary.
2. Would a general transcriber already write it correctly? If yes, and it is also an everyday word, drop it.
3. Does it sound like a common word of that language? If yes, keep it only if it is valuable, and describe the
   clash in `collision_risk` (e.g. "Bund" — "bound", "burn").
4. Does it sound like another term in the same glossary? If yes, drop the variant or confirm both are real.
5. Is it shared? Then it is spelled identically in every file.

## Output

Answer with exactly one JSON object and nothing else: no prose, no Markdown fences. Its keys are the language
codes from {{LANGUAGES}}, plus "review". Schema:

{
  "<lang>": {
    "industry": "<industry key from the profile>",
    "language": "<lang>",
    "base_terms": ["..."],
    "sub_domains": {
      "<LABEL>": {
        "description": "<one English sentence>",
        "anchor_words": ["..."],
        "terms": ["..."]
      }
    },
    "term_details": {
      "<term exactly as listed above>": {
        "kind": "proper_noun | acronym | latin | loanword | technical",
        "shared": true | false,
        "forms": ["<inflected or irregular-plural forms; omit the key if none>"],
        "collision_risk": "<omit the key if none>"
      }
    }
  },
  "review": [
    {"term": "...", "language": "<lang>", "sub_domain": "<LABEL or BASE>", "reason": "<why you are unsure>"}
  ]
}

Rules for the output:
- Every term in base_terms, anchor_words and terms has an entry in term_details for that language.
- `forms` never repeats the term itself.
- No term appears twice within one list. Anchor words do not repeat in the same sub-domain's terms.
- Terms in `review` appear in no glossary list.

## Worked mini-example (format only; your terms come from the profile)

{
  "en": {
    "industry": "prop_trading_macro",
    "language": "en",
    "base_terms": ["basis point", "hawkish", "dovish"],
    "sub_domains": {
      "EUROPEAN_CENTRAL_BANK": {
        "description": "ECB policy meetings, rate decisions and the euro-area bond market.",
        "anchor_words": ["ECB", "Lagarde", "Bund", "BTP", "TLTRO"],
        "terms": ["Deposit Facility Rate", "Bundesbank", "Schatz", "Bobl"]
      }
    },
    "term_details": {
      "basis point": {"kind": "technical", "shared": false},
      "hawkish": {"kind": "technical", "shared": true},
      "dovish": {"kind": "technical", "shared": true},
      "ECB": {"kind": "acronym", "shared": true},
      "Lagarde": {"kind": "proper_noun", "shared": true},
      "Bund": {"kind": "loanword", "shared": true, "collision_risk": "bound, burn"},
      "BTP": {"kind": "acronym", "shared": true},
      "TLTRO": {"kind": "acronym", "shared": true},
      "Deposit Facility Rate": {"kind": "technical", "shared": false},
      "Bundesbank": {"kind": "proper_noun", "shared": true},
      "Schatz": {"kind": "loanword", "shared": true, "collision_risk": "shots"},
      "Bobl": {"kind": "loanword", "shared": true, "collision_risk": "bobble"}
    }
  },
  "sr": {
    "industry": "prop_trading_macro",
    "language": "sr",
    "base_terms": ["bazni poen", "hawkish", "dovish"],
    "sub_domains": {
      "EUROPEAN_CENTRAL_BANK": {
        "description": "ECB policy meetings, rate decisions and the euro-area bond market.",
        "anchor_words": ["ECB", "Lagarde", "Bund", "BTP", "TLTRO"],
        "terms": ["depozitna stopa", "Bundesbank", "Schatz", "Bobl"]
      }
    },
    "term_details": {
      "bazni poen": {"kind": "technical", "shared": false,
                     "forms": ["baznog poena", "baznom poenu", "bazni poeni", "baznih poena", "baznim poenima"]},
      "hawkish": {"kind": "technical", "shared": true},
      "dovish": {"kind": "technical", "shared": true},
      "ECB": {"kind": "acronym", "shared": true, "forms": ["ECB-a", "ECB-u", "ECB-om"]},
      "Lagarde": {"kind": "proper_noun", "shared": true},
      "Bund": {"kind": "loanword", "shared": true},
      "BTP": {"kind": "acronym", "shared": true},
      "TLTRO": {"kind": "acronym", "shared": true},
      "depozitna stopa": {"kind": "technical", "shared": false,
                          "forms": ["depozitne stope", "depozitnoj stopi", "depozitnu stopu", "depozitnom stopom"]},
      "Bundesbank": {"kind": "proper_noun", "shared": true},
      "Schatz": {"kind": "loanword", "shared": true},
      "Bobl": {"kind": "loanword", "shared": true}
    }
  },
  "review": [
    {"term": "Lagardeova", "language": "sr", "sub_domain": "EUROPEAN_CENTRAL_BANK",
     "reason": "possessive form of a foreign name; unsure whether it is standard"}
  ]
}

==================== END PROMPT ====================
```
