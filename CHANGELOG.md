# Changelog

This log tracks **feature-level milestones**, not every commit. It's written for humans (and future Claude Code sessions) trying to understand how the system got to its current shape — one entry per merged feature milestone, newest on top. See `CLAUDE.md` for the architecture these entries plug into.

**Regression fixes name the commit that introduced the problem and the commit that fixed it** (`git show <hash>`), so a repeat of a mistake can be traced straight to its source.

## Handwriting OCR can't read is no longer indexed as text

OCR never invents a line, but on handwriting it writes gibberish. An 1839 letter was indexed as "Analbeig aftea long apok by thelvelist hak mith…", labelled as text read by OCR. The image check now also runs in reverse: an OCR line under 0.7 confidence is kept only if the vision model read the same words. On an image where most lines are that uncertain (handwriting), every line under 0.9 needs that confirmation. `IOcrEngine.recognize` takes a `select_lines` hook so this happens before layout (tables keep their shape), and PDFs are unchanged.

- Withheld lines are stored (`image_descriptions.ocr_withheld_lines`, migration `0002`, the first revision after the baseline), never indexed, and counted in the ingestion report (`image_ocr_lines_withheld`). The excerpt says "Some text on this image could not be read reliably and is not included" when one of them held letters or digits.
- On the seven test images: the letter's 26 lines are withheld. The ski-lift sign's "GIMAr MOntaz MAutino" (0.65–0.76) is kept because the vision model read it too. The timetable's misreads ("585", "888") are withheld and its headers ("Hartf d", 0.79) kept.
- `read_image` now applies the EXIF orientation. OCR of a sideways-stored phone photo had read "1 / 2 60" off a forest path.
- The ingestion report lists only what happened to an image's text. It no longer says "0 text item(s) … were not found by OCR" for an image that lost nothing.
- Autogenerating a migration against `db_test` wrote ten foreign keys back into the revision, because integration tests drop them there. `docs/core/database.md` now generates against a fresh scratch database.

## Old databases catch up through the baseline, not through today's models

Introduced in `9ed948e` (the migration commit), found by the first revision after it (`0002`). A database from before migrations was caught up with `create_all()`, which builds a missing table from *today's* models. That included columns from later revisions, whose own migrations then failed adding them. An appliance ingested before image support (no `image_descriptions` table) would have stopped booting with `DuplicateColumn: ocr_withheld_lines`.

The catch-up now runs the frozen statements for the tables the database has, then revision `0001` itself, which creates only the tables and indexes that are missing (`if_not_exists`), in the baseline's shape. Then it upgrades to head like any other database. The integration test's fake old database lacks `image_descriptions`, so it covers this. A copy of the dev database migrates to `0002` with its data kept and no drift. The tests that asserted "revision is 0001" now assert the head.

## Capitals are a heading only when they stand alone

Found by ingesting Project Gutenberg's *Adventures of Sherlock Holmes* (EPUB and TXT). A line in capitals counted as a title, so dialogue ("“BALLARAT.”", "“HUNTER.”") displaced the book's own title. From the fifth story on, every EPUB chunk was filed under "“BALLARAT.” › V. …", and 36 chunks of *The Copper Beeches* were cited as section "“HUNTER.”". The same rule broke contracts. A hard-wrapped, all-caps liability disclaimer was cut into one "heading" per line, and later citations named "OR IN CONNECTION WITH THIS AGREEMENT, EVEN IF ADVISED OF THE" as their section.

A line in capitals is now not a title if it opens a quotation, or ends a sentence (`.` `!` `?` — "CHAPTER IV." aside) or a clause (`,` `;`). It is also not a title if it directly continues an unfinished line: `blocks.py` keeps it in that paragraph. Colons still end titles ("RECITALS:"), and Markdown and clause headings are unchanged. The EPUB now files all twelve stories under the book, and the disclaimer stays one paragraph under "12. LIMITATION OF LIABILITY".

The same run confirmed the line anchors. All 310 chunks of the TXT copy are found verbatim in the lines they cite (CRLF file).

## Whisper no longer drops speech silently

The 2026-09-24 language benchmark found testimony missing from the index with no warning: faster-whisper's batched decoder, given a clip joining several utterances, can return only the last sentence. On FLEURS en `fleurs_01`, 13.2 % of the detected speech had no transcribed word. Two of three sentences at 13.2–38.0 s were gone.

`FasterWhisperAdapter.transcribe` now checks its own output against the VAD map. Any speech segment of at least 1 s with no word on it means something was dropped:
- A batched clip holding such speech is **decoded again sequentially**, and its batched output is replaced (not merged, so nothing is duplicated).
- Speech still wordless after that (a sequential decode can lose a clip's last sentence too, as at 584 s in the same file) is **decoded on its own**.
- Whatever remains is logged as a warning with its seconds.

`fleurs_01`: 13.2 % untranscribed → 0.0 %, with 6 of 31 clips re-decoded. Batched decoding stays the fast path.

**Benchmark, re-ingested from scratch (2026-09-27 vs 2026-09-24).** The language benchmark now reports untranscribed speech itself (`speech_coverage` in `asr.json`):

| | Untranscribed speech | FLEURS accuracy | Other |
|---|---|---|---|
| en | 4.2 % → 0.04 % | 82.2 % → **92.4 %** | lecture 94.4 % → 94.5 % |
| sl | 0.56 % → 0 % | 79.7 % → 78.7 % | |
| sr | 0.03 % → 0 % | 88.7 % → 88.6 % | ParlaSpeech 92.6 % → 92.6 % |

Slovenian's drop is recovered speech the reference leaves out. At 252 s of `sl/fleurs_01` the reader starts a sentence, says "Once again." and re-reads it. That false start used to vanish; now it is transcribed verbatim (+16 insertions, same substitutions and deletions). None of the three languages reaches the 95 % target yet.

## Database migrations (Alembic)

An appliance holds months of ingested material, so "reset the database" can't be how a schema change ships. Until now `OnyxEngine.start()` ran `create_all()` and then per-feature tuples of idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS` statements. That could add a column, but it couldn't rename, retype or drop one, and nothing recorded which build a database had reached.

- **`start()` now runs `upgrade_to_head()`** (`app/core/database/service/migrations.py`) in one transaction. Postgres DDL is transactional, so a failed revision leaves the database as it was, and `DBMigrationError` stops the boot. An unreachable database is still reported as unreachable.
- **Revision `0001` is the baseline**: the schema as it stood, with the full-text column stemmed in `FTS_LANGUAGE_CONFIG` (read at migration time) and `CREATE EXTENSION vector` first. Migrations are forward-only; the way back is a restore.
- **Databases from before migrations are kept, not reset.** Without `alembic_version` but with Onyx tables, a database is caught up (`create_all()` plus the old statements, frozen in `data/migrations/pre_alembic.py`) and stamped `0001`. Verified on a copy of the dev database: its 576 chunks and chat turns were kept, and the result was identical to a fresh baseline.
- **What the check turned up:** the old statements added `research_reports.mode`, `research_report_evidence.anchor_kind`, `vector_chunks.anchor_kind` and `vector_chunks.section_path` as `TEXT`, where the models declare `VARCHAR`. The catch-up aligns them (binary-coercible, no rewrite). Otherwise every older appliance's next autogenerated revision would carry spurious type changes.
- **The integration suite builds its schema through the migrations**, and `test_migrations_integration.py` fails when a model changes without a revision (autogenerate diff must be empty).

## Images: described by the vision model, checked by OCR

Clients hold standalone images: photographed whiteboards, scans saved as JPG, charts, X-rays, site photos. They have no speech and no document text, so they are understood only through vision. `OnyxEngine.ingest_images()` (CLI option 10, the desktop's "Images" mode) runs the new `BatchImagePipeline`: scan → ingest → `ImageDescriptionPhase` → unification → indexing. It uses one Qwen2-VL load for the whole batch, then the embedder. No transcript stage, contextual SLM or clip extraction runs.

**The excerpt is machine-written, and says so.** An image has no words of its own, so what a citation quotes is written by the vision model and OCR. The product rule is that nothing is stated without grounding, so:
- **Text the vision model reads is kept only if RapidOCR also reads it** (word for word, or 80% of its words including every number). A VLM writes plausible text that isn't there; OCR misreads characters but never invents a line. Items OCR didn't read are left out of the index and listed in the ingestion report. Unconfirmed quoted text and numbers inside the description become `[text not confirmed]` / `[number not confirmed]`.
- The excerpt's blocks are labelled "Image (machine description)" and "Text on the image (read by OCR)", in the pack's language. The answer, reasoner and report prompts describe an `(image)` excerpt as a machine-written description to be reported as what the image shows, never as the client's words. Recordings-only prompts are unchanged.
- **Serbian images are unverified.** There is no Serbian OCR, so the vision model's text is shown under a "not verified" label and counted in the report.
- Descriptions are in English on every appliance (Qwen2-VL is weak in Serbian and Slovenian); text read off the image stays in its own language.

**Reusing what exists.** The still-image mode is a second prompt on the existing `QwenVLMAdapter` (same instance, same `ModelType.VISION`), not a second loader. OCR is exposed as `DocumentConversionAPI.read_image`. `ContextualizerAPI.execute_image` stores a one-chunk transcript without a model, so unification and indexing are untouched. The new `AnchorKind.IMAGE` renders as "image" / "slika".

**What the checks turned up.**
- The video grounding (`ground_observation`) drops any frame with no legible text and no marks. That is right for a talking head and wrong for a photo, whose description is its only evidence, so still images get `ground_still_image`.
- Run on a generated notice, a chart and the X-ray fixture, the real model matched OCR on every text item, reported the chart's red arrow as a mark, and described the X-ray without a diagnosis.

**Desktop.** Cited images show a thumbnail on evidence cards and in the review panel, with "Open image ↗". Only the scaled size is decoded, and EXIF rotation is applied. This also fixes a bug from the books change: the gateway recognised only `page`/`section` anchors as documents, so TXT/Markdown citations showed no location and "Open recording".

**Pipelines don't pick up each other's files, or Onyx's own.** The scan is shared and each mode keeps its own suffixes; the four lists are disjoint. Two gaps turned up while checking this, both older than images but made worse by them:
- **The scanner walked into the vault.** Frames, clips and extracted audio live in `onyx_data/YYYY/…`, and reports in `onyx_data/<env>/reports/`. Pointing any mode at a folder containing them re-ingested Onyx's own files. After the books change, that included the model-written reports as source documents. The scanner now prunes both (`is_onyx_output`).
- **Audio formats were dropped silently.** The audio pipeline accepted `.ogg`, `.opus`, `.wma`, `.aiff`, `.amr`, `.alac` and `.mka`, but the scanner only passed five formats. There is now one list, which the pipeline and the video ingest filter import.

Documents scanned to JPG/TIFF stay images, by decision; scanned PDFs go only through the document pipeline.

**Known gaps.** HEIC isn't accepted (pillow-heif is a native dependency). Multi-page TIFFs use their first page. Whether the 80% word rule is too strict for real handwriting hasn't been measured.

## Books and texts: TXT, Markdown, EPUB, HTML and RTF, cited by line or chapter

Clients also hold books, manuscripts, notes and exported web pages. None of the audio stages apply: their sentence and chunk types need speakers and timestamps. The document path already did everything plain text needs (structure along headings, situate each chunk with the contextual SLM, index), so books and texts are new **document kinds**, not a new pipeline. The "Documents" mode (CLI option 9, the desktop's "Documents & books") takes a folder mixing PDFs, DOCX and all of these.

**Cited by line.** A plain-text or Markdown chunk cites the file's own lines, with its chapter: `[Doc 3] (lines 120–164 · CHAPTER IV.)`, a range anyone can check in an editor. This is a new `AnchorKind.LINE`, stored in the existing `page_start`/`page_end` columns (no DDL). Every consumer already rendered non-time anchors through `format_location`, so only the formatter changed. Labels come from the pack (`line_label`/`lines_label`: line/lines, vrstica/vrstice, red/redovi). EPUB, HTML, DOCX and RTF are converted before their lines exist, so they cite the chapter or section.

**Decisions and what the checks turned up.**
- **TXT and Markdown bypass MarkItDown.** Its plain-text path collapses runs of blank lines on every result, which would shift every cited line. They are decoded by charset-normalizer directly (cp1250/cp1251 files are common in Serbia and Slovenia) and cleaned line by line without dropping a line. The integration test reads the cited lines back out of the cp1250 file and finds the chunk's text there.
- **EPUB and HTML use MarkItDown's own converters.** They are core and local (zipfile, defusedxml, BeautifulSoup), so there is no new dependency. **RTF** uses striprtf, pure Python, which honours the file's code page.
- **Chapter headings are in the language packs.** "CHAPTER IV.", "Glava 3" / "Поглавље 4", "4. poglavje", "Prologue" and so on need the number to end the line or be followed by a separator, so a hard-wrapped "Chapter 3 of the Act provides that" stays prose. "CHAPTER IV." had been rejected as a sentence because of its final period; a period after a number or a Roman numeral no longer counts as a sentence end.

**Known gaps.** A headingless EPUB/HTML/RTF cites only the file. A 100k-word novel is about 400 chunks, one SLM call each.

## Recordings-only appliances see no change

Document support had changed wording that transcript answers pass through: the QA answer prompt, the reasoner's prompts, report scoping/outline/section prompts, the planner's corpus overview, the references header ("Timespan" → "Location") and the CLI's timestamp line. Each is now back to its original text, and the document wording is a separate variant chosen only when a document excerpt is actually in play — per context for answers and sections (the token budget is computed for the variant that is sent), per evidence bank for the outline, per corpus for scoping. The query-rewrite prompt, which runs before anything is retrieved, is simply the original. `git diff` of every prompt file and language pack shows only added lines.

Also: the startup backfill of `vector_chunks.position` sorted the whole table on every boot; it now checks an index that is empty once the backfill is done.

## Tables keep their rows

Document ingestion shipped with tables flattened, and on a real invoice it was worse than lost layout: MarkItDown read a borderless line-item table **column by column** ("Parking place", "Cleaning service", "2", "4", "EUR 40.00", ...), so no amount stayed with its item. The contextual SLM, the index and the answer model all saw disconnected values; "how much was parking?" had no row to match.

Pages with tables are now re-read by `PdfplumberLayoutExtractor` (pdfplumber, which MarkItDown's own PDF converter uses) and emitted as Markdown tables; pages without tables keep MarkItDown's text. Ruled tables come from pdfplumber's line finder; borderless ones from **shared column gutters** — vertical strips several consecutive lines all leave empty, clearly wider than their word gaps. Scans use the same code on RapidOCR word boxes. On the dev appliance, "how many parking places were billed, at what unit price, and what is the total due?" is now answered correctly from the invoice's rows, cited to its page.

Along the way: the integration fixtures generated PDFs at ~3 pt (Qt at 300 dpi writes 72-dpi units), which hid two bugs — pdfplumber's fixed 3-point word tolerance merging neighbouring cells (now size-relative) — and made scans unreadable. Fixtures are now real size. A gap-width rule for cells was tried first and dropped: OCR word spacing jitters across any single threshold.

## Document ingestion: PDF and DOCX, cited by page

Onyx indexed only recordings. Law-firm clients mostly hold documents — contracts, filings, invoices, letters — as a mix of born-digital and scanned PDFs. `OnyxEngine.ingest_documents()` (CLI option 9, the desktop's "Documents (PDF / DOCX)" mode) runs the new `BatchDocumentPipeline`: scan → ingest → `document_conversion` → `document_structuring` + contextual SLM → unification → indexing.

**Cited by page, not by time.** Chunks now carry an anchor (`app/core/anchors`): a time range, a page range or a section path. `vector_chunks` gained `anchor_kind`, `page_start/end`, `section_path` and `position`; neighbours and reading order use `position`, since a document chunk's `start_sec` is 0.0. An existing database is upgraded at start-up (`VECTOR_CHUNKS_UPGRADE_DDL`, positions backfilled in time order), with no re-ingest. Transcript behaviour is unchanged: the whole unit suite passed before any new test was added, and transcript headers still cost the same context budget.

**Choices, and what the checks turned up.**
- **Converter.** Microsoft MarkItDown (MIT), after comparing Docling and Marker (Marker's weights licence restricts commercial use and competing products). MarkItDown drops empty pages and joins the rest, so PDFs are split with pypdfium2 and converted page by page. It reaches the network only via URL strings and its audio converter; the adapter uses bytes only and installs `[pdf,docx]` only.
- **OCR.** RapidOCR PP-OCRv5 on the CPU, for English and Slovenian (`latin` model). A spike on synthetic scans found no model reading Serbian: the Cyrillic one lacks č ć đ š ž, the Latin one lacks Cyrillic, confidence could not choose between them (the Cyrillic model dropped "ž" from "može" at 0.95), and RapidOCR **silently dropped** two of six lines of a mixed-script letter. The adapter counts detected lines against read ones and reports incomplete pages; Serbian scanned pages are reported as unreadable.
- **Three conversion bugs caught on real files before any test existed:** ligatures (`Deﬁnitions`) that no search matches, now expanded; a sparse born-digital table page sent to OCR and returned worse ("Monthty rent", "026"), now OCR'd only when an image covers the page; and page-number removal that deleted the years "2025"/"2026" from a rent table, now limited to edge lines near the real page number.

**Known gaps.** Only PDF and DOCX. (Tables lost their rows — fixed in "Tables keep their rows" above.) No document classification, summary cards or typed fact extraction yet. OCR accuracy on real client scans is unmeasured.

## Language packs: English, Serbian and Slovenian appliances

Onyx assumed English end to end. That covered the Whisper language, the domain scout (`whisper-tiny-en`), the Postgres stemmer, phonetic correction's letter names and spelling rules, the hallucination blacklist, stopwords, the visual grounding regexes, the refusal sentence, citation parsing and the report's boilerplate. Every one of those is now read from a validated **language pack**, `data/languages/<code>.toml` (new core module `app/core/language`, see `docs/core/language.md`). An appliance serves the one language named by `APPLIANCE_LANGUAGE`.

**Why not a translation layer.** Translating transcripts to English would put model-written text under every citation, multiply ASR errors by MT errors, and hide mis-hearings from phonetic correction (a translated mis-hearing no longer sounds like the term). So nothing is translated: the index and the answers stay in the spoken language.

**English doesn't move.** `en.toml` holds exactly the values that used to be hard-coded, and English prompts are byte-for-byte unchanged (`localize_prompt` is a no-op for English). The existing unit suite passed unchanged apart from two fixtures that now build the English config explicitly: the hallucination blacklist and the transcription adapter mock's `detect_language`.

**What Serbian and Slovenian get.**
- **Small models.** Gemma 4 E4B replaces Llama 3.2 3B and Qwen2.5-3B, neither of which covers these languages, in five stages: domain classifier, contextual retrieval, visual synthesis, phonetic verification, visual hallucination correction. Verified on the dev box: it loads through each adapter, including the GBNF-constrained ones and contextual retrieval at 32K context (~6 GB), and handles Serbian input correctly.
- **Domain scout.** Uses large-v3.
- **Punctuation.** Whisper's own. The fullstop classifier was trained only on en/de/fr/it.
- **Serbian script.** Serbian transcripts and queries are normalised Cyrillic → Latin, the exact direction, word by word so timestamps align. Postgres' `serbian` stemmer maps both scripts and both variants to one stem; Slovenian uses `simple`, because Postgres ships no Slovenian stemmer.
- **Output language.** Answers, refusals, reports, and the generated context stored beside the transcript are written in the appliance's language. The VLM stays English.

**Two new safety checks.**
- **`LanguageMismatchError` (transcription).** Whisper's language ID runs over the first 30 s of speech. A file it confidently hears in another language is rejected, not forced: forced, Whisper translates or invents fluent text, which would be indexed as if someone had said it. Serbian audio detected as Croatian or Bosnian is accepted.
- **`IndexLanguageMismatchError` (engine start).** An index whose keyword column was generated for another language refuses to start, instead of silently searching with the wrong stemmer.

**Glossaries are per language.** They live in `data/glossaries/<code>/`; the English ones moved to `en/`. The Wikidata miner collects the pack's label languages (Serbian: `sr`, `sr-el`, transliterated to Latin).

**Known gaps.**
- Serbian and Slovenian have no glossaries yet, so phonetic correction skips itself.
- Their phonetic matching has no inflection handling: a term replaces a span in its dictionary form. This needs work before those glossaries are trusted.
- Their hallucination patterns are unverified seeds.
- The desktop UI is English.
- Whisper's word error rate on real Serbian/Slovenian client audio has not been measured. Measure it before selling either language.

## A gathering round is only entered when it has something new to ask

Exhaustive mode kept no record of what it had already searched for, so it re-searched it.

The follow-up query for an uncovered section comes from the model when it writes one and from a
deterministic fallback when it does not — and that fallback is a pure function of the section's
title and brief, neither of which changes between rounds. A section the model wrote nothing for was
therefore re-searched with the **identical query every round** until the budget ran out. The
"a round added no new chunks" rule caught it, but only after the round had been paid for: one
embedder load, one reranker load and one ~5.7 GB report-model load to re-reach a conclusion already
reached. Introduced with exhaustive mode itself (`766c19d`).

Found by running deep and exhaustive head to head on a section the corpus cannot support. The run
took 3 rounds; rounds 2 and 3 issued the same query, and round 3 was guaranteed to return nothing
before it started.

**What the run now remembers.** A `QueryAttempt` per search — its text, its round, and how many
chunks it added *that were not already in the bank*. Held in memory for the run, like the follow-up
queries themselves; persisting them would put transient queries into
`research_report_sub_questions`, where `REPORT_MAX_SUB_QUESTIONS` rejects the plan on the next
validation.

It is used twice:

- **The model is shown it** (`SEARCHES ALREADY RUN` in the outline prompt, with what each query
  returned). A query credited with nothing is the load-bearing entry — it says those words are
  absent from this archive, so a synonym will fail too. The model cannot re-word a search it cannot
  see, which is why its follow-ups were paraphrases.
- **Follow-ups are filtered against it** afterwards, in `ReportAuthoringAPI`, because neither the
  model nor the fallback can be trusted to remember. Returning no follow-ups is how the API says
  *stop*; the pipeline then writes instead of gathering. Comparison normalises case, whitespace and
  trailing punctuation only — over-matching would drop a round that had something to find.

**Deep mode is untouched.** The history is passed only to a pass that can act on it, so a
single-round run — every deep run — sends the outline prompt it always sent, and the final writing
round of an exhaustive run sends it too.

Re-running the same comparison after the fix: deep 1 round / 87 s, exhaustive **2** rounds / 116 s
(was 3). The follow-up was model-written rather than the fallback for the first time, because the
failed query was now visible to it as having found nothing.

## Report view fixes: a dead gateway import, reports saved into the repo, an invisible approval step

Four defects in the Report view, all introduced with it (`766c19d`), three of them user-visible.

**Every report run failed with "The appliance could not complete that request."** `_import_engine()`
is the deferred cache that keeps `ui/` and `service/` from importing `app.*`, and three symbols the
gateway uses — `OutlineSection`, `SubQuestion` and `markdown_renderer` — were never registered in it.
`_apply_edits` and `_map_report` raised `KeyError`, which the class-name-matching error mapper turned
into the generic failure sentence, so the message named nothing. **Any symbol used anywhere in
`engine_gateway.py` must be registered in `_import_engine()`**; a missing one fails only at call
time. 13 tests now cover the mapping, and all 13 fail if the registration is removed.

**Saved reports landed in the git working tree.** The save dialog was given a bare filename, which
resolves against the process CWD — and `apply_runtime_environment` chdirs to the project root. Every
saved report, which is client transcript content, defaulted into the repository. `default_save_path`
returns an absolute path under `~/Documents`.

**The approval step read as though it were skipped.** Scoping takes about a minute, during which the
previous outline stayed on screen and the new one swapped in underneath it. For the same topic the
two look nearly identical, so nothing appeared to happen and the run looked self-starting.
`ReportView.start_planning` now clears the outline and shows a scoping notice — and `MainWindow`
calls it only once the bridge has *accepted* the request, mirroring the ingest path, so a refusal
(appliance busy) does not throw away an outline the user is still editing.

**Sub-question ordering double-counted.** `order_index=len(questions) + question_index` advanced
twice per question within a section; it is `len(questions)`.

## Exhaustive report mode, and the desktop Report view

Two gaps in Deep Research Report mode, closed.

### 1. "Not covered" used to mean "one sweep missed it"

Gathering ran exactly once: decompose the topic, search, write. But **you don't know what you're
missing until you've read what you found** — the outline pass assigns evidence and may discover a
section has nothing, by which point searching is over. That section was reported as "Not covered by
the available material", which conflated *the corpus doesn't have this* with *the sub-question was
worded badly*. The dev report cites "Draghi", "the scene of the crime", "wedge", "final flush of
volume"; a query written before any of that vocabulary was visible could easily miss material a
re-worded one finds.

**Exhaustive mode** (`OnyxEngine.execute_iterative_research_report`) is a new mode beside the
existing one, selected like Reasoned Answer is selected beside Chat. It re-gathers while sections
come back unsupported, up to `REPORT_MAX_GATHER_ROUNDS` (3).

**Code decides *when*, the model decides *what*.** The trigger is objective — a section got no
evidence, or a round added no new chunks. The model is only asked the thing it is actually good at
and that code cannot do: re-word a failed query using vocabulary now visible in the evidence bank.
It already has the bank index in context during the outline pass, so the follow-ups cost nothing
extra. Asking a model "is this report complete?" was considered and rejected: LLMs asked to
self-assess completeness either always stop or never stop, and it would cost a 9B call per round to
get an unreliable answer.

Measured on the dev corpus with a deliberately unsupportable section added to the outline: round 1
gathered 25 chunks and left it unsupported, round 2 re-searched and found **0 new chunks**, the loop
exited early, and the section was reported as not covered. 130 s against 70 s for deep mode.
"Not covered" now means *searched twice and genuinely absent*.

### 2. The model-swap budget survives the loop

The report model is acquired **once per round**, and the final round's residency also writes — so a
one-round exhaustive run costs exactly what a deep run costs, and each extra round costs one
embedder load, one reranker load and one report-model load.

`ReportAuthoringPhase.execute_report` gained `stop_on_gaps: bool`. Deliberately a flag, not a
callback: a callback invoked while the phase holds the model, whose entire purpose is to decide "go
gather more", would hand the next reader the exact footgun this design exists to avoid —
`acquire_model(wait_for_clear=True)` parks with no time limit, so re-gathering from inside the
residency wedges the process.

**The executive summary now rides the writing residency** instead of reloading the 5.7 GB model for
one short call. That removes a wasted heavy load from *both* modes; the integration test's assertion
tightened from `<= 2` to `== 1`.

### 3. A latent hang this loop would have triggered

`preload_model()` assigned `_preload_future` with no guard. Straight-line pipelines resolve every
preload exactly once, so this never fired — but **in a loop any early return orphans the future**:
the next round overwrites it, the background worker still loads that model, `discard_preload` can
never be called on it, nothing releases it, and the next `wait_for_clear` acquisition parks forever
with no exception and no exit but a restart. `discard_preload` cannot rescue it either —
`future.cancel()` fails once the task is running, and its done-callback only fires after the load
completes, which needs the VRAM it is waiting on.

Fixed three ways: `preload_model()` is self-guarding in the four report-path phases, the loop body
discards per iteration rather than only in the run's `finally`, and no preload is fired on a path
that might not resolve it. An integration test runs the loop on a daemon thread and fails if it does
not finish, so a regression fails loudly instead of hanging.

Also fixed while the loop exposed them: evidence is deduped **across** rounds (a round-1 chunk
re-found in round 2 would violate `UniqueConstraint(report_id, chunk_id)` and kill the run); the
sweep's coverage count is carried forward with `max` (later rounds run no sweep, and passing their 0
through made the renderer drop the coverage line entirely); and `OutlineValidationError` no longer
marks a report FAILED — a user's outline being wrong is not a run crashing, and stranding the report
over a fixable typo helped nobody.

### 4. The desktop Report view

`frontend/onyx_desktop/ui/views/report_view.py` — three states in a stack: compose (topic + mode),
run (determinate progress + working Stop), read (the report). The right pane is the same
`SourcePanel` chat and search use.

- **Real click-to-verify, for the first time.** Worth being precise: the chat view's `[Doc N]` chips
  only *look* clickable — `MessageBubble._body()` renders them as a coloured `<span>`, and you verify
  a chat claim by clicking the evidence card underneath. A report is pages long and has no cards, so
  the citation itself had to become the control. `QTextBrowser` + `setOpenLinks(False)` +
  `anchorClicked` pairs with the `onyx://doc/N` scheme the renderer already emitted in link mode.
- **A determinate progress bar**, because unlike ingestion a report states its step count up front.
- **Cancel bypasses the bridge**, the one deliberate exception to "everything through the worker":
  the worker is busy running the report being stopped, so a queued cancel would arrive exactly when
  it is useless. Safe because the engine only sets a `threading.Event`.
- **Save as Markdown** copies the file the engine already wrote, so what the user saves is
  byte-identical to what the appliance produced and audited. The in-memory fallback strips
  `onyx://doc/N`, which means nothing outside the app.
- Progress ticks are now **routed** rather than broadcast — `MainWindow._on_progress` sent every tick
  to the Ingest view, which would have driven the ingestion counters from report jobs.

### 5. A shipped bug found on the way

`QComboBox` stores userData as a QVariant, and a `str`-derived `Enum` comes back out as a **plain
str**. Every gateway branch compares with `is`, so in the desktop app:

- **Reasoned Answer ran single-pass chat**
- **Verified search ran standard search**
- **Audio ingest ran the video pipeline**

Three of the four mode selectors silently did the wrong thing. Fixed at the boundary — each view now
rebuilds the enum from `currentData()` — so downstream `is` comparisons stay honest.

## Deep Research Report mode: topic in, multi-page cited report out

> **Action required:** none. New tables are created by `Base.metadata.create_all()` on boot, and
> `OnyxEngine.start()` backfills the renamed chat mode's stored values. Reports are written to
> `onyx_data/<env>/reports/`.

Both existing chat modes end in **one answer**. What a law firm or a hospital actually wants is the
other shape of work: give the appliance a topic, have it sweep the whole local corpus, and come back
with a structured, fully-cited report. That is what this adds, as a third mode alongside the other two.

**The existing "Deep Research" chat mode is renamed to Reasoned Answer** (`OnyxEngine.chat_reasoned`,
`LiveReasoningPipeline`, `SessionTurnMode.REASONING`, CLI option 7). Its behaviour is unchanged — it
was always an iterative reasoning loop producing a chat answer, not deep research, and the name now
belongs to the thing that earns it.

### The architecture, and why it is this one

Modelled on **WebWeaver** ([arXiv 2509.13312](https://arxiv.org/abs/2509.13312)): a planner interleaves
evidence gathering with outline revision, writing citations into the outline that point into an evidence
**memory bank**; a writer then composes the report **section by section**, pulling only that section's
evidence into context and discarding it afterwards. Every survey of the field
([2508.12752](https://arxiv.org/abs/2508.12752), [2506.12594](https://arxiv.org/pdf/2506.12594)) reports
the same finding: outline-then-section synthesis beats brute-force long-context generation.

It is also the only shape that fits the hardware. **No single context ever holds the corpus**, and peak
context is max-over-sections rather than their sum. A 128K window makes this *possible*; it is pruning
that makes it *affordable*.

### The model-swap schedule is the design

`app/orchestration/service/pipelines/deep_research_report_pipeline.py` is written as a swap timeline,
because on a single 12 GB card that is what determines the wall clock:

| Stage | Residency | Work |
|---|---|---|
| Scope | `RAG_EMBEDDING`, then `REPORT_AUTHORING` | one grounding search, then propose an outline |
| Gather | `RAG_EMBEDDING` ×1 | **every** sub-question *and* the per-document sweep in one batched search |
| Rank | `RAG_RERANKING` ×1 | every query's pool reranked under one residency |
| Write | `REPORT_AUTHORING` ×1, **held** | outline optimisation + every section + the summary |

Two mechanisms do the work. `RetrievalAPI.search_batch` acquires the embedder **once** for the whole
fan-out — a 40-query sweep costs one 2.2 GB load instead of forty. And `ReportAuthoringPhase` is a
single `execute_report()` rather than `execute_outline()` + `execute_sections()`, because each `execute`
resolves a preload and unloads in its own `finally`: splitting it would mean two loads of the 5.7 GB
GGUF instead of one. Preloads are fired one stage **early**, so each load overlaps the previous stage's
compute rather than stalling after it.

Measured end to end on the dev corpus: **a 4-section report in 78 s**, two heavy model loads for the
whole run. A naive per-section arrangement of identical work costs ~40 loads, 16 of them heavy.

`integration_tests/orchestration/test_deep_research_report_pipeline_integration.py` wraps the loader in
a counting proxy and **asserts the load count**, because without that a later "simplification" would
silently double the wall clock and nothing would fail.

### Grounding is enforced in code, not asked for in prompts

- The writer sees **only `raw_text`**, via the same `build_numbered_context_block` the chat path uses.
  `text_payload` carries generated situating context; an answer grounded in it would still emerge
  carrying a valid-looking citation.
- `[Doc N]` numbers are assigned **once, report-wide**, in reading order — so a number means the same
  thing in section 2, in the summary, in the references table, and in a run resumed three days later.
- Each section's citations are normalised against **that section's** allowed numbers. Without that
  restriction a section could cite a document it was never shown, because numbering is report-wide.
- A section that cites nothing is re-prompted once, then recorded `SKIPPED_NO_EVIDENCE` and listed under
  "Not covered by the available material" — **never uncited prose presented as a finding**.
- The executive summary reads section prose, never evidence, and may cite only numbers the body used.
  It is the one place a report synthesises across sections, and so the easiest place to invent a link.
- The scoping pass sees real source names and real excerpts before proposing anything. In the first live
  run this caught a human misreading: "ECB" in a filename was assumed to be the cricket board, and the
  planner correctly identified European Central Bank trading content from the transcripts.

### Coverage is stated, not implied

`> Coverage: swept 50 of 214 indexed sources.` appears in every report. A report that looked at a
quarter of the corpus and reads as though it covered all of it is dangerous in a product whose entire
pitch is traceability. The scrubber setting is printed for the same reason.

The per-document sweep exists so a relevant recording is not missed merely because no generated
sub-question surfaced it. Its cost is gated by one index-served ANN probe
(`IRetrievalRepository.pick_sweep_sources`) rather than a `GROUP BY` over `vector_chunks`, which no
index can serve and which would degrade into a full scan as the corpus grows.

### The retrieval scrubber is out of the gathering loop

`REPORT_SCRUB_ENABLED = False` by default. It is one sequential Qwen2.5-7B call per chunk and, per the
existing eval, drops the answering chunk for ~1 question in 9. For a report, recall dominates precision:
an irrelevant chunk is simply never cited, but a dropped one *cannot* be. The cross-encoder's score
floor does the cheap filtering instead. The setting is printed in the report so the choice is visible.

### Persistence, resume and cancellation

Every stage boundary commits, which is what makes a run recoverable rather than all-or-nothing.
`resume_research_report` writes only the sections still pending and **loads no retrieval model at all** —
the persisted bank is the run's memory. A crash in section six of eight costs two sections and one load.

`research_report_evidence.chunk_id` is deliberately **not** a foreign key. Chunks cascade from `sources`,
so re-ingesting a file deletes them, and a signed-off report has to survive that: denormalised source
name, timespan, speakers and excerpt keep a citation renderable after its chunk is gone. An integration
test deletes the chunk and asserts the citation still resolves.

Cancellation is new to the codebase — `JobManager.cancel_job` existed but nothing polled it. A
`CancellationToken` is checked at each job boundary, so a stop lands within one section (~90 s) rather
than mid-generation, which llama.cpp does not offer. A cancelled report keeps and renders what it wrote.

### What this touched

- **New feature** `app/features/deep_research_report/` — entities, one adapter under one new
  `ModelType.REPORT_AUTHORING`, Postgres repository, atomic Markdown file store, `ReportAuthoringAPI`.
- **`app/features/rag_retrieval/`** (additive) — `search_batch`, `list_indexed_sources`,
  `pick_sweep_sources`, batched `embed_prompts`.
- **`app/core/jobs/`** — `CancellationToken`, `JobCancelled`, `PROGRESS_CANCELLED`, and
  `JobRequest.progress_detail` so a listener can say "Writing section 4/8: Regulatory exposure" instead
  of "research_section started".
- **Orchestration** — `ResearchScopingPhase`, `ReportAuthoringPhase`, `RagRetrievalPhase.execute_sweep`,
  `DeepResearchReportPipeline`.
- **Engine** — `EngineState.RESEARCHING` (a report holds the GPU exclusively, like ingestion),
  `propose_research_plan` / `execute_research_report` / `resume_research_report` /
  `cancel_research_report` / `get_report` / `list_reports`, and CLI option 8 with an approval gate.

**Known limits.** No desktop UI yet (CLI only) and no PDF/DOCX export — both are the next phase.
Gathering runs exactly **one** round: `REPORT_MAX_GATHER_ROUNDS` and the pipeline's `max_gather_rounds`
argument are plumbed through but the re-gather loop itself is **not implemented yet**, so a section the
first sweep missed is reported as uncovered rather than searched for again. A report locks out chat and
search for its duration, which is not solvable on one GPU.

## Desktop frontend: a native PySide6 app with click-to-verify citations

> **Action required:** install the new GUI dependency with `venv/bin/pip install -r requirements.txt`
> (`PySide6-Essentials`, ~230 MB on disk). Deliberately **not** `PySide6` or `PySide6-Addons`: the app
> has no embedded media player, so QtMultimedia is not needed.

Until now the appliance was operable only through `app/main.py`'s numbered menu. That cannot deliver
the product's central claim — it prints a file path and a timestamp and leaves the user to go find
the video themselves. `frontend/` closes the verification loop: read a claim, click its citation,
watch the speaker say it, without leaving the app.

- **`frontend/main.py`** launches a PySide6 (Qt6) desktop app with three views — Chat (single-pass
  and Deep Research), Search (standard and verified), Ingest (video and audio, with a live log pane).
  `--stub` runs the whole UI against canned data with no GPU, database or weights, which is the loop
  to develop in: a real chat turn costs tens of seconds.
- **In-process, no server.** The app calls `OnyxEngine` directly on one background `QThread`. No
  HTTP, no IPC, no second runtime to vendor onto an air-gapped appliance. Only
  `frontend/onyx_desktop/data/engine_gateway.py` imports `app.*`; everything above it speaks in
  frozen dataclasses, which is what makes the stub and the headless tests possible.
- **One engine call at a time, by construction.** Requests cross to the worker as queued Qt signals,
  so the event loop serializes them — the VRAM pool holds one model at a time, so two overlapping
  GPU calls would collide rather than queue. A second request while one is in flight is *refused*,
  not enqueued.
- **Grounding is enforced in the UI, not just hoped for.** An answer that returns with no citations
  is rendered with an explicit "not traceable" warning rather than as an ordinary reply; an
  unresolved clip path disables its open button with the reason; an unresolved Deep Research
  self-critique shows an amber strip. Evidence cards show the verbatim `raw_text`, never
  `text_payload`.
- **`OnyxEngine.get_chunks(chunk_ids)` and `list_sources()`** (new, declared on `IOnyxEngine`). A
  `QACitation` carries ids, timestamps and a path but **no text**, so showing a cited excerpt beside
  its clip needs the chunks read back. `get_chunks` loads no model and resolves physical paths onto
  its results, which `RetrievalAPI.get_chunks` does not.
- **`JobManager(repository, progress_listener=None)`** and
  `OnyxBootstrapper.build(progress_listener=…)`. Every Phase already wraps its per-file work in
  `run_job`, so one hook observes every file of every phase in every pipeline — the only way to show
  batch progress, since the pipelines report nothing until they return. Defaults to `None`, so the
  CLI is untouched. A listener that raises is logged and ignored.
- **No embedded media player, deliberately.** The first cut had one, and on this hardware Qt's
  FFmpeg backend tried VAAPI hardware decode, failed (`vaExportSurfaceHandle failed`, no
  `nvidia-vaapi-driver` installed) and segfaulted the process mid-answer. Onyx is not a media player:
  an evidence card's **Open clip** button hands the file straight to the desktop's own player via
  `QDesktopServices.openUrl`, in one click. That
  removed the entire codec/hardware-decode failure class and dropped `PySide6-Addons`, taking the Qt
  footprint from ~650 MB to ~230 MB — which matters on an appliance image. The cost is that a system
  player cannot be told to start at a timestamp, so the source panel shows the cited span in large
  copyable text; for extracted clips (all video ingestion) the clip already *is* the excerpt.
  Clicking a card's body instead loads the excerpt into the review panel without launching anything
  — the first cut had only a "Review source" button that merely filled that panel, which left the
  player two clicks away and read as if it were broken.
- **Two hangs found and fixed while building this**, both reproducible: `worker.finished →
  thread.quit` must be a `DirectConnection` (a `QThread` object's affinity is the GUI thread, so an
  auto connection queues `quit()` behind an event loop already blocked in `thread.wait()`), and
  `closeEvent` must defer its shutdown to a `QTimer.singleShot(0, …)` rather than calling
  `self.close()` reentrantly from inside itself.
- Citation tags are re-normalized in the gateway before display: `generated_answer` is the raw model
  text while `citations` was parsed from a normalized copy, so the answer could still contain
  `[Doc 1, 2]` after the citation list had split it. Reuses
  `normalize_citation_tags` rather than adding a second regex.
- `pytest.ini` gains `frontend/tests` and a `gui` marker. `frontend/tests/` deliberately has **no
  `__init__.py`**: a package named `tests` there shadows the root `tests` package and breaks
  collection of the entire existing suite.

**Known limits, all documented in `frontend/README.md`:** ingestion cannot be cancelled (nothing in
the pipeline has a stop point), ingestion progress goes quiet during transcription and diarization
(11 phases have no logger), conversation history is not re-displayed on resume, and there is no
library view yet.

## Qwen3.5-9B with a 128K context, and multi-turn chat

> **Action required:**
> - Download the new answer/reasoner model with `./models_download.sh` (`unsloth/Qwen3.5-9B-GGUF`, `Qwen3.5-9B-Q4_K_M.gguf`, ~5.7 GB).
> - `session_turns` gains columns. `OnyxEngine.start()` adds them to an existing table (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`), so no reset is needed.

- **One long-context model answers and reasons** (`app/core/config.py`).
  - **What changed.** `QA_GENERATION_MODEL_PATH` and `QA_REASONER_MODEL_PATH` both point at Qwen3.5-9B Q4_K_M; before, Gemma 4 E4B answered at 16K and Qwen2.5-7B reasoned at 8K.
  - **Why it fits.** Qwen3.5 is a hybrid Gated DeltaNet / attention model: only 8 of its 32 layers keep a KV cache, so a long context costs little VRAM.
  - **New settings.** `QA_GENERATION_N_CTX=131072` and `QA_REASONER_N_CTX=65536`. The new `QA_LLM_KV_CACHE_TYPE=q8_0` and `QA_LLM_ENABLE_THINKING=False` apply to both.
  - **Measured on the RTX 3060.**
    - The 128K context peaks at 8.6 GB of VRAM.
    - A 96K-token prompt takes 85 s (~1,130 tokens/s), and the model found a fact planted in it.
  - **Known cost.** Both roles use the same file under different `ModelType`s, so switching between reasoner and answer model reloads it.
- **Thinking mode is controlled for real** (`app/core/model_manager/utils/llama_cpp_chat.py`, new).
  - **The problem.** `Llama.create_chat_completion` has a fixed signature and drops template variables, so `enable_thinking` never reached the GGUF chat template.
  - **The fix.**
    - `chat_completion()` calls the model's chat handler directly, which does forward the variable.
    - `strip_reasoning()` removes `<think>` blocks, including reasoning whose opening tag the template put in the prompt.
    - `build_llama()` sets the KV cache type.
  - **Rename.** `GemmaGenerationAdapter` is now `LlamaCppGenerationAdapter`, and `GemmaOOMError` is now `GenerationOOMError`.
- **Answers without citations are no longer returned** (`generation_api.py`).
  - **The problem.** Qwen3.5 writes longer answers than Gemma. On the first eval it answered 3 questions with no citation at all, some with interpretation the transcript doesn't contain.
  - **The fix.**
    - The answer prompt now asks for no interpretation, advice or summary, and a `[Doc N]` tag right after each claim.
    - An answer that is neither the refusal nor cites a shown document is sent back once, in the same model acquisition, to be rewritten with citations (`CITATION_RETRY_PROMPT`). If it still cites nothing, the refusal is returned instead.
    - This applies to `generate`, `generate_with_critique` and `generate_conversation`.
- **Chat is multi-turn** (`OnyxEngine.chat(query, session_id=None)`).
  - **How a follow-up is answered.**
    - It is rewritten into a standalone search query by the reasoner model, before any preload.
    - The answer model then sees the earlier turns: each turn's question, its answer, and the excerpts it showed.
    - It is asked the user's words followed by the standalone form. Without that form, Qwen3.5 refused pronoun follow-ups ("and when did he walk in?") even with the answering excerpt in the conversation.
  - **Stable citations.** `[Doc N]` numbers are conversation-wide.
    - `session_turns` stores `evidence_chunk_ids` and `first_doc_number` per turn, plus `mode` and `standalone_query`.
    - A follow-up can cite an earlier turn's excerpt under its original tag, rebuilt with the new `RetrievalAPI.get_chunks`.
  - **Packing** (`qa_generation/utils/conversation_packer.py`). The current question and its evidence come first, then earlier turns' text newest first, then their evidence newest first. When the context is full, the oldest turns lose their excerpts first, then drop out. Tags in earlier answers that point at a dropped excerpt are removed.
  - **Prompt shape.** Messages are append-only, so a later KV-state cache could reuse the prefix.
  - **History window.** `CHAT_HISTORY_MAX_TURNS=20` bounds the history read per turn.
- **Deep Research shares the conversation.**
  - It reads every earlier turn instead of only the last.
  - Its evidence pool starts with this question's search, then the excerpts earlier turns showed.
  - It records its own turn with the pool, numbering only chunks new to the conversation. Its answers are stored without tags, since it numbers its evidence on its own.
  - Its answer model gets the standalone form of a follow-up too (`app/orchestration/utils/follow_up.py`).
- **CLI.** Option 5 is a conversation loop: a blank line returns to the menu and `/new` starts over. Options 5 and 7 share one session.
- **RAG eval.** Chat mode also scores the two follow-up questions, through a real session.
  - **Gemma 4 E4B, before:** 26/28 answered, 25/28 cite a relevant chunk (89%), 0 uncited, 5/6 unanswerable refused, 817 s.
  - **Qwen3.5-9B, after:** 28/30 answered, 27/30 cite a relevant chunk (90%; both follow-ups included), 0 uncited, 5/6 refused, 903 s.
  - **Not yet re-measured:** Deep Research mode.
- **Still open.** Every chat turn reloads the answer model and prefills the whole conversation, because the embedder, reranker and scrubber are swapped in between. At 50K+ tokens that is tens of seconds per turn.

## Retrieval and answering: evidence kept, context budgeted, citations complete

Measured with the new `scripts/eval/rag_eval.py`. It runs 28 answerable, 6 unanswerable and 2 follow-up questions (`rag_labels.json`) against the dev DB. The labels name chunk ids, so they only fit that DB.

- **A RAG eval exists** (`scripts/eval/rag_eval.py`, `rag_labels.json`).
  - `--stage retrieval` reports hit@k/MRR after RRF and after rerank, and how often a relevant chunk survives the scrubber. `--cache` reruns only the scrubber.
  - `--stage answer --mode chat|deep` runs the real engine and reports:
    - how many answerable questions got an answer, and whether it cites a relevant chunk;
    - uncited answers;
    - whether unanswerable questions were refused.
- **The relevance scrubber no longer throws away most answers** (`app/core/config.py`).
  - **The problem.** Retrieval and rerank were strong: the right chunk was ranked first for 87% of questions and was always in the top 10. But the Qwen2.5-3B scrubber then dropped the chunk that answers the question for about 1 question in 4, e.g. calling a chunk that contains the "7-2 vote" and its price targets "not related". Chat then had nothing to answer from.
  - **The fix.**
    - `RETRIEVAL_SCRUBBER_MODEL_PATH` now points at the Qwen2.5-7B weights already on disk: a relevant chunk survives 25/28 times, against 21/28 for the 3B.
    - The new `RAG_RERANK_MIN_SCORE=0.1` drops about 45% of non-answering candidates before the slow judge; every answering chunk scored at least 0.12.
    - Overall, a relevant chunk now survives 28/32 times (was 22/30), and the retrieval eval runs in 495 s (was 862 s).
  - **Tried and reverted.** Having the scrubber judge `raw_text` instead of the contextualized payload made the 3B stricter (57% survival).
- **Deep Research can no longer overflow the context window** (`qa_generation/utils/context_packer.py` (new), `generation_api.py`, `gemma_adapter.py`, `qa_reasoning/utils/evidence_formatter.py`, `reasoning_api.py`, `qwen_reasoner_adapter.py`).
  - **Introduced in:** `8052035`.
  - **Fixed in:** `75f333c`.
  - **The problem.** The evidence pool grew without a limit (3 passes × 10 chunks, about 12k tokens), while Gemma and the reasoner run at `n_ctx=8192` and nothing counted tokens. llama.cpp raised `Requested tokens exceed context window`, and the call failed.
  - **The fix.**
    - Both models count tokens with their own tokenizer. Evidence is kept best-first while it fits what's left after the prompts, the question, and the answer, critique and reasoning budgets.
    - The new settings are `QA_GENERATION_N_CTX`, `QA_GENERATION_MAX_ANSWER_TOKENS` and `QA_REASONER_N_CTX`.
    - Gemma now runs at a 16K context (about 4.6 GB VRAM instead of 4.1 GB at 8K), so a full Deep Research pool fits. The reasoner stays at 8K.
- **The answer model reads evidence as a timeline.** `[Doc N]` blocks are grouped by recording and ordered by time. The sentence the chunker repeats between neighbouring chunks is shown once.
- **Grouped citations are no longer lost** (`citation_parser.py`).
  - `[Doc 1, 3]`, `[Docs 1-2]` and `[Doc 2 and 3]` are rewritten to `[Doc 1][Doc 3]` in the answer text, and all of them get clip links.
  - Tags pointing at documents that don't exist are removed.
- **Only evidence reaches the answer model.**
  - `generate_with_critique` no longer appends the reasoner's `interim_analysis` to the question. That model-written text could be repeated with a `[Doc N]` tag, the same leak `build_context_block` already prevents for generated context.
  - Instead, Deep Research reranks the whole pool once against the user's question and moves the chunks the reasoner linked its claims to the front.
- **The reasoner cites `[E1]…[En]`, not UUIDs.** Links to tags that aren't in the pool are dropped instead of stored as well-formed but wrong chunk ids.
- **Follow-up questions are searched as standalone queries** (`ReasoningAPI.rewrite_query`, `ReasonerPhase.rewrite_query`).
  - "And for how long should it be traded?" is searched as "For how long should the Bund June contract FGBL M7 be traded?", which moves its answer from RRF rank 18 to rank 1.
  - The reasoner and Gemma still read the user's words.
  - Session memory strips `[Doc N]` tags, which pointed at other chunks in the next turn.
- **The transcript around the best hits is included** (`PostgresSearchRepository.fetch_neighbors`, `RetrievalAPI.expand_neighbors`, `RAG_NEIGHBOR_EXPANSION_TOP_K=3`). Both chat pipelines add the chunks before and after the top 3 survivors, for answers that cross a chunk boundary. The context budget drops these first.
- **Measured end to end** (`--stage answer`, before → after):
  - `chat`: answered 22/28 → 26/28; cites a relevant chunk 21/28 → 25/28; unanswerable questions refused 6/6 → 5/6; about 32 s → 24 s per question.
    - The one miss (u03, "what exactly did Draghi say") is not a hallucination: the answer cites and summarises the chunk that says to watch Draghi's answer on the replay.
  - Deep Research: answered 12/30 → 19/30; clarification requests 24 → 17; 4225 s → 3206 s.
    - It still asks for clarification on about half of the answerable questions, as its prompt instructs ("prefer ask_clarification").
- **No model load just to refuse.** With no surviving evidence, `generate`/`generate_with_critique` return the prompt's refusal directly, and `LiveQAPipeline` discards the Gemma preload.

## Deep Research: multi-pass reasoning chat, opt-in alongside `chat()`

- **New `chat_deep_research()` entrypoint** (`app/features/qa_reasoning/` (new), `app/features/qa_generation/` (`critique_answer`/`generate_with_critique`, additive), `app/features/session_memory/` (new), `app/orchestration/service/phases/reasoner_phase.py` (new), `app/orchestration/service/phases/qa_generation_phase.py` (`execute_with_critique`, additive), `app/orchestration/service/pipelines/live_deep_research_pipeline.py` (new), `app/engine/service/onyx_engine.py`, `app/engine/service/bootstrap.py`, `app/main.py`).
  - Implements the design in `AGENT_DESIGN.md`: a Qwen2.5-7B-Instruct Reasoner (`ModelType.QA_REASONER`, its own `ModelType` despite reusing the same GGUF file as `ASR_PHASE_3`/`VISUAL_CUE_DETECTION`) does chain-of-thought reasoning over an accumulating, chunk-id-deduped evidence pool and picks `ANSWER_READY` / `REFINE_QUERY` / `ASK_CLARIFICATION` each turn. `REFINE_QUERY` re-runs the existing, untouched retrieve → rerank → scrub stack. `ANSWER_READY` hands off to Gemma, which drafts an answer and immediately self-critiques it in the same model acquisition (`generate_with_critique`); a `NEEDS_MORE_EVIDENCE` verdict loops back to the Reasoner with the gap description, bounded by `DEEP_RESEARCH_MAX_ITERATIONS`/`DEEP_RESEARCH_MAX_CRITIQUE_RETRIES`.
  - `LiveDeepResearchPipeline` is the first looping/branching pipeline in the codebase — every other `Live*Pipeline` runs its phases exactly once in a fixed sequence. It reuses the same `RagRetrievalPhase`/`RagRerankingPhase`/`RetrievalScrubberPhase`/`QAGenerationPhase` instances `LiveQAPipeline` already uses.
  - Cross-question memory (what the user asked/was answered last turn, fed into the Reasoner's first iteration) is a new minimal `session_turns` table — one row per turn, only the most recent read back — separate from the intra-question evidence pool, which is ephemeral and discarded when the call returns.
  - Nothing about `LiveQAPipeline`, `QA_SYSTEM_PROMPT`, or the existing single-pass `chat()`/`generate()`/`scrub_retrieved_chunks()`/`search()` call sites changed — this is purely additive, reached via CLI option 7 (`app/main.py`).

## Chat citations resolve to their clips

- **Chat citations point to the same clips as search results** (`qa_generation/domain/entities.py`, `qa_generation/utils/citation_parser.py`, `engine/service/onyx_engine.py`, `app/main.py`).
  - **Introduced in:** `d25b030` (`parse_citations` wrote the placeholder "Unresolved Path") and `b8493d0` (`OnyxEngine._resolve_citation_paths`).
  - **Fixed in:** `f607ccb`.
  - **The problem.** Every chat citation printed as `Unresolved Path (Doc [Doc 1])`, while standard search showed each chunk's clip path and time range.
  - **The causes.**
    - The parser stored the placeholder as the path.
    - The engine only resolved an empty path, so the placeholder counted as resolved.
    - Even then the engine read a `source_id` that `QACitation` didn't have.
    - The unit test hid this by attaching `source_id` to a citation by hand.
  - **The fix.**
    - `QACitation` carries the chunk's `source_id` and `extracted_file_id`, and its `physical_path` starts empty.
    - The engine resolves citations with the same helper as search: the extracted clip, otherwise the source file.
    - Both CLIs print `[Doc N] <path>` with the citation's time range.

## Frozen CLI after ingestion and spoken prices indexed as on-screen text

Found on the re-ingest of the 74-minute video after `4319c65`.

- **FFmpeg no longer takes over the terminal** (`clip_extractor/data/ffmpeg_video_adapter.py`, `audio_extractor/data/ffmpeg_adapter.py`, `frame_extraction/data/ffmpeg_adapter.py`, `legacy_features/scene_detector/data/ffmpeg_scene_adapter.py`).
  - **Introduced in:** `c94dd34` (FFmpeg adapters run without closing stdin) and `7f7015f` (clip extraction runs alongside RAG indexing).
  - **Fixed in:** `4446a6b`.
  - **The problem.** After a successful ingestion the menu prompt accepted no input. The terminal was left in raw mode (`-icanon -echo -icrnl`): Enter sent a carriage return that `input()` never saw as a line end, and typing wasn't echoed.
  - **The cause.** FFmpeg reads the terminal for its interactive keys and switches it to raw mode while it runs; overlapping or interrupted runs leave it that way.
  - **The fix.** Every FFmpeg/FFprobe call now passes `-nostdin` and `stdin=subprocess.DEVNULL`.
  - **Not a model leak.** The 280 MiB the process kept on the GPU after the run is CUDA runtime overhead (PyTorch, cuBLAS/cuDNN, llama.cpp), released only when the process exits. A bare process that just touches CUDA already uses 122 MiB, and the smallest model Onyx loads is over 1 GB.
- **A bare number the speaker says is not accepted as on-screen text** (`visual_analysis/utils/grounding_check.py`).
  - **Introduced in:** `2e49ade`. The speech-copy filter only looked at text items of two or more content words.
  - **Fixed in:** `4446a6b`.
  - **The problem.** On two frames of the "BoE REVERSAL 22/3/18" chart (1337.8 s and 1415.1 s) the axis labels are unreadable, but the VLM listed "141.90", "142.04" and "141.46", all prices the speaker says. They were indexed as `[Visual Context: … referencing specific price levels ("141.90", "142.04", "141.46") …]`.
  - **The fix.** A text item that is only a number, or a number plus one word, is dropped when the speech contains it. A spoken value that really is on screen loses only that one item.

## Model preload wait limit removed

- **A preload waits for the GPU as long as the phase using it runs** (`core/model_manager/service/orchestrator.py`, `core/config.py`, `core/model_manager/exceptions/`; `MODEL_WAIT_FOR_CLEAR_TIMEOUT_SEC` and `ModelAcquireTimeoutError` removed).
  - **Introduced in:** `3bfde2d` (review batch 1). It capped `acquire_model(wait_for_clear=True)` at 120 s, so that a leaked model could not park the caller forever.
  - **Exposed by:** `2e49ade`. Visual analysis of the 74-minute video grew from 71 s to 174 s.
  - **Fixed in:** `cc7f34b`.
  - **What happened on the re-ingest.**
    1. The visual hallucination preload is queued before visual analysis. It gave up after 120 s while the VLM was still working, so that phase failed.
    2. The synthesis preload, already queued, then loaded with no phase left to release it.
    3. The embedder preload waited behind that model, timed out too, and indexing produced 0 vectors.
  - **Why the limit was wrong.** Nothing had leaked: every model was released when its phase finished. A resident model belongs to a phase that is still working, and batch phases run as long as the video needs. Any video about twice as long would have failed the same way before `2e49ade`.
  - **Why removing it is safe.** The leak the limit guarded against was already fixed in `3bfde2d` itself: every pipeline discards unused preloads in `finally`, and the engine releases all models after an ingestion or a failed live call. The wait itself (`ce4374d`) stays, because a preload must never evict a model in use.
- **A halted visual branch's preloads are released before the embedder preload** (`orchestration/service/pipelines/batch_media_pipeline.py`).
  - **Introduced in:** `14cd969` (synthesis preload queued before visual hallucination correction) and `e6738a7` (visual hallucination preload queued before visual analysis).
  - **Fixed in:** `cc7f34b`.
  - **The problem.** If a visual phase fails or has nothing to do, preloads already triggered for later phases still load and stay resident until `finally`. With no wait limit the embedder preload would wait behind them forever.
  - **The fix.** `_discard_pending_preloads()` now runs right before the embedder preload; every earlier phase has already run by then. A unit test covers a failed visual hallucination correction.

## Phonetic over-correction and weakly grounded visual descriptions

Found on the second real ingestion of the 74-minute Axia video. Each change was measured offline on the stored run before and after with `scripts/eval/` (nothing written to the database).

- **Phonetic correction only fixes mis-heard glossary terms** (`domain_phonetic_correction/utils/term_candidate_finder.py`, `utils/phonetic_key.py`, `utils/span_replacer.py`, `data/qwen_adapter.py`, `data/prompts.py`, `service/correction_api.py`, `PHONETIC_CANDIDATE_MIN_SIMILARITY`, `PHONETIC_LENGTH_RATIO_MIN/MAX`, `PHONETIC_MAX_WORD_PROBABILITY`; `PHONETIC_MIN_SIMILARITY` removed).
  - **Introduced in:** `a0c6772` (the free-rewrite `<orig>/<corr>` patch prompt) and `9a526c1` (the letters-only similarity gate, which passed "those 73s" → "those 73 Bunds" at exactly 0.6 because "73s" appears inside "73bunds").
  - **Fixed in:** `7183566`.
  - **The problem.** Whisper had transcribed the speech correctly, and the model rewrote it towards written or glossary form. Both patches applied on the stored run were wrong ("in the bund" → "in Bunds" as well). An earlier run proposed "very first" → "initial", "Bank of England" → "BOE", "13.50.45 seconds" → "13:50:45" and "that level" → "142.27".
  - **Candidates come first, and they are deterministic.**
    - A span becomes a candidate only if it sounds like exactly one dictionary term: letters and a small sound key are compared, and every heard word and every term word must supply its letters.
    - A span is never a candidate if it contains a digit, is already the term up to case, hyphens or plural, holds a correctly written term, is an ordinary word matched to an acronym, or would turn a plural into a singular or a verb form into a noun.
    - Spans whose Whisper words were all heard with probability ≥ 0.95 are skipped.
  - **The model only confirms.** Qwen 3B picks between the sentence as transcribed and the sentence with the term (A/B, under a grammar), in both orders. A candidate is applied only if the term wins both times.
    - Asked for true/false verdicts in a batch, it rejected all 18 real mis-hearings.
    - With a single order, position bias accepted 2–3 ordinary words ("bond" → "Bund").
  - **The edit is deterministic.** Only the exact span is replaced, with the term in the heard singular/plural form. The model never writes text, so it can't paraphrase, insert, delete or change digits.
  - **Measured on the stored transcript** (334 real sentences) plus 20 labelled mis-hearings, with the real model:

    | | Before | After |
    |---|---|---|
    | Precision | 0.80 | 1.00 (held-out half: 1.00) |
    | Recall | 0.60 | 0.85 (held-out half: 0.90) |
    | False edits on real speech | 2 | 0 |
    | Digit changes | 0 | 0 |
    | Time | 83 s | 12 s |

    - Four of the mis-hearings are genuine: re-transcribing the video without hotwords gave "burn", "burns", "front" and "boom" for "Bund(s)". Whisper heard ECB, Draghi and Bank of England correctly even without hotwords. The other 16 are plausible hand-written confusions.
  - **Trade-offs.**
    - Mis-hearings that neither look nor sound like the term ("front" or "boom" for "Bund") aren't found.
    - Oversized sentences are no longer split for the model; the verifier sees a window around each candidate.
    - Spelled-out acronyms ("E C B") are recovered again, reversing the trade-off noted in `9a526c1`.
    - The model is loaded only when a candidate exists.
- **Frame descriptions come from what the VLM can read, not from the speech** (`visual_analysis/data/prompts.py`, `data/vlm_adapter.py`, `utils/frame_description_parser.py`, `utils/grounding_check.py`, `utils/observation_renderer.py`, `service/analysis_api.py`, `VISION_MAX_NEW_TOKENS_PER_FRAME`).
  - **Introduced in:** `c557fe6` (free-text frame descriptions) and `112225e` (review batch 4 prompt: transcript in the prompt plus a free-form "describe each frame" instruction).
  - **Fixed in:** `2e49ade`.
  - **The problem.** 11 of the 20 stored descriptions paraphrased the speech ("The speaker explains the concept of a position squeeze…") and 9 were indexed. Yet every one of those frames showed a slide with legible text: "POSITION SQUEEZE" bullets, "BoE REVERSAL 22/3/18 / Result 7-2 for hike BUT", chart callouts. Values such as "142.10 → 142.70" came from the speech, not the screen.
  - **Fields, not prose.** The VLM now reports `SHOWS`, `TEXT` (quoted legible text, at most 8 items), `MARKS` (drawn marks or pointers and what they point at) and `LINK` (the spoken reference they show). The stored description is rendered from those fields, with what is on screen and what the speaker refers to in separate sentences.
  - **A frame is described only if visible evidence survives deterministic filters.** They remove:
    - speaker-only frames;
    - watermarks, player chrome and outline sidebars;
    - text copied from the speech (content words sitting together in one stretch of speech, e.g. "160.84 if we can get back above 63", or the spoken sentence reported as text on an X-ray);
    - marks aimed at nothing specific, or naming a spoken number;
    - text or marks another frame of the same chunk already reported.
  - **Measured on the 20 stored frames**, against ground truth written by viewing each frame (`scripts/eval/visual_labels.json`):

    | | Stored | Now |
    |---|---|---|
    | Paraphrases | 11 | 0 |
    | Anchored to legible text or numbers | 5 | 14 |
    | Generic ("a ladder chart with columns") | 4 | 0 |
    | No description | 0 | 6 |
    | Slide title captured | 7 | 13 |
    | Spoken numbers stated as visible | 2 frames | 0 |
    | Visual analysis time | 71 s | 175 s |

    - The 6 frames without a description repeat an earlier frame of their chunk, or show nothing legible.
    - The scorer still flags one frame, but its "0" is the chart's own footer text ("0 in View").
  - **Tried and dropped.**
    - One image per call: 217 s, plus spoken numbers inside marks and repetition loops.
    - Bare `...` placeholders in the format: the model copied them for frames with nothing legible.
  - **Known limits.**
    - In a multi-frame call the model sometimes gives one frame the text of another frame in the same chunk (403.4 s, 1107.3 s). The text is on screen during that chunk, but its frame is wrong.
    - OCR slips survive ("W1" for "WVPOC").
    - A slide bullet the speaker reads out word for word is dropped.
    - The 4-bit 7B model's output shifts noticeably with small prompt changes, so re-run `scripts/eval/visual_eval.py` after any prompt edit.
- **The visual check accepts on-screen text that shares the chunk's topic** (`visual_hallucination_correction/utils/topic_overlap.py`, `data/prompts.py`, `service/correction_api.py`, `VISUAL_CHECK_MIN_SHARED_TOPIC_WORDS`).
  - **Introduced in:** not a regression. The text-only 3B checker (accept-biased prompt from `9a526c1`) can't see the frames.
  - **Fixed in:** `2e49ade`.
  - **The problem.** On the rendered descriptions it rejected 4–7 of 12 matching chunks between runs, with self-contradicting reasons ("text about a trading drill and ECB statements… completely unrelated to a slide about trading drills and ECB statements").
  - **The change.** A chunk is accepted without the model when its quoted on-screen text shares at least 2 topic words, or one specific number such as "7-2", with the chunk's text and context. Only other chunks go to the model, which is loaded only when needed. The prompt now explains the description format.
  - **Measured.**
    - 10 of 12 real chunks accepted. The 2 rejections come from the model, on chunks with fewer than 2 shared words.
    - Medical and legal descriptions tried against the same speech: all 24 rejected.
    - An A/B "real frames or a decoy" checker was tried and dropped: it accepted 5 of 12 legal mismatches.
- **Visual synthesis can't add numbers or quotes** (`visual_context_synthesis/data/prompts.py`, `utils/narrative_guard.py`, `service/synthesis_api.py`).
  - **Introduced in:** `38b24e3` (synthesis prompt without grounding rules).
  - **Fixed in:** `2e49ade`.
  - **The change.** The prompt requires on-screen text and numbers to be copied exactly, and the speaker's reference to stay separate from what is on screen.
  - **The guard.** A narrative with a number or quote that no frame description contains is replaced by the descriptions themselves. On the stored frames this happened for 2 of 10 narratives.
  - **Limit.** The guard checks numbers and quotes only; editorial wording ("a series of cryptic phrases") can still appear.
- **Ingestion time.** On the stored run, phonetic correction dropped from 64 s to about 12 s, while visual analysis rose from 71 s to 175 s and synthesis/check changed little. Expect roughly +50 s overall (about 530 s, 12% of the video's length). Confirm on the re-ingest.

## Whisper prompt leakage and sub-domain router

Found while inspecting the first real ingestion; both were verified on the stored run before any code changed.

- **Whisper no longer gets a sentence prompt, and prompt text it repeats is removed** (`domain_adaptation/utils/glossary_builder.py`, `domain_adaptation/data/prompts.py`, `transcription/utils/prompt_echo_filter.py`, `transcription/service/transcription_api.py`, `WHISPER_PROMPT_ECHO_MIN_TOKENS`).
  - **Introduced in:** `9bc403a` (review batch 3), which replaced the instruction prompt with "This recording discusses {domains}, including terms such as {terms}."
  - **Fixed in:** `05ed5f4`.
  - faster-whisper feeds `hotwords` and `initial_prompt` to the model as "previous transcript" text. Whisper copied the prompt into the 74-minute video's transcript 5 times, at high confidence (no-speech ≈ 0.01, not flagged), and it reached 6 indexed chunks.
  - `GlossaryBuilder` now returns an empty initial prompt; the hotword list carries the vocabulary bias.
  - `TranscriptionAPI` removes any run of at least 5 consecutive words that appears, in order, in the hotwords or prompt, before the transcript is saved. On the stored run this removes exactly the 6 leaked runs (60 words). With the trading hotwords it removes nothing from real speech. At 3 words, real mentions of "Bank of England" would also be removed, hence 5.
- **The sub-domain router scores topics instead of copying the label list** (`domain_adaptation/data/llama_adapter.py`, `domain_adaptation/data/prompts.py`, `domain_adaptation/utils/glossary_builder.py`, `DOMAIN_ROUTER_MIN_SCORE`).
  - **Introduced in:** `251ec04` (router prompt with a bare label list) and `0921cd6` (grammar accepting any string).
  - **Fixed in:** `05ed5f4`.
  - Asked for "the sub-domains", the 3B model returned every label in listed order: all 8 `PROP_TRADING_MACRO` labels for an ECB/BoE video. That put irrelevant names ("Macklem, Poloz, Loonie, Ueda") into Whisper's hotwords and 153 terms into the phonetic dictionary.
  - It now rates every label 0–10 against a short description from the glossary. The grammar allows only real labels, and labels scoring at least 5 are kept, highest first.
  - Checked results: the ECB/BoE video gets European Central Bank 9 and Central Bank Macro 8, rest 0. The options-trading test clip gets `OPTIONS_TRADING` 10 with `FINANCE`, and all 0 with `PROP_TRADING_MACRO`. An unrelated medical transcript gets all 0. A stricter "select only clearly discussed topics" list prompt returned nothing even for the video, so it wasn't used.
  - **Known limit (follow-up):** Bank of England scored 0 on that video. The scout hears at most 2 minutes of speech from each of the start, middle and end, and those windows mostly cover the ECB part. Too few sub-domains only means fewer hotwords and fewer phonetic corrections, never wrong ones.

## Fixes from the first real ingestion

The first full ingestion of a real 74-minute video (Axia Futures, ECB replay drills) ran in 547 s, 12% of its length, but the indexed text was damaged. The causes were traced in the stored data. Each item names the commit that introduced the problem and the one that fixed it; most were regressions from review batches 3 and 4.

- **The appliance never used a trading glossary** (`APPLIANCE_INDUSTRY`, `data/glossaries/`).
  - **Introduced in:** `b85f6cb` (setting created as `GENERAL`); `0e5b021` deleted the options `finance.json`.
  - **Fixed in:** `.env` (not tracked; key added to `.env.example` in `9a526c1`) and `2628585` (`finance.json` restored).
  - The industry has been `GENERAL` since the setting was introduced, and domain adaptation only chooses sub-domains inside the configured industry.
  - A trading video therefore got legal/medical Whisper hotwords ("affidavit, plaintiff, diagnosis, cardiology") and a legal/medical phonetic dictionary.
  - Fix: `.env` sets `APPLIANCE_INDUSTRY=PROP_TRADING_MACRO`, and the key is now listed in `.env.example`.
  - The original options-trading `finance.json` is restored. It was deleted in `0e5b021` and replaced by the integration-test copy.
- **Phonetic patches must sound like what was heard** (`domain_phonetic_correction/service/correction_api.py`, `PHONETIC_MIN_SIMILARITY`).
  - **Introduced in:** `9bc403a` (review batch 3: dictionary-term-only grounding, minimum dictionary size lowered to 1).
  - **Fixed in:** `9a526c1`.
  - The old rule only required a patch to introduce a dictionary term, which let the model turn ordinary words into terms. All 14 applied patches were wrong ("Bank of England"→"Onyx", "decision"→"prognosis").
  - Each changed word region now needs letters-only similarity of at least 0.6. Inserted or deleted words are rejected, and case-only changes are allowed only for acronym-like terms.
  - Regions are judged separately, so a two-edit patch can't pass on the words the edits share.
  - Trade-off: spelled-out acronyms ("the sea a grr"→"CAGR") are now rejected.
- **Whisper punctuation is stripped before the punctuation model** (`sentence_reconstruction/data/punctuation_adapter.py`). For "develop," the last subtoken was the comma itself, labelled "no punctuation", and the assembler also removed Whisper's mark. Periods and commas were erased, giving "sentences" of up to 5,184 characters.
  - **Introduced in:** `9bc403a` (review batch 3 punctuation adapter rewrite).
  - **Fixed in:** `9a526c1`.
- **Numbers and hyphenated words stay intact** (`sentence_reconstruction/utils/deterministic_assembler.py`, `diarization/utils/aligner.py`).
  - **Introduced in:** `2802326` (assembler joined every word with a space) and `9bc403a` (aligner rebuilt split segments the same way).
  - **Fixed in:** `9a526c1`.
  - Whisper marks sub-word continuations ("160" + ".99", "pre" + "-planning") by omitting the leading space. Rejoining every word with a space produced "160 .99" and "pre -planning".
  - Continuations now attach to the previous word, and the word before one never takes punctuation or ends a sentence.
- **The visual hallucination check no longer rejects matching descriptions** (`visual_hallucination_correction/data/prompts.py`). The batch 4 "merely restates the transcript" rule made the 3B checker reject 6 of 10 descriptions that matched their slides. The rule is removed; temperature stays 0.0.
  - **Introduced in:** `112225e` (review batch 4).
  - **Fixed in:** `9a526c1`.
- **Echoed speaker labels are stripped case-insensitively** (`contextual_retrieval/data/slm_adapter.py`). Diarization emits `speaker_0`, but the regex only matched `Speaker_`.
  - **Introduced in:** `36d4810` (review batch 2).
  - **Fixed in:** `9a526c1`.
- **Dev tooling.**
  - `scripts/reset_dev_environment.sh` wipes the dev database and storage vault, replacing `reset_dev_db.sh`.
  - `scripts/init.sql` used `#` comments, which aren't valid SQL, so re-applying it failed. **Introduced in:** `0c88f9a`. **Fixed in:** `2628585`.
  - Integration tests seed glossaries into a throwaway directory (new `GLOSSARY_PATH` setting, set by `integration_tests/conftest.py`). The conftest copied every test resource into `data/glossaries/`, overwriting the restored options `finance.json` on every run. **Introduced in:** `e240389`. **Fixed in:** `9a526c1`.

## Review fixes: stability, retrieval grounding, audio correctness, throughput

Fixes from the full review of RAG retrieval/storage and the audio/video ingestion pipelines, delivered in four batches on `fix/review-pipeline-retrieval`. Each bullet is one change: what changed, why, and where.

### Batch 1 — Stability & data integrity

- **Model residency is keyed on `(model_type, model_path)`** (`core/model_manager/service/orchestrator.py`). The same GGUF is loaded by several adapters with different settings (e.g. Llama 3.2 3B at `n_ctx` 4096 for domain classification and 32768 for contextual retrieval), so a resident instance could previously be handed to the wrong feature. The idempotency check also now runs under the orchestrator lock instead of iterating `_active_models` lock-free.
- **`acquire_model(wait_for_clear=True)` has a timeout** (`MODEL_WAIT_FOR_CLEAR_TIMEOUT_SEC`, default 120s) and raises the new `ModelAcquireTimeoutError` (`core/model_manager/exceptions/timeout.py`). A leaked model used to park the caller — and the single preload worker thread — forever.
- **Unresolved preloads are discarded, not leaked.** `BackgroundModelWorker.discard_preload` / `AsyncModelService.discard_preload` cancel a queued preload or release the model as soon as a running one finishes. Every GPU phase clears its future once resolved and exposes `discard_preload()`, and all five pipelines call it for every phase in a `finally`. Previously a failed live query (e.g. an empty chat prompt) left the scrubber/QA preloads loading with nobody to release them, hanging every later search and chat until restart; the batch pipelines had the same leak on every early return.
- **The engine evicts all models after ingestion and after any failed live call** (`OnyxEngine._cleanup_after_ingestion`, `_release_models_after_failure`, via `AsyncModelService.release_all`). Before, cleanup only ran `gc` and orchestrator-tracked models stayed resident.
- **Blank search prompts are rejected at validation** (`SearchQuery.prompt` is stripped and requires `min_length=1`), instead of failing inside the embedder after a model was already loaded.
- **Clip extraction cut the wrong length** (`clip_extractor/data/ffmpeg_video_adapter.py`). With input seeking (`-ss` before `-i`), FFmpeg treats an output `-to` as a *duration*, so a chunk at 3600–3700s produced a ~3700s clip (verified on `integration_tests/resources/test_video.mp4`: a 3→5s request produced 5s). The command now passes `-t (end - start)`.
- **Clips vaulted before a mid-run failure are deleted** (`ClipExtractorAPI` + new `StorageAPI.delete_raw_file` / `IStorageRepository.delete_file`). They had no DB rows, so they became orphaned files and were re-extracted on the next run.
- **Job output is actually persisted** (`core/jobs/data/repositories.py`). `finish_job` wrote to the unmapped attribute `output_result`, so nothing was ever stored; it now writes the `output_params` column, which the `Job` entity reads back. `JobManager` stores a compact JSON-safe summary (scalar dict entries without `_`-prefixed transient keys, or `{result_type, result_id}`) instead of `str()` of whole transcripts.
- **RAG indexing is atomic and failures are real failures** (`rag_storage`). Chunks are embedded first, then `replace_rag_chunks` deletes and inserts in one transaction — previously the old index was deleted before embedding, so a failure left the source with no chunks. `index_transcript` raises typed `RAGStorageError`s instead of returning `success=False` (which marked the job COMPLETED), and `map_storage_error` always returns a feature exception with the original attached.
- **Silent-track fallback only for media with no audio stream** (`audio_extractor/data/ffmpeg_adapter.py`). Streams are counted with `ffprobe`; previously any FFmpeg stderr containing "Invalid argument" was treated as "no audio", silently indexing a video with an empty transcript.
- **Diarization is crash-safe and fallback use is recorded** (`diarization`). The map (the idempotency marker) is saved after the aligned transcript is stored, so a crash in between no longer skips alignment forever. Clustering fallbacks log a warning instead of `print` and set `DiarizationMap.fallback_applied` (new column).

### Batch 2 — Retrieval & storage correctness (Contextual Retrieval alignment)

> **Action required:** the `vector_chunks` schema changed (new `raw_text`, `context_text`, `speaker_labels` columns). There are no migrations, so reset the dev database and storage vault with `scripts/reset_dev_environment.sh` and re-ingest. To enable reranking, download `BAAI/bge-reranker-v2-m3` to `models/bge-reranker-v2-m3`; without it, search falls back to RRF order.

- **The answer model only sees verbatim transcript text** (`qa_generation/utils/citation_parser.py`, `data/prompts.py`). The context block used to be the full `text_payload` — `[Audio Context: …][Visual Context: …]` plus transcript — so an answer grounded only in an SLM/VLM-written context sentence still received a valid `[Doc N]` citation. Each block now carries `raw_text` under a `[Doc N] (HH:MM:SS–HH:MM:SS)` header, and the prompt explains speaker labels and forbids inventing names for them.
- **Raw transcript, generated context and speakers are stored separately** (`UnifiedChunk.raw_text` / `context_text` / `speaker_labels`, `VectorChunkModel` columns, `PostgresRagRepository`, `RetrievedChunk`). `text_payload` stays the contextualized string that is embedded and keyword-indexed, exactly as Anthropic's Contextual Retrieval prescribes; the generated context is never presented as evidence.
- **Chunks and the contextualizer's document are speaker/timestamp-labelled** (new `contextual_retrieval/utils/transcript_formatter.py`). Text is rendered as `[Speaker_0 @ 00:12:31]: …` turns in both the chunk (`ContextualizedChunk.original_text`) and the document the SLM reads. Previously only `sentence.text` was joined, so "who said what, when" never reached retrieval or answers. The context SLM itself reads a lighter `Speaker_0: …` form (no brackets/timestamps), its prompt forbids starting with a speaker label, and `SLMContextAdapter` strips any echoed leading label — with timestamped labels, Llama 3.2 3B answered the integration test with nothing but `[Speaker_0 @ 00:00:00]`.
- **Long transcripts no longer fail contextual retrieval** (`contextual_retrieval/utils/document_window.py`, `ContextualizerAPI`, `SLMContextAdapter.count_tokens`). The whole transcript was sent into a fixed 32k context; a fast-talking or >2h recording overflowed it and failed the source after every expensive phase had run. If the transcript fits (`CONTEXTUAL_RETRIEVAL_N_CTX - CONTEXTUAL_RETRIEVAL_RESERVED_TOKENS`), behaviour is unchanged (one stable, KV-cached prefix); otherwise chunks are situated in overlapping document sections, reusing the cached prefix until the section changes.
- **Keyword search uses OR semantics** (`rag_retrieval/data/postgres_search_repo.py`). `plainto_tsquery` ANDs every lexeme, so a natural question like "what did the doctor say about the MRI results" only matched chunks containing *all* its terms and the sparse half of hybrid search effectively never contributed. The query is rewritten to OR (`' & '` → `' | '`, keeping Postgres stemming/stop words), ranked with length-normalized `ts_rank(…, 1)`, and a stop-word-only question yields no sparse query. The text search configuration is the new `FTS_LANGUAGE_CONFIG` setting.
- **New `rag_reranking` feature: cross-encoder reranking** (`app/features/rag_reranking/`, `RagRerankingPhase`). Anthropic's results show reranking gives the largest single gain on top of contextual embeddings + BM25. Retrieval now returns an RRF-fused candidate pool (`RAG_RETRIEVAL_CANDIDATES`, 40) and `CrossEncoderRerankerAdapter` (bge-reranker-v2-m3, fp16) narrows it to `query.limit`. All three live pipelines run retrieve → rerank (→ scrub → generate). Without weights on disk the phase skips its preload and returns the top chunks in RRF order.
- **`SearchQuery.limit` defaults to 10** (max 20) and the CLI no longer hard-codes 5, so the reranker and QA model see a useful number of chunks.
- **One malformed scrubber verdict no longer fails the whole query** (`RetrievalScrubberAPI`). A JSON parse failure on a chunk keeps that chunk (the prompt already says "when in doubt, accept") and records `ScrubAudit.evaluation_failed=True`; a non-object JSON verdict counts as a parse failure. Model/OOM errors still abort.
- **Scrubber, QA and reranking phases only retain a model they own** (`release_after=loaded_model is None`). If a phase runs without a resolved preload, the feature API now releases the model it acquired instead of leaving it resident.
- **Shared torchcodec import guard** (`core/model_manager/utils/torchcodec_guard.py`). The `sys.modules["torchcodec"]` workaround was duplicated in both embedders; it is now one context manager used by both embedders and the reranker.
- **Integration tests create the full schema regardless of which tests run** (`integration_tests/conftest.py`). Hybrid search joins `video_clips`, but that table was only created if some collected test imported its model, so the live pipeline tests failed when run on their own. The conftest now imports every `app/**/data/models.py` before `create_all`.

### Batch 3 — Audio correctness

- **Every audio phase reads the normalized 16kHz mono WAV** (`AudioExtractorAPI.get_extracted_audio_path`, `DomainAdaptationPhase`, `TranscribePhase`, `DiarizePhase`). Only VAD used the extracted WAV; domain adaptation, transcription and diarization re-decoded the original media (e.g. 48kHz stereo AAC inside an MP4) — three full decodes and resamples per file, and a different decoder path than the VAD map was computed from. `AudioExtractorAPI` now also accepts AUDIO sources, and `BatchAudioPipeline` runs `ExtractAudioPhase` instead of pointing phases at the original file.
- **VAD scores windows in GPU batches** (`vad/data/nemo_adapter.py`, `VAD_BATCH_SIZE`). One forward call (plus a GPU sync) per 0.63s window meant ~90k calls for a 2-hour recording; windows are now built with `unfold` and scored hundreds at a time.
- **VAD segments no longer clip the first/last words** (`VAD_PAD_SEC`). Windows are timestamped at their centre instead of their start (which ended segments up to ~0.5s before speech stopped), segments are padded by 0.3s and overlapping pads merged, and audio shorter than one window is still scored.
- **Short utterances are transcribed, not dropped** (`transcription/data/faster_whisper_adapter.py`, `WHISPER_MIN_SLICE_SEC`). Spans under 0.4s were skipped, silently losing isolated answers like "Yes." or "Objection."; they are now widened symmetrically to at least 1s.
- **Batched Whisper transcription** (`WHISPER_BATCH_SIZE`, default 8). VAD segments are packed into ≤29.5s clips that start and end only in silence and decoded with faster-whisper's `BatchedInferencePipeline` (`clip_timestamps`, word timestamps, absolute times). Because batched mode transcribes only the first 30s of any clip, a single uninterrupted stretch of speech longer than that goes through the sequential path instead — nothing is truncated. `WHISPER_BATCH_SIZE=1` restores fully sequential decoding.
- **Whisper domain hints are well-formed** (`domain_adaptation/utils/glossary_builder.py`, `data/prompts.py`). faster-whisper feeds `hotwords` into the prompt as plain text, so `"term:15.0"` injected literal `:15.0` noise into every window; hotwords are now a plain comma list. The instruction-style initial prompt ("Use proper capitalization…"), which Whisper does not follow and can regurgitate, is replaced by a natural sentence containing the domain terms.
- **Hallucination correction stops deleting real speech** (`hallucination_correction/utils/heuristics.py`). Any single signal (`no_speech_prob > 0.6`, `avg_logprob < -1.0`, or `compression_ratio > 2.4`) deleted the segment — including accented, jargon-heavy or poorly recorded speech. Deletion now requires low log-probability together with likely silence or a repetition loop (Whisper's own rule); a segment tripping a single signal is kept and marked `TranscriptSegment.low_confidence=True` (counted in `CorrectionTelemetry.flagged_low_confidence`).
- **Speakers are attributed per word** (`diarization/utils/aligner.py`). A Whisper segment was assigned one speaker by maximum overlap, so a quick question-and-answer inside one segment gave the answer to the asker. Each word now gets its best-overlapping speaker (nearest speaker if it falls in a gap), segments containing a speaker change are split, and lookups use a sorted index instead of an O(segments × speakers) scan. `"Unknown"` only occurs when diarization produced no segments.
- **Long-form diarization clustering** (`diarization/data/nemo_adapter.py`). `chunk_cluster_count` / `embeddings_per_chunk` enable NeMo's chunked clustering so multi-hour recordings don't build one affinity matrix over every embedding.
- **Punctuation reads the last subtoken of each word** (`sentence_reconstruction/data/punctuation_adapter.py`). The HF pipeline + character-offset mapping took the first overlapping entity group, which dropped punctuation on multi-subtoken words (run-on sentences → worse chunks) and could leave the model's `"0"` ("no punctuation") label unconverted. The adapter now tokenizes pre-split words in overlapping 230-word windows (`word_ids()`, shrinking windows instead of truncating) and maps labels exactly 1:1; unknown labels become `NONE`.
- **Sentence assembly keeps casing and never writes tags as text** (`deterministic_assembler.py`). `str.capitalize()` turned "MRI" into "Mri"; only the first letter is uppercased now. Only real punctuation marks are rendered, so a stray `"0"` tag can no longer produce text like "So0" or "drill0".
- **Phonetic correction only applies dictionary-grounded patches** (`domain_phonetic_correction/service/correction_api.py`, `PHONETIC_MIN_DICTIONARY_TERMS`). Patches were applied case-insensitively to every match without checking the dictionary, and the default GENERAL glossary has just 2 terms. A patch must now introduce a dictionary term (e.g. "high pole glycemia" → "hypoglycemia"); with an empty dictionary the LLM pass is skipped. Integration fixtures that swapped in test glossaries now restore `data/glossaries/*.json` afterwards instead of deleting the tracked files. Passthrough results are now persisted — an empty transcript previously returned without saving, so the next phase failed with "Phonetic Transcript missing".
- **Oversized sentences are split at whitespace** (`domain_phonetic_correction/utils/sentence_splitter.py`) with the word array divided in step with the text, instead of cutting mid-word by character index with a mismatched audit trail.
- **Unit-test hygiene:** `test_bootstrap_unit` now shuts down the engine it builds. Its leaked telemetry thread polled `torch.cuda.is_available()` every 0.5s and intermittently broke `test_generate_diarization_map_pure_mock`.

### Batch 4 — Throughput & visual branch

- **Frame extraction decodes each chunk once** (`frame_extraction/data/ffmpeg_adapter.py`). The adapter spawned one FFmpeg process per second of footage, each re-opening and re-seeking the video (about 3,600 processes per flagged hour). It now runs one `-ss start -t duration -vf fps=1` decode per chunk and stamps frame *i* at `start + i`. Frames written before an FFmpeg error are kept, and if an interval yields nothing, a single midpoint frame is taken as a fallback. The worker pool is `max(2, cpu_count // 2)` (`FRAME_EXTRACTION_WORKERS`) instead of `min(32, cpu_count * 2)` — each worker is now a multi-threaded decode, and the old pool oversubscribed the CPU.
- **Each chunk keeps at most 4 visually distinct frames** (`frame_extraction/utils/frame_selector.py`, `FRAME_MAX_PER_CHUNK`). Sequential pHash deduplication only collapses back-to-back duplicates, so a long chunk sent every changed second to the 7B VLM. Greedy farthest-point selection on pHash distance (sharpest first, ties to the sharper frame) keeps distinct scenes, e.g. a slide, a chart and a close-up, rather than neighbouring seconds of one slide.
- **An all-blurry chunk keeps its sharpest frame** (`frame_extraction/utils/quality_filter.py`, `FRAME_BLUR_THRESHOLD`). Low-texture evidence such as scanned pages and white slides falls under the Laplacian threshold, so a chunk the cue detector flagged ended up with no visual at all. Perfectly uniform frames (score 0) are still dropped.
- **The VLM may only describe what is visible** (`visual_analysis/data/prompts.py`, `vlm_adapter.py`). The transcript stays in the prompt to steer attention, with new rules:
  - describe only what is visibly present;
  - transcribe visible text and numbers exactly;
  - never state a value that isn't legible, even if the speaker says it;
  - say when a frame doesn't show what is described.

  Decoding is greedy (`do_sample=False`), replacing sampling at temperature 0.1, which could invent details and made results non-reproducible.
- **Frames are described in one multi-image call per chunk** (`VisualAnalysisAPI.process_chunk`, `IVisualLanguageModelAdapter.analyze_frames`). Previously every frame was a separate call that re-encoded the same prompt.
  - Up to `VISION_MAX_IMAGES_PER_CALL` (4) frames go in each call.
  - Each image is capped at `VISION_MULTI_IMAGE_MAX_PIXELS` (768·28·28).
  - The generation budget is `VISION_MAX_NEW_TOKENS_PER_FRAME` (160) × frames. This scales with the frame count instead of the planned flat 320, so frames full of text aren't truncated.
  - The model answers in `Frame N:` lines, split by `visual_analysis/utils/frame_description_parser.py`. An unformatted answer is kept and attributed to the first frame.
- **Visual timelines have real timestamps** (`FrameAnalysis.timestamp_sec`). `FrameAnalysis` had no timestamp, so `visual_context_synthesis/utils/fuzzy_grouper.py` stamped every event with the chunk start and sorted frames arbitrarily.
- **The visual hallucination check rejects transcript restatements** (`visual_hallucination_correction/data/prompts.py`). A description with no visual content that only paraphrases the speaker is now rejected — the VLM sees the transcript, so a restatement proves nothing about the screen. "The discussed item is not visible" still counts as a description. Temperature is 0.0.
- **Visual cue detection is deterministic and idempotent** (`visual_cue_detection`). Temperature is 0.0, and the new `ContextualizedChunk.cue_evaluated` flag lets a rerun skip chunks already judged. A fully evaluated transcript no longer loads the model or rewrites the row.

## Unreleased / In progress

- **Environment/hardware diagnostics**: new integration tests documenting two subprocess-isolated regressions — cross-feature `sys.modules["torchcodec"]` leakage in long-lived processes, and the confirmed llama.cpp SIGSEGV in the Qwen-VL vision encoder (native HF Transformers remains the required path for vision). See "Known gotchas" in `CLAUDE.md`.
- **Config/dependency updates** in progress alongside the diagnostics above.
- **Next milestone: the RAG-grounded conversational agent** — moving beyond the existing single-pass retrieve → verify → generate flow (`LiveQAPipeline`) toward multi-turn, tool-using interaction built on `RetrievalAPI`, `RetrievalScrubberAPI`, and `QAGenerationAPI`.

## QA Generation integrated into orchestration & engine

`OnyxEngine.chat()` now runs the full retrieve → verify → generate chain via `LiveQAPipeline`: hybrid RAG retrieval, hallucination-scrubbed filtering, then Gemma-generated answers with citations parsed back to source chunks.

## QA Generation feature

New `qa_generation` feature: `GemmaGenerationAdapter` loads a local Gemma GGUF model via `llama-cpp-python` (`n_gpu_layers=-1`, 8K context, flash-attn). `QAGenerationAPI.generate(query, retrieved_chunks)` builds a strict context block from retrieved chunks, runs chat completion, and parses citations back out of the generated text to their source chunk/document ids — this is the boundary that enforces "never answer outside the retrieved context."

## Verified search integrated into the engine

`retrieval_hallucination_scrubber` wired into `LiveVerifiedRetrievalPipeline` and exposed as `OnyxEngine.search_verified()`. "Verified search" = standard hybrid retrieval followed by an SLM relevance judge (`IRetrievalScrubberAdapter.evaluate_relevance`) that drops chunks it judges irrelevant or likely to produce a hallucinated answer, before anything reaches generation.

## RAG retrieval + contextual retrieval pipelines

`rag_retrieval` and `contextual_retrieval` features and their pipelines created and integrated. `RetrievalAPI.search(SearchQuery)` performs hybrid search: dense (pgvector cosine distance) + sparse (Postgres full-text `ts_rank_cd`) combined via Reciprocal Rank Fusion (`RAG_RRF_K_CONSTANT`), returning `RetrievedChunk` objects with text, timing, and a link back to the source clip.

## Retrieval hallucination scrubber

New `retrieval_hallucination_scrubber` feature: unit- and integration-tested SLM-based relevance judge for filtering irrelevant or likely-hallucinated retrieved chunks before they reach an LLM — the first half of what later becomes "verified search."

## Core ingestion pipeline (undated — pre-existing at time of writing)

The batch media/audio ingestion pipeline (`BatchMediaPipeline` / `BatchAudioPipeline`), built stage by stage as features under `app/features/` and wired together by `app/orchestration/`:

- Media ingestion & extraction (FFmpeg, SHA-256 dedup)
- Voice Activity Detection (NVIDIA NeMo MarbleNet)
- Domain adaptation / "Scout" (Faster-Whisper Tiny draft + DeBERTa v3 classifier + hotword extraction — detects Legal/Medical/General)
- High-fidelity transcription (Faster-Whisper Large v3, hotword-biased)
- Speaker diarization (NVIDIA NeMo TitaNet)
- Hallucination correction, sentence reconstruction (deterministic punctuation classification + speaker-aware assembly), domain phonetic correction (glossary-grounded local LLM correction with an audit trail)
- Visual branch: visual cue detection → frame extraction → visual analysis (Qwen2-VL) → visual hallucination correction → visual context synthesis
- Context unification and RAG indexing (embedding via BGE-M3 + storage into the hybrid pgvector/full-text table), run concurrently with clip extraction

Underpinning all of it: `app/core/model_manager`'s `ModelManagerOrchestrator` — a VRAM-budget-driven admission-control and swap-on-acquire model registry, built specifically to fit this whole pipeline through a single 12GB GPU on dev hardware by loading and unloading each model just-in-time.
