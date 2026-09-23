# anchors

`app/core/anchors/`

Every chunk records where it sits in its source, which is what a citation points at: a recording's **time range**, a PDF's **page range**, or a DOCX's **section path** (`AnchorKind.TIME / PAGE / SECTION`).

## Storage

`vector_chunks` carries `anchor_kind`, `start_sec`/`end_sec`, `page_start`/`page_end`, `section_path` and `position`. Document chunks store `start_sec = end_sec = 0.0`; `anchor_kind` says which fields mean something. `research_report_evidence` stores the same anchor fields so a signed-off report still renders its references after a re-ingest.

**`position`** is the chunk's reading order within its source (the index it was written at). Neighbour lookup (`_NEIGHBORS_SQL`), reading order (`order_for_reading`) and neighbour sorting use it, never `start_sec`, which is meaningless for documents. In-memory chunks without a position (cached eval candidates) fall back to time order.

The columns arrive on an existing database through `VECTOR_CHUNKS_UPGRADE_DDL` / `RESEARCH_REPORT_UPGRADE_DDL`, run by `OnyxEngine.start()`: rows from an earlier build become `time` anchors and get `position` backfilled in `start_sec` order. No re-ingest.

## Formatting

`utils/formatting.format_anchor` / `format_location` are the one place a location becomes text: `00:12:31–00:14:02`, `p. 7`, `pp. 7–8`, `str. 3` (sl/sr, from the pack's `documents.page_label`/`pages_label`), and for a page anchor `p. 7 · Article 7 Termination`. Citation headers, the reasoner's evidence tags, the report's evidence bank and references table, the CLI and the desktop gateway all use it, so a document chunk never shows as `00:00:00`.

**Recordings-only behaviour is unchanged.** Prompts that describe excerpts (the QA answer prompt, the reasoner, report scoping, outline and section writing) have a document variant that is used only when the context, evidence pool, bank or corpus actually holds a document excerpt (`has_document_chunks`, `qa_system_prompt(has_documents)`, …); the token budget is computed for the same variant that is sent. With recordings alone every prompt, the report's references header ("Timespan") and the CLI output are exactly what they were before documents existed — the prompt files and language packs only gained lines.

A document header costs `DOCUMENT_HEADER_TOKENS` (48) in the context budget, a transcript header still `DOC_HEADER_TOKENS` (24) — transcript packing is unchanged. `strip_repeated_overlap` only runs between two TIME chunks: document chunks never overlap.
