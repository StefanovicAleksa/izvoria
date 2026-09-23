# document_structuring

`app/features/document_structuring/`

## What it does

Cuts a `ConvertedDocument` into retrieval chunks along the document's own structure, each knowing its pages and the headings it sits under (`StructuredChunk.section_path`, e.g. `LEASE AGREEMENT › Article 7 Termination`). Pure and fast: no model, no storage — `DocumentContextualizationPhase` calls it inside the contextualization job, so it has no phase or table of its own.

## How it does it

1. **Blocks** (`utils/blocks.py`). Each page splits at blank lines; Markdown tables (`|` rows) are blocks of their own; a heading line inside a paragraph splits it (OCR output rarely has a blank line between a heading and its body). Hard-wrapped lines join with a space.
2. **Page furniture** is removed, but only from a page's top and bottom two lines: a bare number within 2 of the real page number, and short non-sentence lines repeated at the edges of at least half the pages (≥ 3). Both limits exist because an earlier version deleted "2025" and "2026" from a rent table as page numbers, and would have deleted any sentence repeated at the top of short pages.
3. **Headings** (`utils/headings.py`): Markdown `#` (DOCX), the pack's `documents.clause_patterns` ("Article 7", "3. člen", "Član 5"), short numbered titles ("4.2 Late Payment", depth from the numbering), or a short ALL-CAPS line (level 1). A line of more than 12 words, or one that ends a sentence, is body text.
4. **Chunks** (`utils/chunker.py`): a title or clause heading starts a new chunk; sub-clauses share one until `DOC_CHUNK_MAX_WORDS` (250). Long paragraphs split at sentence ends; long tables split into row groups that each repeat the header row. A document under `DOC_SHORT_DOCUMENT_WORDS` (400) — a receipt, a one-page letter — is one chunk. No overlap between chunks. Trailing headings with no body join the chunk before.

On a Serbian appliance every block goes through `normalize_script`, the same Cyrillic → Latin as transcripts.

## Inputs / Outputs

Input: `ConvertedDocument`. Output: `StructuredDocument(anchor_kind=PAGE for PDF, SECTION for DOCX, chunks)`. `EmptyDocumentError` when no page has text.

## Gotchas

- Heading detection is lexical. A contract whose headings are only bold formatting (no numbering, no clause word, mixed case) yields chunks cut by the word budget instead of by section.
