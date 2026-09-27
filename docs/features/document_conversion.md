# document_conversion

`app/features/document_conversion/`

## What it does

Turns an ingested document into page-tagged Markdown (`ConvertedDocument`): one `PageContent` per PDF page, each saying how its text was obtained (`text_layer`, `ocr`, or `none`), or one unpaged `PageContent` for everything else — DOCX, and books and texts (TXT, Markdown, HTML, RTF, EPUB). This is the document pipeline's equivalent of transcription — the text every later stage indexes and every citation quotes.

## How it does it

`DocumentConversionAPI.execute(source_id)` resolves the source's vault file and is idempotent per source (`converted_documents` table, JSONB pages).

**PDF, one page at a time.** `PdfiumReader` (pypdfium2) splits the PDF into single-page PDFs and `MarkItDownConverter` converts each. Whole-PDF conversion is not an option: MarkItDown 0.1.8 joins pages and drops empty ones (`converters/_pdf_converter.py`), so page numbers and scanned pages would both be lost.

**Scanned or not.** A page goes to OCR only when **both** hold: its text layer has fewer than `DOC_MIN_TEXT_LETTERS_PER_PAGE` letters (40), and images cover at least `DOC_SCANNED_IMAGE_COVERAGE` of it (0.5, `IPdfDocument.image_coverage`). The image test matters: a born-digital page holding just a heading and a small table has few letters, and a letters-only rule OCR'd it and replaced "Monthly rent" / "2026" with "Monthty rent" / "026".

**OCR.** `RapidOcrEngine` (RapidOCR 3.9.2, PP-OCRv5 on onnxruntime, CPU — no VRAM, no model manager) renders the page at `DOC_OCR_DPI` (300) and reads it. Two RapidOCR behaviours shape the adapter:
- It downloads missing models from modelscope.cn at runtime. The adapter always passes explicit `Det/Cls/Rec.model_path`s into `DOC_OCR_MODEL_DIR` and raises `OcrModelsMissingError` when a file is absent; `scripts/download_ocr_models.py` fetches them once at build time, SHA256-checked against rapidocr's own `default_models.yaml`.
- It silently drops every line whose recognition is empty (`rapidocr/main.py`). A detection-only pass counts lines first; `ocr_lines_read < ocr_lines_detected` marks the page `ocr_incomplete`, and the ingestion report lists it.
- Its `use_det/use_cls/use_rec` flags are sticky between calls, so both calls name all three.

Lines are put in reading order (rows top to bottom, left to right; a gap over 1.6 line heights becomes a paragraph break). `Global.text_score` is 0: low-confidence lines are kept and the mean confidence recorded, not silently discarded.

**Which language gets OCR** is the pack's `documents.ocr_rec_model`. English and Slovenian use `latin` (its alphabet has č š ž, and ć đ). Serbian has none: the `cyrillic` model lacks č ć đ š ž, `latin` lacks Cyrillic, `eslav` lacks ђ ј љ њ ћ џ, and on a mixed-script letter RapidOCR lost two of six lines. A scanned page on a Serbian appliance is `method=none`, reported, never guessed. A missing-models deployment error is reported the same way, per page, instead of failing the document.

**Tables** (`utils/layout.py`, `data/pdfplumber_layout.py`). MarkItDown reads a borderless table column by column — "Parking place", "Cleaning service", "2", "4", "EUR 40.00", ... — so no value stays with its row, and neither the contextual SLM nor retrieval can answer "what did parking cost". Each text-layer page therefore also goes through `PdfplumberLayoutExtractor`; if it finds a table, its text replaces MarkItDown's for that page (pages without tables keep MarkItDown's text):
- **Ruled tables:** pdfplumber's line-based `find_tables` gives exact cells, multi-line cells included. Its text strategy is not used: on prose it cuts sentences into "cells".
- **Borderless tables:** from word positions. Consecutive lines that leave the same vertical strips empty (**gutters**) — at least `_MIN_GUTTER` (0.35) line heights wide and at least `MIN_GUTTER_TO_WORD_GAP` (1.5×) the gaps between words inside the cells — become a table, if there are ≥ 3 rows, 2–8 columns, every row has ≥ 2 filled cells, and cells average ≤ 6 words. Shared gutters rather than a per-gap threshold, because word spacing (especially OCR's) jitters too much for one gap width to separate cells from words, while prose never keeps its word gaps in the same place line after line. Any column alignment works (right-aligned amounts included). A heading line or a wrapped cell has one cell, so it ends the table.
- Words are extracted with `x_tolerance_ratio` (relative to character size): pdfplumber's fixed 3-point default glues neighbouring cells together when the font is small.
- **Scans** use the same code: RapidOCR is asked for word boxes (`return_word_box=True`), because its detector often boxes two neighbouring cells as one line.

Tables reach `document_structuring` as Markdown table blocks, which are never split mid-row (a long one splits into row groups that each repeat the header), and the document context prompt tells the SLM that a row's cells belong together.

Measured on generated invoices: every row of a bordered table, a borderless 4-column line-item table and a 2-column payment table exact, as text layer and as a 300 DPI scan; justified prose, numbered lists and signature lines produce no table. With unusually tight gutters (~0.4 em) in a scan, the header row can fall out of the table onto its own line; the rows keep their values.

**Text cleanup** (`utils/text.normalize_extracted_text`) changes no wording: ligatures (`ﬁ` → `fi`, which otherwise defeats keyword search), tabs (pdfminer separates positioned words with them), odd spaces, soft hyphens, and runs of blank lines.

**DOCX** converts whole (one `PageContent(page_no=None)`); Word's headings arrive as Markdown `#` headings.

**Books and texts.** `DOCUMENT_KINDS` in `service/conversion_api.py` is the one list of accepted suffixes; the pipeline's filter is built from it.
- **TXT and Markdown** (`.txt`, `.md`, `.markdown`) do **not** go through MarkItDown. It runs `re.sub(r"\n{3,}", "\n\n")` on every result (`markitdown/_markitdown.py`), and `normalize_extracted_text` also collapses blank lines and strips the text; either would shift every line number these files are cited by. `CharsetTextDecoder` decodes them instead: UTF-8 (BOM dropped), otherwise charset-normalizer's best guess, so cp1250 Slovenian and cp1251 Serbian files read correctly. A file in no recognisable encoding raises `UndecodableTextError` rather than indexing mojibake. `normalize_text_lines` then does the same in-line cleanup as for PDFs but keeps every line, so line N of the stored text is line N of the file.
- **EPUB and HTML** go through MarkItDown's own converters: EPUB is a zip of XHTML read with `zipfile` and `defusedxml` in spine order, with the title/author metadata as a first paragraph; `<h1>`… become `#` headings, so chapters become sections. `<script>` is dropped. Nothing is fetched.
- **RTF** is read by `StripRtfReader` (striprtf, pure Python, no dependencies). The bytes are read as Latin-1 and the file's own `\ansicpg` code page decides what its escapes mean; a character that cannot be decoded shows as U+FFFD rather than vanishing.

**Air gap.** MarkItDown reaches the network only through `convert(<url string>)` and its audio converter's `recognize_google()`. The adapter calls only `convert_stream(bytes)` with `enable_plugins=False`, and only `markitdown[pdf,docx]` is installed (`[all]` adds Azure, YouTube and speech-recognition clients). The EPUB and HTML converters are part of MarkItDown's core and local.

## Inputs / Outputs

Input: a `SourceType.DOCUMENT` source id. Output: `ConvertedDocument` with per-page `text`, `method`, `ocr_confidence`, `ocr_lines_detected/read`, `note`; properties `pages_ocr`, `pages_ocr_incomplete`, `pages_unreadable`.

## Where it fits

`DocumentConversionPhase` in `BatchDocumentPipeline`, right after ingest. The contextual SLM preloads in the background while it runs.

## Gotchas

- `rapidocr` is installed with `--no-deps` from `requirements-nodeps.txt`: it declares `opencv-python`, whose `cv2` would overwrite `opencv-python-headless`'s. `pip check` reports that one missing requirement on purpose.
- Accuracy on real scans is unmeasured. On synthetic 300 DPI scans English read at 0.0% CER and Latin-script text lost only đ/ć (read as d/c).
