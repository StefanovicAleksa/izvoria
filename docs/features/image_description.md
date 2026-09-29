# image_description

`app/features/image_description/`

## What it does

Makes a standalone image (a photographed whiteboard, a scan saved as JPG, a chart, an X-ray, a site photo) citable. An image has no words of its own, so its excerpt is written by machines. This feature writes it so that every part says who wrote it, and text the image doesn't actually hold never reaches the index:

```
Image (machine description): A bar chart with quarterly revenue in EUR k, with a dip indicated by a red arrow pointing to the Q3 bar. Marked: Red arrow pointing to the Q3 bar.
Text on the image (read by OCR):
Quarterly revenue (EUR k)
...
```

The chunk is anchored `AnchorKind.IMAGE` and cited as `[Doc 4] (image)`. Its file path is the original image.

## How it does it

`ImageDescriptionAPI.execute(source_id, vlm_model)` is idempotent per source (`image_descriptions` table). It never loads a model: `ImageDescriptionPhase` holds the VLM for the whole batch and passes it in.

1. **Metadata.** PIL opens the file; this also rejects a non-image before the VLM is asked about it (`ImageReadError`). It records width, height and the EXIF date taken (`utils/exif.py`, DateTimeOriginal then DateTime). GPS is never read.
2. **Vision.** `VisualAnalysisAPI.describe_image` runs Qwen2-VL with the still-image prompt (see [visual_analysis](visual_analysis.md)). It returns SHOWS (one or two sentences), TEXT (up to 20 quoted items) and MARKS, and has no speech or LINK. `ground_still_image` keeps a photo with no text (its description is the evidence) but removes websites, interface text and repeats.
3. **OCR.** `DocumentConversionAPI.read_image` runs RapidOCR on the CPU. The image is first turned upright by its EXIF orientation (a phone photo is stored sideways, and OCR of the stored pixels read "1 / 2 60" off a forest path). The long side is capped at `IMAGE_OCR_MAX_SIDE` (4000), and a multi-page TIFF uses its first page. It returns `None` on an appliance without OCR (Serbian) or with its models missing.
   - **The reverse check.** OCR never invents a line, but on handwriting it writes gibberish ("Analbeig aftea long apok"), which used to be indexed as the image's text. `read_image` takes a `select_lines` hook, and this feature passes `select_ocr_lines`. An OCR line under `IMAGE_OCR_MIN_LINE_CONFIDENCE` (0.7) is kept only if the vision model read the same words. On an image where at least half the lines (and at least 5) are under that confidence, i.e. handwriting, every line under `IMAGE_OCR_UNRELIABLE_LINE_CONFIDENCE` (0.9) needs that confirmation. Lines are withheld before layout, so a table keeps its shape without them. Withheld lines are stored in `ocr_withheld_lines` (never indexed) and counted in the ingestion report. When one of them held letters or digits, the excerpt says "Some text on this image could not be read reliably and is not included" (pack `images.withheld_text_note`). Measured on real images: an 1839 letter's 26 gibberish lines are all withheld; sign lettering at 0.65–0.76 is kept because the vision model read it too; a timetable's misreads ("585", "888") go, and its headers at 0.7–0.8 ("Hartf d") stay.
4. **The check** (`utils/text_verification.py`). The two readers fail differently: a VLM reads stylised text OCR can't, but also writes plausible text that isn't there, while OCR misreads characters but never invents a line.
   - A VLM text item is **kept only if OCR read it too**: word for word, or ≥ `IMAGE_TEXT_MIN_OCR_COVERAGE` (0.8) of its words **including every word with a digit**. A one-word item needs that exact word. Comparison is NFKC, case-folded, in the pack's script (Serbian Cyrillic matches Latin OCR).
   - Items OCR didn't read are `dropped_items`: never indexed, counted in the ingestion report, and listed per image in its `errors`.
   - Quoted spans and multi-digit numbers in SHOWS/MARKS that OCR didn't read are replaced by `[text not confirmed]` / `[number not confirmed]`, not deleted, so a reader sees that something was there.
   - If OCR ran and read nothing, every VLM text item is dropped.
   - **Without OCR** (Serbian), nothing can be checked. The VLM's items are kept as `unverified_items` under the pack's "not verified" label, and the report counts the image in `images_text_unverified`.
5. **Excerpt** (`utils/excerpt_renderer.py`). The description block, then OCR's full text (the verbatim reading), or else the unverified items. Labels come from the pack's `[images]` table. The description itself is English on every appliance: Qwen2-VL is weak in Serbian, and text read off the image stays in its own language.
6. `render_context` gives the retrieval context: `file name · date taken · W×H`. It contains no words, so there is no language to get wrong.

`EmptyImageError` is raised when neither reader found anything; the image is reported and not indexed. A description with no SHOWS but OCR text (a scan the VLM couldn't structure) is indexed.

## Inputs / Outputs

Input: a `SourceType.IMAGE` source id and the loaded VLM. Output: `ImageDescription` (`shows`, `marks`, `ocr_text`, `ocr_lines_detected/read`, `ocr_checked`, `verified_items`, `unverified_items`, `dropped_items`, `taken_at`, `width`, `height`). `render()` and `render_context()` give the chunk's `original_text` and `context_string`.

## Where it fits

`BatchImagePipeline`: scan → ingest (`SourceType.IMAGE`; the VLM preloads meanwhile) → `ImageDescriptionPhase` (per image: describe → OCR → check → `ContextualizerAPI.execute_image`, which saves a one-chunk transcript with no model) → context_unification (`[Image Context: …]`) → rag_storage. It uses one VLM load for the whole batch, then the embedder. There is no transcript stage, contextual SLM or clip extraction. `OnyxEngine.ingest_images()`, CLI option 10, and the desktop's "Images" mode.

Formats: `.jpg .jpeg .png .webp .tif .tiff .bmp`. HEIC is not included because it needs pillow-heif, a native dependency.

## Gotchas

- **A document scanned to JPG/TIFF is an image here, by decision.** The document pipeline accepts no image files, so such a scan is indexed as an image: a machine description plus its full OCR text, cited `(image)`, without clause structure or page anchors. Scanned **PDFs** only ever go through the document pipeline. Save scans as PDF to have them treated as documents.

- **The excerpt is machine-written.** The answer, reasoner and report prompts are told so: an `(image)` excerpt is "a machine-written description of a picture", reported as what the image shows and never as the client's words. The desktop shows a thumbnail beside it so a reader can check it against the picture.
- **Serbian images are unverified.** There is no OCR for Serbian (see document_conversion), so the VLM's text stands alone, labelled.
- **OCR's reading order on charts is rough** (axis values come out in detection order). It only ever reads what is printed, which is the property the check relies on.
