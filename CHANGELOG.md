# Changelog

This log tracks **feature-level milestones**, not every commit. It's written for humans (and future Claude Code sessions) trying to understand how the system got to its current shape — one entry per merged feature milestone, newest on top. See `CLAUDE.md` for the architecture these entries plug into.

**Regression fixes name the commit that introduced the problem and the commit that fixed it** (`git show <hash>`), so a repeat of a mistake can be traced straight to its source.

## Chat citations resolve to their clips

- **Chat citations point to the same clips as search results** (`qa_generation/domain/entities.py`, `qa_generation/utils/citation_parser.py`, `engine/service/onyx_engine.py`, `app/main.py`).
  - **Introduced in:** `d25b030` (`parse_citations` wrote the placeholder "Unresolved Path") and `b8493d0` (`OnyxEngine._resolve_citation_paths`).
  - **Fixed in:** pending
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
