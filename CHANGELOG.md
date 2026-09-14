# Changelog

This log tracks **feature-level milestones**, not every commit. It's written for humans (and future Claude Code sessions) trying to understand how the system got to its current shape — one entry per merged feature milestone, newest on top. See `CLAUDE.md` for the architecture these entries plug into.

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

> **Action required:** the `vector_chunks` schema changed (new `raw_text`, `context_text`, `speaker_labels` columns). There are no migrations, so reset the dev database with `scripts/reset_dev_db.sh` and re-ingest. To enable reranking, download `BAAI/bge-reranker-v2-m3` to `models/bge-reranker-v2-m3`; without it, search falls back to RRF order.

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
