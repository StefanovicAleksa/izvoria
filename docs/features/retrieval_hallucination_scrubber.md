# retrieval_hallucination_scrubber

`app/features/retrieval_hallucination_scrubber/`

## What it does

The core of "Verified Search" — an SLM-based relevance judge that filters out retrieved chunks that are irrelevant to the user's actual query or likely to produce a hallucinated/off-topic answer, before anything reaches the answer-generation LLM. This is a direct enforcement mechanism for the project's zero-hallucination requirement.

## How it does it

Exposes an `IRetrievalScrubberAdapter.evaluate_relevance` contract, implemented against a small local model (`RETRIEVAL_SCRUBBER` model type in `app/core/config.py`) that zero-shot-judges each retrieved chunk against the user's query text and drops chunks it judges irrelevant or unreliable. The public entrypoint (`RetrievalScrubberAPI.scrub_retrieved_chunks`, wired via `retrieval_scrubber_phase.py`) takes the raw hybrid-search results from `rag_retrieval` and returns a filtered, higher-precision subset. If the SLM's verdict for a chunk can't be parsed (malformed or non-object JSON), that chunk is **kept** and its `ScrubAudit` records `evaluation_failed=True` — consistent with the prompt's "when in doubt, accept" rule — instead of the whole query failing. Model and OOM errors still abort the scrub.

**The reranker's confident picks skip it.** A chunk the cross-encoder scored at or above `RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE` (0.5) is kept without asking the SLM; its audit says `kept_by_rerank_score=True`. The 7B judge threw away answering evidence the reranker had ranked first, for reasons like "the sign says 'STOP' and not 'GIMAr MOntaz MAutino'" or "does not provide the specific year". Measured 2026-09-28 on 62 answerable and 21 unanswerable questions (the en/sl/sr benchmark lectures, a book, images): the best candidate for an unanswerable question never scored above 0.01, while the relevant chunk of an answerable one never scored below 0.29. `None` makes the SLM judge every chunk again. The model is still acquired and released exactly as before.

**The prompt asks what a chunk is about, not whether it answers.** It names the excerpt kinds (transcript, document text, image description with OCR text), and says the answer's correctness is not the scrubber's concern. A `chunk_subject` field comes before the verdict: without it, the book and image questions lost their evidence again. Two variants were measured and rejected. One quoted the closest passage first; that put quotes and Cyrillic into the JSON, gave 45 unparseable verdicts on Serbian, and dropped book and image evidence. The other had no preliminary field and lost the image evidence.

| (relevant chunk survives / kept for unanswerable) | en | sl | sr | book + images |
|---|---|---|---|---|
| previous prompt | 10/10 · 1 | 9/10 · 0 | 32/32 · 0 | 7/10 · 0 |
| current prompt + rerank trust | 10/10 · 1 | 9/10 · 0 | 32/32 · 0 | 9/10 · 0 |

The remaining misses aren't the scrubber's: one answering chunk never reaches the reranker's top 10, and one scores 0.29, under the trust floor. Re-measure with `scripts/eval/rag_eval.py --stage retrieval --labels <file> --cache <file>` (the cache reruns the scrubber alone).

## Inputs / Outputs

Input: the user's query text + `List[RetrievedChunk]` from `rag_retrieval`. Output: a filtered `List[RetrievedChunk]` — only chunks judged relevant survive.

## Where it fits

Runs after `rag_retrieval` → `rag_reranking` in `LiveVerifiedRetrievalPipeline` (exposed as `IzvoriaEngine.search_verified()`) and in `LiveQAPipeline` (before `qa_generation`, exposed as `IzvoriaEngine.chat()`). Combined with `rag_retrieval`, this is what "verified search" means in this codebase — there is no separate `verified_search` feature directory; it's this scrubber wired in sequence after standard retrieval.

## Gotchas / design notes

Any agent built on top of RAG that wants hallucination-filtered context (rather than raw hybrid-search results) should route retrieved chunks through this feature before generation — it's the enforcement point, not `qa_generation` itself.
