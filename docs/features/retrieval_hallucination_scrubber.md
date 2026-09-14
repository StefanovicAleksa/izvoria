# retrieval_hallucination_scrubber

`app/features/retrieval_hallucination_scrubber/`

## What it does

The core of "Verified Search" — an SLM-based relevance judge that filters out retrieved chunks that are irrelevant to the user's actual query or likely to produce a hallucinated/off-topic answer, before anything reaches the answer-generation LLM. This is a direct enforcement mechanism for the project's zero-hallucination requirement.

## How it does it

Exposes an `IRetrievalScrubberAdapter.evaluate_relevance` contract, implemented against a small local model (`RETRIEVAL_SCRUBBER` model type in `app/core/config.py`) that zero-shot-judges each retrieved chunk against the user's query text and drops chunks it judges irrelevant or unreliable. The public entrypoint (`RetrievalScrubberAPI.scrub_retrieved_chunks`, wired via `retrieval_scrubber_phase.py`) takes the raw hybrid-search results from `rag_retrieval` and returns a filtered, higher-precision subset.

## Inputs / Outputs

Input: the user's query text + `List[RetrievedChunk]` from `rag_retrieval`. Output: a filtered `List[RetrievedChunk]` — only chunks judged relevant survive.

## Where it fits

The second stage of `LiveVerifiedRetrievalPipeline` (after `rag_retrieval`, exposed as `OnyxEngine.search_verified()`) and the second stage of `LiveQAPipeline` (before `qa_generation`, exposed as `OnyxEngine.chat()`). Combined with `rag_retrieval`, this is what "verified search" means in this codebase — there is no separate `verified_search` feature directory; it's this scrubber wired in sequence after standard retrieval.

## Gotchas / design notes

Any agent built on top of RAG that wants hallucination-filtered context (rather than raw hybrid-search results) should route retrieved chunks through this feature before generation — it's the enforcement point, not `qa_generation` itself.
