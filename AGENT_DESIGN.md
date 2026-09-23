# Design: multi-pass "deep research" inference layer for chat (revision 2)

> **Status (2026-09-21):** this document describes the mode now called **Reasoned Answer**
> (`OnyxEngine.chat_reasoned`, `LiveReasoningPipeline`, `SessionTurnMode.REASONING`). It was
> originally shipped as "Deep Research"; the name moved to the separate multi-page report mode
> (`app/features/deep_research_report/`), which is a different pipeline entirely. Names below are
> pre-rename — the design they describe is unchanged.

Status: implemented (commit `8052035`), then revised by `75f333c` and `7e9013c` — see `CHANGELOG.md` for what changed after this design was written.

## Context

This revises the first-pass design after feedback on two points:

1. The original front-door "planner" leaned on Qwen2.5-3B to emit a bare 3-way
   routing decision (refine / answer / clarify) with almost no reasoning behind
   the choice. That's too weak — it should behave like a real reasoning step:
   chain-of-thought over the actual retrieved evidence, with claims tied to
   specific evidence chunks even at the intermediate stage, using the strongest
   local model available for that role.
2. There was no step after Gemma produces a draft answer to check whether that
   draft is actually fully grounded and complete before showing it to the user.

On "rivaling hosted frontier models": worth stating plainly. On this hardware
(12GB dev GPU, 24GB target split across two GPUs, single-slot VRAM pool — see
`ModelManagerOrchestrator`), we cannot match a hosted model's parameter count.
The largest local text-reasoning model already on disk is **Qwen2.5-7B-Instruct**
(today only used for ASR/visual-cue detection). What we *can* get closer to
parity on is behavior: genuine multi-step reasoning over real evidence, explicit
per-claim grounding, and iteration — the actual lever hosted "deep research"
products use is test-time iteration, not just base model size, and that lever is
fully available to us.

Decisions carried over from revision 1 (unchanged):
- Conversation/session state is **Postgres-persisted** (existing single DB) —
  this is memory *across* separate questions/turns, not the intra-question loop
  state described below (see "Two different kinds of memory" below — keeping
  these distinct matters).
- This ships as a **new, opt-in entrypoint** alongside today's single-pass
  `chat()` — nothing about `LiveQAPipeline`, `QA_SYSTEM_PROMPT`, or the existing
  retrieval/rerank/scrub APIs changes.

New decisions from this revision:
- The front-door reasoning role is upgraded from Qwen2.5-3B to **Qwen2.5-7B-
  Instruct** (already on disk, its own `ModelType` so it doesn't couple to the
  other features already using that weights file), doing real reasoning + an
  evidence-linked interim analysis, not just a label.
- A new **post-generation self-critique step** runs after Gemma drafts an
  answer: **Gemma critiques its own draft, in the same model acquisition** (no
  extra GPU swap — draft, then immediately ask it a second, differently-framed
  question on the same loaded model, then release once). If Gemma flags its own
  draft as under-supported, the loop goes back for one more evidence pass rather
  than showing an ungrounded answer. Upgrade path if this proves too weak in
  practice: swap in an independent smaller judge (Qwen2.5-3B, same JSON-verdict
  pattern already used by the scrubber) without changing the pipeline shape —
  just which adapter the critique call goes through.

## Two different kinds of memory — don't conflate them

- **Intra-question iteration state** (new, ephemeral, lives only for the
  duration of one `chat_deep_research()` call): the accumulated evidence pool,
  the reasoning/decision history, the iteration counter. Gone once the call
  returns.
- **Cross-question session state** (new, Postgres-persisted): what the user
  asked last turn and what was answered, so a follow-up question ("what about
  the renewal?") has context. This feeds *into* the reasoner's first iteration
  of the *next* question as prior context, but is otherwise a separate concern
  from the loop below.

## The pipeline

```
                          ┌───────────────────────────────────────────────────┐
                          │              LiveDeepResearchPipeline                │
                          │                                                     │
  user question       ──▶│  ReasonerPhase  (Qwen2.5-7B, chain-of-thought +      │
  + prior session         │       │          evidence-linked interim analysis,  │
    context (if any)      │       │          ~500-1000 token budget)            │
                          │       │                                            │
                          │       ├─ ANSWER_READY ──────────────────────┐       │
                          │       │                                     │       │
                          │       ├─ REFINE_QUERY(q') ─┐                 │       │
                          │       │                    ▼                 │       │
                          │       │        RagRetrievalPhase             │       │
                          │       │        RagRerankingPhase             │       │
                          │       │        RetrievalScrubberPhase        │       │
                          │       │             │                        │       │
                          │       │      merge survivors into            │       │
                          │       │      evidence_pool (dedup by         │       │
                          │       │      chunk_id), loop back ───────────┘       │
                          │       │      to ReasonerPhase                        │
                          │       │      (bounded by max_iterations)             │
                          │       │                                              │
                          │       └─ ASK_CLARIFICATION(text) ──▶ return early,   │
                          │                          no generation, no critique  │
                          │                                       ▼              │
                          │                          QAGenerationPhase           │
                          │                    (Gemma: draft, THEN self-critique,│
                          │                     same acquisition, one release)   │
                          │                                       │              │
                          │                    ┌── SUFFICIENT ────┘              │
                          │                    │                                 │
                          │                    └── NEEDS_MORE_EVIDENCE(gap) ─┐   │
                          │                                                  │   │
                          │            loop back to ReasonerPhase with gap ◀─┘   │
                          │            as extra context (bounded by             │
                          │            max_critique_retries, separate small cap) │
                          └───────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                                  QAResponse + full iteration/critique trace
```

## The loop, step by step

State carried for one `chat_deep_research()` call only (see "two kinds of
memory" above):
- `original_question` — fixed.
- `current_query` — what actually gets searched this iteration (starts as the
  question, then the Reasoner can rewrite it).
- `evidence_pool` — every `RetrievedChunk` that survived scrubbing so far,
  deduped by `chunk_id`, growing across iterations.
- `decision_history` / `iteration_trace` — a running log of what was searched,
  what was kept, and why the Reasoner/critic made each call. This becomes both
  (a) context fed back into the Reasoner so it doesn't repeat a query it already
  tried, and (b) the audit trail returned to the caller — this product's whole
  premise is traceability, so the multi-pass process must stay inspectable, not
  just its final citations.
- `iteration_count` / `critique_retry_count` — two small counters, each capped.

**One pass through the loop:**

1. **Reasoner turn.** Acquire Qwen2.5-7B (`wait_for_clear=True`, evicts whatever
   was resident). Feed it: the original question, the actual text of the current
   `evidence_pool` (not just a summary — a 7B model at `n_ctx≈8192` has room for
   this, unlike the smaller 3B router in revision 1), the decision history, and
   — on the very first iteration of a new question — the prior turn's Q&A from
   the persisted session. Ask for structured JSON output containing: a short
   chain-of-thought (kept in the trace, not shown raw to the user), an interim
   analysis tying specific claims to specific `chunk_id`s where evidence already
   supports them, and exactly one decision: `ANSWER_READY`, `REFINE_QUERY` (with
   a rewritten/narrower query string), or `ASK_CLARIFICATION` (with the question
   to ask back). Release Qwen2.5-7B in `finally` before anything else runs — this
   is the step that must happen or the next acquire deadlocks (single-slot pool).

2. **Branch:**
   - `ASK_CLARIFICATION` → stop, return the clarifying question — no Gemma call,
     since there's nothing ready to ground yet.
   - `ANSWER_READY` → go to step 4.
   - `REFINE_QUERY` → go to step 3.

3. **Retrieval turn.** Exactly today's existing single-pass chain, run once with
   `current_query`: `RagRetrievalPhase` → `RagRerankingPhase` →
   `RetrievalScrubberPhase` (all three untouched). Merge survivors into
   `evidence_pool` by `chunk_id`. Increment `iteration_count`; if it hits
   `max_iterations`, force a jump to step 4 with whatever's accumulated so far,
   even if the Reasoner would have kept going. Otherwise loop back to step 1.

4. **Generation + self-critique turn.** Acquire Gemma once
   (`wait_for_clear=True`). Call `generate_answer()` (existing, unchanged) over
   the full `evidence_pool` and the Reasoner's interim analysis as extra context
   — this is what lets the final answer synthesize across multiple sub-queries
   into one detailed, "deep research"-style overview instead of being limited to
   one query's top-k. **Without releasing the model**, immediately make a second
   call asking Gemma to check its own draft: does every claim actually match the
   text of the chunk it cites, and does the draft fully address the original
   question? This returns `SUFFICIENT` or `NEEDS_MORE_EVIDENCE(gap_description)`.
   Release Gemma once, in `finally`, covering both calls.

5. **Branch:**
   - `SUFFICIENT` → return the `QAResponse` plus the full trace.
   - `NEEDS_MORE_EVIDENCE` → if `critique_retry_count` is under its (separate,
     small) cap, feed `gap_description` back into the Reasoner as extra context
     and loop to step 1; otherwise return the draft anyway but flag in the trace
     that self-critique had unresolved concerns, rather than looping forever.

## Why the model-role split is what it is

| Role | Model | Why |
|---|---|---|
| Reasoning / deciding what's missing | Qwen2.5-7B-Instruct (new `ModelType`) | Largest local text model besides Gemma; reused weights file, independent `ModelType` so its own `n_ctx`/settings don't couple to ASR/visual-cue-detection's use of the same file. |
| Retrieval / rerank / relevance-scrub | BGE-M3 embedder, cross-encoder reranker, Qwen2.5-3B scrubber (all existing, unchanged) | Already built, already tuned, out of scope to touch. |
| Final answer + citations | Gemma-4-E4B (existing `QA_GENERATION`, unchanged adapter/prompt) | Already has a production-tuned strict-grounding system prompt; kept as the only model that ever writes user-facing prose. |
| Self-critique of the draft | Gemma again, same acquisition | Zero extra GPU swap (draft + critique share one acquire/release cycle, following the same "one acquire, many calls" pattern already used by `RetrievalScrubberAPI`/`DomainPhoneticCorrectionAPI`). Known weaker than an independent judge at catching subtle self-introduced errors, but the check here is narrow and mechanical (does claim X match cited text Y — closer to entailment-checking than open-ended judgment), so it's a reasonable place to start. If it proves insufficient, swap to an independent Qwen2.5-3B critic later — same JSON-verdict pattern as the scrubber, no pipeline-shape change needed. |

## Latency budget (be upfront about this)

Every swap between Reasoner / retrieval-stack / Gemma is a real GGUF/weights
reload on this hardware. One full loop iteration costs: 1 Reasoner swap + 3
retrieval-stack swaps = 4 swaps, before generation. Generation+self-critique adds
1 more swap (shared between both calls). A 2-iteration "deep research" answer is
therefore roughly 9 model swaps versus today's 3-swap single-pass `chat()`. At an
estimated 1-3s per swap this is a real, user-visible latency cost — the reason
`max_iterations` and `max_critique_retries` should both default small (e.g. 2-3
and 1, respectively) rather than open-ended.

## Where this plugs in

- New feature `app/features/qa_agent/` (or similar name, peer to `rag_retrieval`/
  `qa_generation`) housing the Reasoner: `domain/interfaces.py`
  (`IReasonerAdapter(IModelLoader)` with a `reason(model, question, evidence_pool,
  decision_history, prior_session_context) -> ReasonerDecision` method),
  `data/qwen_reasoner_adapter.py` (llama-cpp-python, JSON-structured output,
  larger `n_ctx`/`max_tokens` than the scrubber's), `service/reasoning_api.py`.
- `app/features/qa_generation/` grows one new adapter method,
  `critique_answer(model, user_query, draft_answer, evidence_block) ->
  AnswerCritique`, and `QAGenerationAPI` grows `generate_with_critique(...) ->
  Tuple[QAResponse, AnswerCritique]` that acquires Gemma once, calls
  `generate_answer` then `critique_answer` on the same `LoadedModel` handle, and
  releases once — the existing `generate()` method and `LiveQAPipeline` are
  untouched, this is purely additive.
- New `ModelType.QA_REASONER` + `QA_REASONER_MODEL_PATH`/
  `QA_REASONER_VRAM_REQUIREMENT_MB` in `app/core/config.py`, pointed at the
  existing `Qwen2.5-7B-Instruct-Q4_K_M.gguf` file.
- New `ReasonerPhase` (`app/orchestration/service/phases/`) following the
  existing `preload_model`/`discard_preload`/`execute()`-with-`trigger_unload`-
  in-`finally` shape every other Phase already uses.
- `QAGenerationPhase` gains a sibling method (e.g. `execute_with_critique(...)`)
  alongside its existing `execute(...)`, both backed by the same adapter/model
  lifecycle machinery — existing call sites (`LiveQAPipeline`) don't change.
- New `LiveDeepResearchPipeline` (`app/orchestration/service/pipelines/`) — the
  first pipeline in the codebase allowed to loop/branch instead of running
  linearly. Owns `ReasonerPhase` plus the existing `RagRetrievalPhase`/
  `RagRerankingPhase`/`RetrievalScrubberPhase`/`QAGenerationPhase` instances
  (reused, not duplicated).
- New `OnyxEngine.chat_deep_research(session_id, user_query)`-shaped method,
  guarded by the same `_enforce_online_state()`/`EngineBusyError` rule as
  `chat()`, wired into `OnyxBootstrapper.build()` alongside (not replacing) the
  existing `live_qa_pipeline`.
- Postgres-persisted session table + repository (new, follows
  `IRetrievalRepository`'s shape) for the cross-question memory described above
  — separate concern from the intra-question loop state.

## Verification (once this moves to implementation)

- Unit tests mirroring the new feature dir, fully mocked, following existing
  conventions (`RetrievalScrubberAPI`'s tests are the closest template for a
  service that acquires once and loops multiple model calls).
- An integration test running the full loop against real `db_test` Postgres,
  self-skipping with a descriptive reason if the Qwen2.5-7B weights aren't
  present on the machine (matching every other model-dependent integration
  test).
- Manual check via `app/main.py`'s menu: ask a multi-part question
  that needs two distinct retrieval passes, confirm the final answer's citations
  span both passes and the trace shows the Reasoner's intermediate decisions;
  ask a genuinely unanswerable question and confirm it still refuses rather than
  fabricating, even after exhausting `max_iterations`; and force a case where
  self-critique should catch an under-supported claim to sanity-check the
  critique step actually fires.
