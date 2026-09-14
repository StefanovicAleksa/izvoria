# visual_hallucination_correction

`app/features/visual_hallucination_correction/`

## What it does

Acts as a strict SLM gatekeeper that checks whether the VLM's per-frame visual descriptions actually match the corresponding transcript text, producing an immutable audit ledger that downstream synthesis relies on to decide which visuals to trust and which to discard as hallucinated/irrelevant.

## How it does it

`VisualHallucinationCorrectionAPI.execute(source_id)` pulls the `ContextualizedTranscript`, is idempotent via `get_by_source_id`, and for every chunk with `needs_visual=True` fetches the associated visual-analysis record, concatenates all frame descriptions into a numbered evidence block, truncates to `MAX_CHARS=10000` (appending `"...[TRUNCATED TO PREVENT OOM]"`) as a context-window safeguard, and calls `IVisualHallucinationAdapter.evaluate_visual_match()`. The audit prompt defaults to accepting a description. It rejects only two cases:

- the description is completely unrelated to the text;
- the description contains no visual content and only restates or paraphrases the transcript. The VLM sees the transcript, so a restatement proves nothing about what was on screen.

A description saying the discussed item is *not* visible counts as visual content and is accepted.

The concrete `QwenHallucinationAdapter` (Qwen 2.5 3B via `llama_cpp`, `n_ctx=4096`) deliberately avoids strict JSON-schema constraints — a comment in the code notes this was "removed... because it caused thread locks" on this small model — and instead requests `{"type": "json_object"}` at `temperature=0.0`, with a regex fallback extraction (`re.search(r'\{.*}', ...)`) before `json.loads`, trading strict guarantees for robustness against a smaller, less reliable model.

## Inputs / Outputs

Input: `source_id`. Output: `VisualHallucinationEvaluatedTranscript` (a list of `VisualHallucinationAudit` entries, each with `is_visual_accepted`/`discard_reason`).

## Where it fits

Phase 13 (`VisualHallucinationCorrectionPhase`) — runs after visual analysis, before visual context synthesis (which only synthesizes from accepted evidence).

## Gotchas / design notes

Chunks with no `needs_visual` flag or no visual-analysis records are silently skipped — not audited at all. This means **absence from the ledger implies "not accepted"** downstream, not "accepted by default" — worth knowing when debugging why a chunk has no visual context.
