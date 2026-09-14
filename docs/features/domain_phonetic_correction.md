# domain_phonetic_correction

`app/features/domain_phonetic_correction/`

## What it does

An LLM-based post-correction pass that fixes domain-specific terminology and phonetic mis-transcriptions (e.g. Whisper mishearing a specialized legal or medical term) against a controlled glossary, applied per-sentence to already-reconstructed sentences, with a full audit trail of every change made.

## How it does it

`DomainPhoneticCorrectionAPI.execute` takes a `ReconstructedTranscript`, sanitizes text via `PhoneticTextSanitizer`, splits oversized sentences at whitespace (`split_massive_sentences`, 1200-char limit, dividing the acoustic word array in step with the text), then batches sentences into LLM-sized chunks using `sliding_window_chunker` (a generic char-budgeted sliding window with `overlap_items=2` for cross-boundary context).

`DictionaryLoader.build_prompt_dictionary` loads industry + sub-domain JSON glossary terms into a bulleted string. If it holds fewer than `PHONETIC_MIN_DICTIONARY_TERMS` (1) terms — i.e. the glossary is empty — the LLM pass is skipped and the sanitized sentences are persisted as a passthrough — there is nothing to correct against. Otherwise, for each chunk, and `QwenDomainAdapter` (Qwen via `llama_cpp`, `flash_attn=True`) is prompted to emit `<patch><orig>...</orig><corr>...</corr></patch>` XML tags. These are parsed by a regex "firewall" (`xml_parser.parse_xml_patches`) that tolerates hallucinated XML attributes but rejects malformed/nested tags. A parsed patch is only kept if its correction introduces a dictionary term (case-insensitively present in the correction but not already present, in that exact spelling, in the original) — the model may not rewrite ordinary words. Kept patches are applied via `AnchorPatchApplicator.apply`, which diffs word-by-word common prefix/suffix between the original and correction (case-sensitive, to preserve deliberate capitalization changes like `ebitda`→`EBITDA`) and does a case-insensitive regex replace that reconstructs the exact original spacing/case around the changed words. Every applied patch is recorded in an `AppliedDomainCorrectionRecord`.

## Inputs / Outputs

Input: `ReconstructedTranscript`, `industry: str`, `sub_domains: List[str]`. Output: `DomainPhoneticCorrectedTranscript` (`sentences: List[ReconstructedSentence]`, `audit_trail: List[AppliedDomainCorrectionRecord]`).

## Where it fits

Phase 8 (`DomainPhoneticCorrectionPhase`) — runs after sentence reconstruction, before contextual retrieval (chunking).

## Gotchas / design notes

- If the LLM hits `max_tokens` (800) mid-generation, truncated tail patches are dropped/salvaged rather than erroring.
- Identical original==correction patches are silently skipped and logged only at debug level, to avoid noisy logs for no-op corrections.
