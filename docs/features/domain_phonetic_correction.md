# domain_phonetic_correction

`app/features/domain_phonetic_correction/`

## What it does

Fixes genuine mis-hearings of glossary terms in already-reconstructed sentences. For example, Whisper writes "the burn" where the speaker said "the Bund", or "Mario Draggy" for "Mario Draghi". Every change is recorded in an audit trail.

It never rewrites natural speech. Articles, singular/plural, filler, number shorthand ("those 73s", "the 20s"), digits and correctly spelled terms stay exactly as transcribed.

## How it does it

`DomainPhoneticCorrectionAPI.execute` runs in four steps. The model never writes text: it can confirm or reject a candidate, and nothing else.

1. **Sanitize and load terms.**
   - `PhoneticTextSanitizer` cleans each sentence.
   - `DictionaryLoader.load_terms` loads the industry's base terms plus the chosen sub-domains' anchor words and terms.
   - If there are fewer than `PHONETIC_MIN_DICTIONARY_TERMS` terms, the sanitized sentences are saved as a passthrough.
2. **Find candidates deterministically** (`utils/term_candidate_finder.py`, `utils/phonetic_key.py`). `find_term_candidates` looks for spans of 1–5 tokens that sound like exactly one dictionary term.
   - **Score.** The mean of two similarities: letters-only `difflib` similarity, and similarity of a small metaphone-style sound key (voiced/voiceless pairs merged, vowels dropped).
     - The letters must also be at least 0.6 similar, and the score must reach `PHONETIC_CANDIDATE_MIN_SIMILARITY`.
     - The letter-length ratio must stay within `PHONETIC_LENGTH_RATIO_MIN/MAX`.
     - Spelled-out acronyms ("E C B", "tee el tee ar oh") are compared letter by letter.
   - **Never a candidate:**
     - a span with a digit;
     - a span that already is the term up to case, hyphens or plural ("bund", "E.C.B.", "deposit rates");
     - a span containing a correctly written single-word term ("the bund", "ECB meeting");
     - fewer tokens than the term has words, which would match part of a correct term ("Philip" of "Philip Lane");
     - an acronym matched to an ordinary word ("from" for FOMC);
     - a plural heard for a singular term;
     - an -ed/-ing verb heard for a noun term ("burned" → "Bund").
   - **Alignment.** Every heard token must supply, and every term word must receive, at least 60% of its letters in the letter alignment. This stops a span absorbing a neighbouring word, and stops a term replacing only part of what was heard ("depot sit rates" → "Deposit Rate rates").
   - **Whisper confidence.** Span tokens are mapped back to `sentence.words`. A span whose words were all heard with probability ≥ `PHONETIC_MAX_WORD_PROBABILITY` is skipped. Unknown probabilities don't block a span.
   - **Output.** The replacement keeps the heard number ("burn" → "Bund" for the term "Bunds"). Overlaps resolve by best score, then shortest span.
3. **Verify** (`QwenDomainAdapter.verify_candidates`).
   - Qwen 2.5 3B sees the previous sentence and two versions of the sentence, which differ only in the bracketed span: one as transcribed, one with the term. It answers "A" or "B" under a GBNF grammar.
   - A candidate is confirmed only if the term version wins **both** when shown first and when shown second. Either order alone let position bias accept ordinary words.
   - Long sentences are windowed to `CONTEXT_CHARS` on each side of the span.
   - The model is loaded only if at least one candidate exists.
4. **Apply** (`utils/span_replacer.py`).
   - `replace_spans` writes each confirmed replacement over its exact character span and copies everything else unchanged.
   - The sentence is left untouched if its digits would change.
   - Each change is recorded as an `AppliedDomainCorrectionRecord` with `DomainPhoneticPatch(original=heard span, correction=replacement)`.

## Inputs / Outputs

- **Input:** `ReconstructedTranscript`, `industry: str`, `sub_domains: List[str]`.
- **Output:** `DomainPhoneticCorrectedTranscript` (`sentences: List[ReconstructedSentence]`, `audit_trail: List[AppliedDomainCorrectionRecord]`).

## Where it fits

Phase 8 (`DomainPhoneticCorrectionPhase`): after sentence reconstruction, before contextual retrieval (chunking).

## Gotchas / design notes

**Why not free rewrites.** An earlier version asked the model to emit `<orig>/<corr>` rewrite patches. It paraphrased natural speech into written form:
- "very first" → "initial";
- "Bank of England" → "BOE";
- "13.50.45 seconds" → "13:50:45";
- "the bund" → "Bunds";
- "those 73s" → "73 Bunds".

The gate in front of it could not tell these apart from real fixes.

**Evaluation.**
- `scripts/eval/phonetic_eval.py` runs the real API over the stored transcript plus `scripts/eval/phonetic_labels.json`, with an in-memory repository.
- The genuine mis-hearings in the labels come from re-transcribing the video without hotwords (`scripts/eval/retranscribe_without_hotwords.py`).
- Results: see `CHANGELOG.md`.

**Limits.**
- Mis-hearings that neither look nor sound like the term ("front" or "boom" for "Bund") aren't found.
- Terms of sub-domains the router didn't select aren't in the dictionary.

**Other.**
- The Qwen model shares its weights file with visual hallucination correction, but has its own `ModelType`.
- `utils/sentence_splitter.py` is no longer used by this feature; `tests/features/sentence_reconstruction` still covers it.

## Language

The phonetic key's alphabet, letter folds, spelling rewrites, silent letters and spelled-out letter names, the candidate tokenizer, the rogue-zero sanitizer and the plural/verb-suffix rules all come from the language pack's `[phonetic]` table (`../core/language.md`). The English pack holds the values the code used to hard-code. Serbian and Slovenian set `plural_suffix = ""`: a term replaces a mis-heard span in its dictionary form, since a replacement can't reproduce a case ending. Inflection-aware matching is still needed before sr/sl glossaries are trusted. Glossaries are read from `data/glossaries/<code>/`.

