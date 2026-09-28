# language

`app/core/language/` · packs in `data/languages/<code>.toml`

## What it does

Holds every rule in Onyx that depends on the language being spoken, one validated **language pack** per language. An appliance serves exactly one language, `settings.APPLIANCE_LANGUAGE` (`en`, `sr`, `sl` today). Nothing is translated: transcript, index and answers all stay in the language that was spoken, so a `[Doc N]` citation always points at words someone actually said.

Translation was ruled out deliberately. A translated index is model-written text under every citation, it multiplies ASR errors by MT errors, and it removes the sound-alike link phonetic correction relies on to find a mis-hearing.

## How it does it

- `LanguagePack` (`domain/entities.py`) is a frozen pydantic model with one table per concern. Its regexes are compiled at validation, so a bad pack fails at startup, not mid-ingestion.

  | Table | Contents |
  |---|---|
  | `transcription` | Whisper language, accepted detections, detection threshold, script, hallucination patterns |
  | `punctuation` | `model` or `whisper` |
  | `search` | Postgres FTS config |
  | `text` | stopwords, citation joiners |
  | `phonetic` | alphabet, letter names, folds, rewrites, plural/verb suffixes |
  | `glossary` | Wikidata languages |
  | `answers` | refusal, output instruction, transcript instruction |
  | `report` | boilerplate strings |
  | `models` | Settings overrides |

- `TomlLanguagePackRepository` (`data/`) reads `<code>.toml` with the stdlib `tomllib`, so there is no new dependency on a box without a package manager. It merges an NLTK stopword corpus into the pack when `nltk_stopwords` names one.
- `get_language_pack()` (`service/language_api.py`) is the entrypoint, cached per `(directory, code)`. It is a module-level function rather than an injected dependency, because most callers are pure helpers deep inside a feature (a phonetic key, a citation parser), and the language is fixed for the life of the process, like `settings`.
- `utils/script.py`: Serbian Cyrillic → Latin (exact; the reverse is ambiguous, e.g. "injekcija"), `normalize_script`, and `fold_to_ascii` (for filenames).
- `Settings` (`app/core/config.py`) applies a pack's `[models]` table and `search.fts_config` in a `model_validator`, unless the field was set explicitly (env, `.env` or constructor). It reads those two tables with `tomllib` directly: `app/core/language` imports `settings`, so `settings` can't import it back.

**Prompt localisation.** Prompts stay in English, which is what these models follow best.
- `LanguagePack.localize_prompt()` swaps the English refusal sentence for the pack's and appends an `<OUTPUT_LANGUAGE>` instruction. It is used only where output is stored or shown: QA, critique, the reasoner, the report, contextual retrieval and visual synthesis.
- `localize_transcript_prompt()` tells a transcript-editing prompt never to translate.
- For English both return the prompt byte-for-byte unchanged.
- `is_refusal()` accepts the pack's refusal and the English one, because a model can fall back to the sentence it saw in its instructions.

## What each language changes

| | en | sr | sl |
|---|---|---|---|
| FTS config | `english` | `serbian` (Snowball; stems Cyrillic, Latin, ekavian and ijekavian to one stem) | `simple` (Postgres has no Slovenian stemmer, so inflection is carried by the dense leg) |
| Script | — | Cyrillic → Latin, transcript and queries | — |
| Punctuation | fullstop classifier | Whisper's own | Whisper's own |
| Small models | Llama 3.2 3B / Qwen2.5-3B | Gemma 4 E4B | Gemma 4 E4B |
| Domain scout | whisper-tiny-en | faster-whisper-large-v3 | faster-whisper-large-v3 |
| Whisper detections accepted | en | sr, hr, bs | sl, hr |

The Gemma swap covers the domain classifier, contextual retrieval, visual synthesis, phonetic verification and visual hallucination correction. Qwen2.5-7B (cue detection, scrubber) and Qwen3.5-9B (QA, reasoner, report) serve every language.

## Where it fits

It is cross-cutting, like `config`. The consumers are:

- **Transcription:** Whisper language, the language guard, script normalisation.
- **Hallucination correction:** blacklist.
- **Sentence reconstruction:** punctuation adapter choice, made in `bootstrap.py`.
- **Phonetic correction:** key, tokenizer, sanitizer, grammar suffixes.
- **Glossaries:** `settings.GLOSSARY_DIR` is `data/glossaries/<code>/`, and the miner uses the pack's languages.
- **Retrieval:** query script, and the index-language guard at `OnyxEngine.start()`.
- **QA / reasoner / report:** localised prompts, refusal, citation joiners, report strings, slug.
- **Visual grounding:** alphabet, function words, script.

## Adding a language

1. **Measure Whisper large-v3's WER on real client recordings first.** If it can't reach the accuracy target, the language isn't sellable; point `WHISPER_MODEL_PATH` in the pack's `[models]` at a fine-tuned model and measure again.
2. Copy `sl.toml` to `<code>.toml` and fill every table. `tests/core/language/data/test_toml_pack_repository.py` pins the list of shipped packs; add the code there.
3. Check `SELECT cfgname FROM pg_ts_config` for a stemmer, or use `simple`.
4. Put `<code>` in `accepted_detections`, plus any languages Whisper confuses it with.
5. Build glossaries in `data/glossaries/<code>/` (`scripts/run_miner.py` with `APPLIANCE_LANGUAGE=<code>`). Without them, phonetic correction skips itself.

## Gotchas / design notes

- **Changing `APPLIANCE_LANGUAGE` on an existing index fails startup** with `IndexLanguageMismatchError`. The `fts_document` column is generated with the old stemmer at table creation, and no migration regenerates it. Reset the database and re-ingest.
- **A file in the wrong language is rejected, not transcribed.** `TranscriptionAPI` runs Whisper's language ID over the first 30 s of VAD speech. A confident detection outside `accepted_detections` raises `LanguageMismatchError`. Forcing a language on foreign speech makes Whisper translate or invent fluent text, which would look grounded in the index.
- **sr/sl phonetic correction has no inflection handling** (`plural_suffix = ""`). A glossary term replaces a mis-heard span in its dictionary form, so a case ending can be lost ("infarkta" → "infarkt"). Needs inflection-aware matching before sr/sl glossaries are trusted.
- **The sr/sl hallucination patterns are seeds.** Confirm them against real transcripts: a pattern deletes a whole segment.
- **The VLM stays English** (Qwen2-VL is strongest there, and the grounding checks read its English fields). Visual synthesis writes the stored context in the appliance's language; that text is never citable, so writing it in another language changes no evidence.
- **The desktop UI is not localised.**
