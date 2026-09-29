# language

`app/core/language/` · packs in `data/languages/<code>.toml`

## What it does

Holds every rule in Onyx that depends on the language being spoken, one validated **language pack** per language. An appliance serves every language that has a pack in `data/languages/` (`en`, `sr` today), in one database. Packs in `data/languages/legacy/` are kept but not served (Slovenian; see its README):

- **A batch has a language.** It is transcribed and indexed in the language picked for it (the Ingest view's "Language of these files", or the CLI's prompt).
- **The index holds every language.** Each chunk is keyword-indexed with its own language's stemmer (`vector_chunks.fts_config`), and search finds evidence in all of them.
- **Answers are written in the active language**, `settings.APPLIANCE_LANGUAGE`, which is switched at runtime (the desktop's "Answer in", or the CLI's menu) and remembered across runs.

Nothing is translated: transcript, index and excerpts stay in the language that was spoken, so a `[Doc N]` citation always points at words someone actually said. An English answer over Serbian evidence quotes the Serbian excerpt verbatim.

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
- `get_language_pack()` (`service/language_api.py`) is the entrypoint, cached per `(directory, code)`. It is a module-level function rather than an injected dependency, because most callers are pure helpers deep inside a feature (a phonetic key, a citation parser). It serves the active language, or `get_language_pack(code)` a named one (a stored report renders with its own). `get_supported_language_packs()` lists them all.
- `utils/script.py`: Serbian Cyrillic → Latin (exact; the reverse is ambiguous, e.g. "injekcija"), `normalize_script`, and `fold_to_ascii` (for filenames).
- `Settings` (`app/core/config.py`) applies a pack's `[models]` table and `search.fts_config` in a `model_validator`, unless the field was set explicitly (env, `.env` or constructor). It reads those two tables with `tomllib` directly: `app/core/language` imports `settings`, so `settings` can't import it back.

**Switching the active language.** There is no per-call language parameter. Only one engine job runs at a time (`EngineState`, one desktop worker thread), so the language is switched between jobs and every existing `get_language_pack()` / `settings.*` call site follows it.
- `Settings.use_language(code)` sets `APPLIANCE_LANGUAGE` and re-applies the pack. Fields the previous pack set go back to their defaults first, so leaving Serbian doesn't keep its Gemma. An explicit field still wins over every pack. The explicit set is snapshotted at construction (`_explicit_fields`), because pydantic adds every assigned field to `model_fields_set`.
- `OnyxBootstrapper.switch_language(engine, code)` does the switch. It checks the engine is idle, stops it, calls `use_language`, then `build()` and `start()`. Model paths are baked into manifests at build time, so the engine is rebuilt rather than patched. That takes seconds and loads no model, since `stop()` frees every resident model first. A failed rebuild returns settings to the previous language.
- `Settings.remember_language()` writes `onyx_data/<ENV>/active_language`. Without an explicit `APPLIANCE_LANGUAGE`, the next start reads it back (`REMEMBER_LANGUAGE`, which the root `conftest.py` turns off for tests). The hosted demo sets the variable, so its landing-page choice wins.
- `settings.supported_languages()` is the codes of the `*.toml` packs, and `settings.SEARCH_FTS_CONFIGS` their text search configurations.
- A cache over pack-dependent results must key on the language: `term_candidate_finder._cached_key_similarity` does.

**Prompt localisation.** Prompts stay in English, which is what these models follow best.
- `LanguagePack.localize_prompt()` swaps the English refusal sentence for the pack's and appends an `<OUTPUT_LANGUAGE>` instruction. It is used only where output is stored or shown: QA, critique, the reasoner, the report, contextual retrieval and visual synthesis.
- `localize_transcript_prompt()` tells a transcript-editing prompt never to translate.
- For English both return the prompt byte-for-byte unchanged.
- `is_refusal()` accepts the pack's refusal and the English one, because a model can fall back to the sentence it saw in its instructions.

## What each language changes

| | en | sr |
|---|---|---|
| FTS config | `english` | `serbian` (Snowball; stems Cyrillic, Latin, ekavian and ijekavian to one stem) |
| Script | — | Cyrillic → Latin, transcript and queries |
| Punctuation | fullstop classifier | Whisper's own |
| Small models | Llama 3.2 3B / Qwen2.5-3B | Gemma 4 E4B |
| Domain scout | whisper-tiny-en | faster-whisper-large-v3 |
| Whisper detections accepted | en | sr, hr, bs |

The Gemma swap covers the domain classifier, contextual retrieval, visual synthesis, phonetic verification and visual hallucination correction. Qwen2.5-7B (cue detection, scrubber) and Qwen3.5-9B (QA, reasoner, report) serve every language.

## Where it fits

It is cross-cutting, like `config`. The consumers are:

- **Transcription:** Whisper language, the language guard, script normalisation.
- **Hallucination correction:** blacklist.
- **Sentence reconstruction:** punctuation adapter choice, made in `bootstrap.py`.
- **Phonetic correction:** key, tokenizer, sanitizer, grammar suffixes.
- **Glossaries:** `settings.GLOSSARY_DIR` is `data/glossaries/<code>/`, and the miner uses the pack's languages.
- **Retrieval:** every pack's query script, and one keyword query per text search configuration.
- **QA / reasoner / report:** localised prompts, refusal, citation joiners, report strings, slug. A report records its language (`research_reports.language`): it renders in it, and resuming it switches to it.
- **Storage:** `sources.language`, the language a file was ingested in.
- **Visual grounding:** alphabet, function words, script.

## Adding a language

1. **Measure Whisper large-v3's WER on real client recordings first.** If it can't reach the accuracy target, the language isn't sellable; point `WHISPER_MODEL_PATH` in the pack's `[models]` at a fine-tuned model and measure again.
2. Copy `sr.toml` (or the legacy `legacy/sl.toml`) to `<code>.toml` and fill every table. `tests/core/language/data/test_toml_pack_repository.py` pins the list of shipped packs; add the code there.
3. Check `SELECT cfgname FROM pg_ts_config` for a stemmer, or use `simple`.
4. Put `<code>` in `accepted_detections`, plus any languages Whisper confuses it with.
5. Build glossaries in `data/glossaries/<code>/` (`scripts/run_miner.py` with `APPLIANCE_LANGUAGE=<code>`). Without them, phonetic correction skips itself.

## Gotchas / design notes

- **A file in the wrong language is rejected, not transcribed.** This is how a wrong batch-language pick is caught. `TranscriptionAPI` runs Whisper's language ID over the first 30 s of VAD speech. A confident detection outside `accepted_detections` raises `LanguageMismatchError`, which names the language to ingest the file as instead. Forcing a language on foreign speech makes Whisper translate or invent fluent text, which would look grounded in the index. `TranscribePhase` then un-stamps the source (`sources.language = NULL`) and discards its glossary, which was built from a scout pass in the wrong language, so a retry with the right language starts clean.
- **A file keeps the language it was ingested in.** Re-ingesting it with another language selected is refused (`SourceLanguageConflict`). Its transcript and every later stage are cached per source in the first language, and the language check vouched for that one.
- **Documents and images have no language check.** A wrong pick indexes them with the wrong stemmer and OCR model, and `SourceLanguageConflict` then keeps them that way. Resetting the index is the only fix today.
- **A mixed-language recording is transcribed in its batch's language.** A deposition in two languages is not split.
- **Serbian phonetic correction has no inflection handling** (`plural_suffix = ""`). A glossary term replaces a mis-heard span in its dictionary form, so a case ending can be lost ("infarkta" → "infarkt"). Needs inflection-aware matching before Serbian glossaries are trusted.
- **The Serbian hallucination patterns are seeds.** Confirm them against real transcripts: a pattern deletes a whole segment.
- **The VLM stays English** (Qwen2-VL is strongest there, and the grounding checks read its English fields). Visual synthesis writes the stored context in the appliance's language; that text is never citable, so writing it in another language changes no evidence.
- **Slovenian is legacy, not deleted.** Its pack sits in `data/languages/legacy/`, its glossaries in `data/glossaries/sl/`, and its benchmark sources in `scripts/eval/language_benchmark/sources.toml`. A database that already holds Slovenian chunks keeps them (`fts_config = simple`): the dense leg still finds them, but the keyword leg doesn't query `simple` while no served pack uses it. `sparse_query.py` keeps its handling of unstemmed configurations for that reason.
- **The desktop UI is not localised.** Only the language pickers name each language in itself ("Srpski").
