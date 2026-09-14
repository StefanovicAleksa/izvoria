# glossary_miner

`app/tools/glossary_miner/`

## What it does

An **offline dev tool** (not part of the runtime engine/bootstrap — never wired into `OnyxBootstrapper`) that mines Wikidata for domain-specific terminology and packages it into a JSON "glossary manifest." These manifests are what `domain_adaptation` and `domain_phonetic_correction` load at runtime to build Whisper hotwords/prompts and correction dictionaries — this tool is how those glossary files get built in the first place, ahead of time, not at appliance runtime.

## How it does it

`GlossaryMinerAPI` (`service/miner_api.py`) has two entry points: `mine_and_package_subdomain()` (legacy, single sub-domain) and `build_industry_manifest()` (aggregates multiple sub-domains into one manifest). Pipeline per sub-domain:

1. **Extract** — `IWikidataAdapter.fetch_domain_terms(q_id, limit)`; the concrete `WikidataSparqlClient` (`data/sparql_client.py`) uses `SPARQLWrapper` against `https://query.wikidata.org/sparql` with a required custom User-Agent, running a query that unions subclass (`wdt:P279+`) and instance (`wdt:P31/wdt:P279*`) branches under a given Q-ID, plus optional `skos:altLabel` aliases filtered to `en`/`la`. Results fold into nested `WikidataItem(item_id, label, aliases_by_lang)` entities.
2. **Transform** — `clean_and_extract_terms()` (`utils/term_cleaner.py`) applies heuristics: drop terms over 3 words, strip parenthetical/bracket content, strip commas/colons (Whisper's `initial_prompt` is comma-separated so raw commas would corrupt it), normalize whitespace, enforce ASCII-only, dedupe, sort.
3. **Package** — builds a `SubDomainGlossary(anchor_words, terms)` → `OnyxGlossaryManifest(base_terms, sub_domains)`.
4. **Load** — `IGlossaryStorageAdapter.save_manifest()`; the concrete `LocalFileStorageAdapter` writes to `data/glossaries/<file>.json` via `model_dump_json(indent=4)`.

## Inputs / Outputs

Input: Wikidata Q-IDs + an anchor-words config dict. Output: the absolute path to a saved JSON glossary file under `data/glossaries/`.

## Where it fits

Not part of any orchestration Phase or Pipeline — it's a build-time/setup-time CLI tool (`scripts/run_miner.py`) run manually by a developer to produce the glossary files that `domain_adaptation`/`domain_phonetic_correction` read at runtime.

## Gotchas / design notes

- `service/llm_cleaner.py` exists but is **completely empty** (0 bytes) — a stub for an apparently-planned LLM-based cleanup step that was never implemented; only the regex-based `term_cleaner.py` is actually used.
- Has a hard network dependency on the public Wikidata endpoint, with no local cache or offline fallback — worth flagging given the rest of the appliance is explicitly offline/air-gapped. This is acceptable only because it's a dev-time tool, never run on client hardware.
