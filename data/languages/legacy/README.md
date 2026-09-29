# Legacy language packs

Packs kept for reference but **not served**: only `data/languages/*.toml` counts as a supported language, so nothing
here reaches the language pickers, the CLI, search (`settings.SEARCH_FTS_CONFIGS`) or the hosted demo. Starting an
appliance with `APPLIANCE_LANGUAGE` set to a code here fails with a message saying the language is legacy.

| Pack | Retired | Why |
|---|---|---|
| `sl.toml` (Slovenian) | 2026-09-29 | Whisper large-v3 reached only ~79% on FLEURS (target 95%), and bge-reranker-v2-m3 scores Slovenian questions about English material near zero, so cross-language search lagged far behind (40–50% hit@1 vs 100% native). Its glossaries stay in `data/glossaries/sl/`. |

**Bringing a pack back:** move it to `data/languages/`, add its code back to the shipped-pack list in
`tests/core/language/data/test_toml_pack_repository.py` and to `deploy/` (`LANGUAGES`, `DEFAULT_LANGUAGES`), then
measure it again with `scripts/eval/language_benchmark` and `scripts/eval/cross_language` before selling it.
Chunks already indexed in it keep their `fts_config`; they are keyword-searched again once the pack is back.
