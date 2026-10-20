# source_profile

`app/features/source_profile/`

## What it does

Gives every ingested source (recording, document, book, image) a **title** and a one-sentence **topic**, and uses them in three places:

1. **Every excerpt names its source.** The answer model and the reasoner see `[Doc 3] Lecture 4 Conditional Probability Statistics 110 (00:12:31–00:14:02)`, not just a time range, so "what did lecture 4 say" can be answered from what it was shown.
2. **The index names it too.** Each chunk's `context_text` and `text_payload` start with `[Source: <title> — <topic>]`, so the source's name is embedded and keyword-indexed with every chunk. `raw_text`, the citable text, never contains it. The reranker and the relevance scrubber read `RetrievedChunk.content_payload`, the payload without that line: it is the same on every chunk of a source, so judged with it every chunk of a lecture looked relevant to a question about the lecture's subject (English benchmark: rerank hit@1 10/10 → 9/10 with it, 10/10 without).
3. **A question that names one source is answered from that source** ("all the formulas in lecture 4 of Stat 110"). See "Scoped questions" below.

File names can mean nothing (`module_01.mp4`), which is why the topic is generated from the content, not taken from the name.

## How a profile is written

`SourceProfileAPI.generate(source_id, sections, model)` is called by `ContextualizerAPI.execute` / `execute_document` with the contextual SLM they already hold, so it costs no model load.

- A source in one section is profiled from its whole text. A longer one is profiled from one note per section, at most `MAX_NOTED_SECTIONS` (8) spread over it, the first always included.
- The prompt (`data/prompts.py`) tells the model to start from the file name when it names the source and keep its numbers. It has **no example titles**: the first version had one ("Statistics 110, Lecture 4"), and the 3B model copied it onto about 25 other lectures.
- **The number guard** (`utils/grounding.py`) runs without a model. A title stating a number the file name doesn't (or, for a name with no numbers, that the source's opening 3,000 characters don't), or dropping one the file name has, is replaced by the tidied file name; the generated topic is kept. A wrong lecture number would send every question about that lecture to the wrong recording.
- Titles are tidied of file extensions and YouTube `[id]` suffixes (`clipped_draft`), and capped at 120 characters (topics at 400).
- A reply that isn't usable JSON falls back to the file name (`ProfileOrigin.NAMED`) rather than failing ingestion. An escaped apostrophe (`Gambler\'s`), which small models write, is accepted.
- Images get a profile without a model (`name_only`): the file name, and the first sentence of the description as topic.
- A profile a person edited (`ProfileOrigin.EDITED`) is never replaced by a generated one.

Stored in `source_profiles` (migration 0007), one row per source, deleted with it.

## Scoped questions

`SourceScopePhase` runs first in `LiveQAPipeline` (chat) and `LiveReasoningPipeline` (reasoned answers), before any preload.

1. **Shortlist, no model** (`utils/scope_matcher.py`). The question's words are matched against each indexed source's name, file name and title, weighted by how rare they are. A source is listed only through a distinctive word: a number, or a word of 4+ letters that at most half the sources share. "stat" matches "statistics" by prefix. A question naming nothing (most questions) stops here and costs nothing.
2. **The answer model picks** among at most 5 candidates (`pick_scope`, JSON, thinking off), or says none. It is told that a topic is not a reference to a source, and that lecture 4 is not lecture 14. It is acquired with `wait_for_clear=True` and released; it is usually warm, and on the hosted version it is an external `llama-server` type, which strict residency allows.
3. **A picked source that fits** (`SOURCE_SCOPE_WHOLE_READ_MAX_TOKENS`, 48,000, estimated at 3 characters a token) is handed to the answer model whole, in reading order, with no search, rerank or scrub. That is what makes "list everything in lecture 4" answerable. A larger one is searched with `source_id_filter` and `SCOPED_SEARCH_LIMIT` (20). A reasoned answer always searches, with every retrieval pass filtered to the source.
4. The answer carries `QAResponse.scoped_source_title`, shown as "Answered from: …" in the desktop and web chat, so a wrong pick is visible.

Turned off with `SOURCE_SCOPE_ENABLED=false`. In a conversation, the scope is taken from the standalone (rewritten) question. Search modes are not scoped: they return excerpts, and a person can see the source of each.

## Editing

`IzvoriaEngine.edit_source_profile(source_id, title, topic)` saves a person's own title and topic and rewrites that source's `[Source: …]` line in place (`RAGStorageService.refresh_source_line`: same chunk ids, re-embedded with the embedder only, seconds). On the web, firm administrators edit them in the Library (`PUT /api/library/{source_id}/profile`). The desktop app has no Library view yet.

## Backfill

Sources ingested before profiles existed have none:

```
venv/bin/python -m scripts.backfill_source_profiles              # every source without a profile
venv/bin/python -m scripts.backfill_source_profiles --dry-run    # list them
venv/bin/python -m scripts.backfill_source_profiles --source <id> [--source <id> ...]
venv/bin/python -m scripts.backfill_source_profiles --regenerate # rewrite every profile nobody edited
```

Pass 1 loads the SLM once and profiles each source from its stored chunk texts. Pass 2 rewrites the `[Source: …]` lines and re-embeds in place. Chunk contexts are not regenerated. On the dev database (88 sources, 1,621 chunks) a full run takes about 7 minutes.

## Limits

- A topic is model-written and can be wrong ("a module on audio engineering" for a trading course). It only feeds the index and the scope judge, never a citation.
- Formulas written on a board are only as good as the transcript: the lecturer's spoken form ("P of A given B") is what gets indexed and quoted.
