# search_history

`app/features/search_history/`: every standard and verified search is kept with what it returned, so it can be reopened exactly as it was shown. This page also covers the conversation half of history (listing, re-reading and deleting chat conversations), which is `session_memory`'s.

## What is stored

`search_history` (migration `0003`), one row per successful search:

| Column | Meaning |
|---|---|
| `query`, `mode` (`standard` / `verified`), `result_limit` | what was asked, and how |
| `chunk_ids uuid[]`, `scores float8[]` | the hits as shown, in rank order, with the score each had. For a verified search these are the hits that survived the scrubber |
| `execution_time_ms`, `created_at` | |

**`chunk_ids` is not a foreign key**, following the same rule as `research_report_evidence.chunk_id`. Re-ingesting a file cascades its chunks away, and the record of what was searched for has to outlive that. A reader hydrates the ids with `IzvoriaEngine.get_chunks` and simply gets fewer back. The desktop app counts the gap ("1 hit is no longer in the index") rather than hiding it.

`SearchRecord` refuses a row whose scores don't line up with its chunks (`SearchHistoryDataError`).

## Where it is written

`IzvoriaEngine.search_standard` / `search_verified` call `SearchHistoryAPI.record` after the search succeeds, outside the search's `try`. A failed search records nothing. A failure to record is logged and never costs the user the search they just ran: history is a convenience, the search is the job.

## Reading it back (engine facade)

These are database reads that take no engine lock and load no model. They need only `start()` (`_require_started`), so they work while an ingestion or a report holds the GPU:

- `list_searches(limit)` → `SearchRecordSummary` (hit count via `cardinality(chunk_ids)`), newest first.
- `get_search(id)` → `SearchRecord`, or `None`.
- `delete_search(id)`.
- `list_conversations(limit)` → `SessionSummary` per session: its title is the first question, `started_at` / `last_at` / `turn_count`, most recently active first. One `GROUP BY session_id` over `session_turns`, with the title from `array_agg(question ORDER BY created_at)[1]`.
- `get_conversation(session_id)` → every `SessionTurn`, oldest first. It's `SessionMemoryAPI.get_all_turns`, not `get_history`, which is windowed to `CHAT_HISTORY_MAX_TURNS` for the model's prompt.
- `delete_conversation(session_id)` drops the turns. Reports started from that conversation are kept.

`get_chunks` was relaxed to the same `_require_started` check. It used to refuse while `INGESTING`/`RESEARCHING` although it only reads the database, which would have made evidence unreadable exactly when history is most useful.

## Reopening a conversation

The desktop gateway rebuilds `[Doc N]` from the turns alone (`IzvoriaEngineGateway.load_conversation`). The numbers are conversation-wide: turn *k*'s new evidence is numbered `first_doc_number`, `first_doc_number + 1`, … So one number → chunk map resolves every tag in every answer, including a follow-up that cites an earlier turn's excerpt.

- A chat answer is stored with its tags. They are normalised (`[Doc 1, 3]` → `[Doc 1][Doc 3]`), and its cards are exactly what it cites.
- A Reasoned Answer is stored without tags (its own numbering would collide), so its cards are the evidence it showed first.

## Tests

- `tests/features/search_history/`: repository, API and error mapper, mocked.
- `integration_tests/features/search_history/`: a real round trip on `db_test`, including ids that no longer exist being stored rather than rejected.
- `integration_tests/core/database/test_migrations_integration.py` drops `search_history` when it fakes a pre-Alembic database, because those never had it.
