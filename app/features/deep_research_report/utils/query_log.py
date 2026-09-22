# Path: app/features/deep_research_report/utils/query_log.py

"""
The run's memory of what it has already searched for.

Exhaustive mode is allowed to gather more than once, and a round only earns its cost if it asks
something the run has not asked before. Two things stop it repeating itself:

* `already_asked` — a normalised set of every query issued so far, so a follow-up that restates one
  is dropped before it is paid for. The fallback follow-up is derived from a section's title and
  brief, which do not change between rounds, so without this the same query is re-issued every round
  until the budget runs out.
* `format_attempts` — the same history rendered for the outline prompt, with what each query returned.
  A query that added nothing is the load-bearing entry: it tells the model those words are not in this
  archive, which is the difference between re-wording a failed question and paraphrasing it.

Kept in memory for the duration of a run, like the follow-up queries themselves. None of it is part of
the plan the user approved, and persisting it would put transient queries in `research_report_sub_questions`
where `REPORT_MAX_SUB_QUESTIONS` would reject the plan on the next validation.
"""

import re
from typing import Iterable, List, Set

from app.features.deep_research_report.domain.entities import QueryAttempt

# How much of a query is shown to the model. Long enough to be recognisable, short enough that a dozen
# of them do not crowd the evidence index out of the window.
_MAX_SHOWN_CHARS = 160


def normalise_query(text: str) -> str:
    """
    The form two queries are compared in.

    Case, surrounding whitespace, internal runs of whitespace and trailing punctuation all vary between a
    model-written follow-up and the fallback built from a section's title and brief, while the search they
    produce is identical. Matching on the raw string would call those two different queries and re-run one.

    Deliberately conservative: it collapses only differences that cannot change what retrieval returns.
    Anything that reads as a genuinely different question stays different, because wrongly treating a new
    query as already-asked would silently drop a round that had something to find.
    """
    collapsed = re.sub(r"\s+", " ", text or "").strip().casefold()
    return collapsed.rstrip(" .?!,:;-")


def already_asked(attempts: Iterable[QueryAttempt]) -> Set[str]:
    """Every query issued so far, in comparable form."""
    return {normalise_query(attempt.text) for attempt in attempts}


def format_attempts(attempts: List[QueryAttempt]) -> str:
    """
    The history as a prompt block, or "" when nothing has been searched yet.

    Returning an empty string for round 1 matters: the block is interpolated into the outline prompt in
    both modes, and a deep run must not be shown a heading with nothing under it.
    """
    if not attempts:
        return ""

    lines = []
    for attempt in attempts:
        text = attempt.text.strip()
        if len(text) > _MAX_SHOWN_CHARS:
            text = text[:_MAX_SHOWN_CHARS - 1].rstrip() + "…"
        yielded = (
            f"{attempt.chunks_added} new excerpt(s)" if attempt.chunks_added
            else "NOTHING NEW -- these words are not in this archive"
        )
        lines.append(f'- "{text}" -> {yielded}')

    return "\nSEARCHES ALREADY RUN\n" + "\n".join(lines) + "\n"
