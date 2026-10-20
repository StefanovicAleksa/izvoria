# Path: app/features/source_profile/utils/grounding.py

"""
Checks, without a model, that a generated title states its source's numbers and only those. A small model copies
numbers from wherever it saw them (a prompt's example, another lecture), and "Lecture 4" on lecture 20 would send
every question about lecture 4 to the wrong recording; it also drops them ("Exponential Distribution" for lecture
16), and the number is how people refer to a lecture, unit or module. Words are not checked: a descriptive title for
a badly named file legitimately uses words the source never says.
"""

from typing import List, Sequence

from app.features.source_profile.utils.profile_text import title_from_name
from app.features.source_profile.utils.scope_matcher import tokens

# Where a source introduces itself ("today, lecture four…", a document's cover page)
OPENING_CHARS = 3000


def invented_numbers(title: str, names: Sequence[str], texts: Sequence[str]) -> List[str]:
    """
    The numbers in `title` the source does not state. A file name with numbers in it is the authority on them (a
    transcript mentions many numbers, so its body proves nothing); a name without numbers lets the title take them
    from the source's opening.
    """
    named_numbers = _name_numbers(names)
    allowed = named_numbers or {token for token in tokens(" ".join(texts)[:OPENING_CHARS]) if token.isdigit()}
    return sorted(token for token in tokens(title) if token.isdigit() and token not in allowed)


def dropped_numbers(title: str, names: Sequence[str]) -> List[str]:
    """The numbers of the source's file name that `title` leaves out."""
    title_tokens = tokens(title)
    return sorted(token for token in _name_numbers(names) if token not in title_tokens)


def _name_numbers(names: Sequence[str]) -> set:
    """The numbers in the file names, read without extension or YouTube id."""
    return {token for name in names if name for token in tokens(title_from_name(name)) if token.isdigit()}
