"""
What a firm is called everywhere: its slug (the sign-in name its people type), and the database and
directories derived from it.

Shared by the gateway (which only knows slugs) and the GPU box (which turns a slug into a database and a vault), so
the two can never disagree about which firm a request belongs to. Standard library only: the gateway venv has no
appliance dependencies.
"""

import re
from pathlib import Path

# A short name a person can type, that is also safe inside a Postgres identifier once '-' becomes '_'
_SLUG = re.compile(r"^[a-z][a-z0-9-]{1,30}[a-z0-9]$")
# Names that are never a firm ("admin" is the owner's console)
RESERVED_SLUGS = frozenset({"admin", "www", "api", "app", "mail", "static", "assets", "status", "control"})
DATABASE_PREFIX = "onyx_firm_"


class InvalidSlug(ValueError):
    pass


def validate_slug(slug: str) -> str:
    slug = (slug or "").strip().lower()
    if not _SLUG.match(slug) or "--" in slug:
        raise InvalidSlug(
            f"'{slug}' is not a valid firm name: 3-32 lowercase letters, digits and single hyphens, "
            f"starting with a letter."
        )
    if slug in RESERVED_SLUGS:
        raise InvalidSlug(f"'{slug}' is reserved.")
    return slug


def database_name(slug: str) -> str:
    """onyx_firm_legal_demo: the firm's own database. One per firm, shared by all of its users."""
    return DATABASE_PREFIX + validate_slug(slug).replace("-", "_")


def firm_root(tenants_root: Path, slug: str) -> Path:
    """Everything the firm stores on the box: vault/, reports/, uploads/."""
    return tenants_root / validate_slug(slug)

