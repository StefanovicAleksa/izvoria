# Path: app/core/database/data/credentials.py
"""
Where a connection's password comes from when it must not sit in the environment.

The appliance never sets one: its password is in settings.DATABASE_URL, as always. The hosted GPU box does
(web_deploy/common/box_credentials.py): it holds no database password on disk or in any process environment, and
asks its gateway for the current one each time a new connection is opened, because the gateway changes it every time
it starts the box and disables it when it stops it.
"""

from typing import Callable, Optional

_provider: Optional[Callable[[], str]] = None


def set_password_provider(provider: Optional[Callable[[], str]]) -> None:
    """Every engine opened in this process then takes its password from `provider` (None goes back to the URL's)."""
    global _provider
    _provider = provider


def current_password() -> Optional[str]:
    """The password for a new connection, or None to keep the one in the URL."""
    return _provider() if _provider is not None else None


def apply_to_connection(cparams: dict) -> None:
    """For a SQLAlchemy `do_connect` listener: the provider's password, if there is a provider."""
    password = current_password()
    if password is not None:
        cparams["password"] = password
