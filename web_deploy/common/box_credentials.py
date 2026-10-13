"""
The GPU box's database password, kept off its disk and out of every process environment.

The gateway gives the box's database role a fresh random password each time it starts the box, and disables the role
when it stops it (web_deploy/provision/izvoria-box-access). The box asks the gateway for the current password, through
its tunnel, whenever one of its processes opens a new database connection: `install_from_env()` registers that as the
password source for every engine in the process (app/core/database/data/credentials.py). Pools reuse connections, so
this is a local call now and then, not per query.

A process holds the password only in memory. Reading it means attacking a running process, not `cat`-ing a file, and
it stops working when the box is stopped. Standard library only.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional
from urllib.parse import urlparse

from web_deploy.common.signing import sign_service

CACHE_SECONDS = 10.0
# Waits this long for the gateway (the tunnel coming up, the gateway restarting) before giving up on a connection
WAIT_SECONDS = 180.0


class CredentialsUnavailable(RuntimeError):
    pass


def password_provider(url: str, secret: str, cache_seconds: float = CACHE_SECONDS, wait_seconds: float = WAIT_SECONDS,
                      fetch: Optional[Callable[[str, dict], str]] = None,
                      clock: Callable[[], float] = time.monotonic,
                      sleep: Callable[[float], None] = time.sleep) -> Callable[[], str]:
    """A callable returning the current password, asked of the gateway at most every `cache_seconds`."""
    fetch = fetch or _fetch
    path = urlparse(url).path
    lock = threading.Lock()
    cached = {"password": None, "at": float("-inf")}

    def current() -> str:
        with lock:
            if cached["password"] is not None and clock() - cached["at"] < cache_seconds:
                return cached["password"]
            deadline = clock() + wait_seconds
            delay = 1.0
            while True:
                try:
                    password = fetch(url, sign_service(secret, "POST", path))
                    cached.update(password=password, at=clock())
                    return password
                except CredentialsUnavailable:
                    if clock() + delay > deadline:
                        raise
                    sleep(delay)
                    delay = min(10.0, delay * 2)

    return current


def install_from_env(env: Optional[dict] = None) -> bool:
    """
    Makes every database connection in this process take its password from the gateway, when BOX_CREDENTIALS_URL is
    set (the hosted box). Returns whether it did; elsewhere (dev hardware, the VPS) nothing changes.
    """
    env = os.environ if env is None else env
    url = env.get("BOX_CREDENTIALS_URL", "").strip()
    if not url:
        return False
    from app.core.database.data.credentials import set_password_provider
    set_password_provider(password_provider(url, env.get("WEB_API_SECRET", "")))
    return True


def _fetch(url: str, headers: dict) -> str:
    request = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            password = json.loads(response.read().decode()).get("password")
    except urllib.error.HTTPError as exc:
        raise CredentialsUnavailable(f"The gateway refused the database password ({exc.code}).") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CredentialsUnavailable(f"The gateway could not be reached for the database password: {exc}") from exc
    if not isinstance(password, str) or not password:
        raise CredentialsUnavailable("The gateway sent no database password.")
    return password
