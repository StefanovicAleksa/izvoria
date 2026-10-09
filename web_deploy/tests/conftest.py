"""
web_deploy/tests runs in its own venv (web_deploy/requirements-dev.txt). Under the appliance venv, which has no web
stack, a bare `pytest` skips this tree rather than failing collection. Tests that also need the appliance itself
(`app.*`: torch, a database) live in tests/integration and skip on their own.
"""

import importlib.util

_NEEDED = ("starlette", "httpx", "sqlalchemy", "pytest_asyncio")

if any(importlib.util.find_spec(name) is None for name in _NEEDED):
    collect_ignore_glob = ["*"]
