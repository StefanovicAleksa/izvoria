"""
deploy/tests runs in its own venv (deploy/requirements-dev.txt): the gateway and agent depend on Starlette, psutil
and websockets, which the air-gapped appliance venv deliberately doesn't have. Under the appliance venv, a bare
`pytest` skips this tree rather than failing collection.
"""

import importlib.util

_NEEDED = ("starlette", "psutil", "websockets", "pytest_asyncio")

if any(importlib.util.find_spec(name) is None for name in _NEEDED):
    collect_ignore_glob = ["*"]
