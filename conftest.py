# Path: conftest.py

"""
Loaded by pytest before any test package's conftest, so before anything imports app.core.config.
"""

import os

# The desktop app and the CLI start in the language they were last switched to. A developer's remembered
# language must never decide which language a test runs in.
os.environ["REMEMBER_LANGUAGE"] = "false"
