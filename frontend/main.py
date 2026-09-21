#!/usr/bin/env python
# Canvas: desktop_main_v1
# Path: frontend/main.py

"""
Onyx desktop entrypoint.

Nothing above `apply_runtime_environment()` may import `app.*` or torch: this file exists to set
the environment those modules read at import time, before anything reads it.

    venv/bin/python frontend/main.py --stub     # UI only: no GPU, no database, canned answers
    venv/bin/python frontend/main.py            # the real appliance
"""

import argparse
import sys
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent


def main(argv: list) -> int:
    # `onyx_desktop` lives under frontend/; `app` lives at the repo root.
    for path in (str(FRONTEND_ROOT), str(PROJECT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    parser = argparse.ArgumentParser(description="Onyx desktop application")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run against a canned in-memory gateway instead of the real engine",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="value for ENV, which selects .env / .env.test / .env.prod (default: development)",
    )
    args, qt_args = parser.parse_known_args(argv[1:])

    if args.env:
        import os

        os.environ["ENV"] = args.env

    from onyx_desktop.service.desktop_app import apply_runtime_environment, run

    apply_runtime_environment(PROJECT_ROOT)
    return run([argv[0]] + qt_args, stub=args.stub)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
