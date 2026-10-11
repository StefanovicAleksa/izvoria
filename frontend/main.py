#!/usr/bin/env python
# Canvas: desktop_main_v1
# Path: frontend/main.py

"""
Izvoria desktop entrypoint.

Nothing above `apply_runtime_environment()` may import `app.*` or torch: this file exists to set
the environment those modules read at import time, before anything reads it.

    venv/bin/python frontend/main.py --stub     # UI only: no GPU, no database, canned answers
    venv/bin/python frontend/main.py            # the real appliance
    venv/bin/python frontend/main.py --autostart  # the real appliance, booted without a click
    venv/bin/python frontend/main.py --stub --mode light  # light or dark; also switchable in the app
"""

import argparse
import sys
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent


def main(argv: list) -> int:
    # `izvoria_desktop` lives under frontend/; `app` lives at the repo root.
    for path in (str(FRONTEND_ROOT), str(PROJECT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    parser = argparse.ArgumentParser(description="Izvoria desktop application")
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
    parser.add_argument(
        "--autostart",
        action="store_true",
        help="start the appliance as soon as the window opens, as if 'Start appliance' were clicked",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=("dark", "light"),
        help="light or dark. Default: the mode last picked in the app, else $ONYX_MODE, else dark",
    )
    args, qt_args = parser.parse_known_args(argv[1:])

    if args.env:
        import os

        os.environ["ENV"] = args.env

    from izvoria_desktop.service.desktop_app import apply_runtime_environment, run

    apply_runtime_environment(PROJECT_ROOT)
    return run([argv[0]] + qt_args, stub=args.stub, autostart=args.autostart, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
