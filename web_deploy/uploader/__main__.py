"""
Izvoria's uploader, for a firm with a lot of video: run on the firm's own computer, it shrinks each recording to the
720p copy the appliance keeps anyway, and sends only that, in parallel pieces that survive a dropped connection.
Raw footage at ~100 GB an hour becomes ~1 GB, so a day of uploading becomes minutes.

  python -m web_deploy.uploader --server https://izvoria.example.com --firm legal-demo --email you@firm.com \\
      --mode video --language en "D:/Depositions 2026"

It asks for the firm password and your own (or reads IZVORIA_FIRM_PASSWORD and IZVORIA_PASSWORD). Run the same
command again after an interruption and it continues where it stopped. Needs Python 3.10+ and, for video, ffmpeg and
ffprobe on the PATH; nothing else. Audio, documents and images are sent as they are.

docs/web_deploy.md, "Uploading a lot of video".
"""

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from web_deploy.common.video_copy import CopySettings
from web_deploy.uploader.client import Gateway, GatewayError
from web_deploy.uploader.transfer import CHUNK_BYTES, Plan, State, collect, run


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m web_deploy.uploader",
                                     description="Send a batch of files to Izvoria, shrinking videos to 720p first.")
    parser.add_argument("paths", nargs="+", type=Path, help="files and folders (a folder's subfolders are walked)")
    parser.add_argument("--server", required=True, help="the address you sign in at, e.g. https://izvoria.example.com")
    parser.add_argument("--firm", required=True, help="your firm's sign-in name")
    parser.add_argument("--email", required=True)
    parser.add_argument("--mode", required=True, choices=["video", "audio", "document", "image"])
    parser.add_argument("--language", required=True, help="the language spoken or written in these files (en, sr)")
    parser.add_argument("--no-shrink", action="store_true", help="send videos as they are")
    parser.add_argument("--parallel", type=int, default=4, help="pieces sent at once (default 4)")
    parser.add_argument("--chunk-mb", type=int, default=CHUNK_BYTES // (1024 * 1024), help="piece size (default 32)")
    parser.add_argument("--crf", type=int, default=26, help="720p copy quality, lower is better (default 26)")
    parser.add_argument("--preset", default="faster", help="x264 speed (default faster)")
    parser.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()) / "izvoria-uploader",
                        help="where copies wait to be sent (at most two at a time)")
    args = parser.parse_args(argv)

    sources, skipped = collect(args.paths, args.mode)
    if not sources:
        print(f"Nothing to send: no {args.mode} files in {', '.join(map(str, args.paths))}.", file=sys.stderr)
        return 1
    print(f"{len(sources)} {args.mode} file(s) to send" + (f"; {skipped} of another kind left out" if skipped else ""))

    plan = Plan(mode=args.mode, language=args.language, sources=sources, shrink=not args.no_shrink,
                parallel=max(1, args.parallel), chunk=args.chunk_mb * 1024 * 1024,
                copy=CopySettings(crf=args.crf, preset=args.preset))
    firm_password = os.environ.get("IZVORIA_FIRM_PASSWORD") or getpass.getpass(f"Password of the firm {args.firm}: ")
    password = os.environ.get("IZVORIA_PASSWORD") or getpass.getpass(f"Your password ({args.email}): ")

    gateway = Gateway(args.server)
    try:
        gateway.sign_in(args.firm, firm_password, args.email, password)
        state = State.for_plan(args.scratch / "state", gateway.base_url, plan)
        job = run(gateway, plan, state, args.scratch / "copies")
    except GatewayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.retryable:
            print("Run the same command again to continue where it stopped.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it stopped.", file=sys.stderr)
        return 130
    print(f"Queued: {job.get('file_count')} file(s), {job.get('status')}. Follow it on the Ingest page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
