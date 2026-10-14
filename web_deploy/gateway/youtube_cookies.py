"""
The YouTube account the GPU box downloads with: one cookies.txt, set by the owner.

YouTube refuses data-centre IPs that aren't signed in ("Sign in to confirm you're not a bot"), and the rented box has
one. So the owner exports the cookies of a spare Google account (docs/web_deploy.md, "Adding videos from YouTube") and
uploads them in the console or with admin_cli. They stay here, readable by the gateway only. The VPS staging sends them
to the box for a YouTube job (staging_server.stage_in), and the box deletes them as soon as the download step ends.

Only youtube.com and google.com cookies are kept: whatever else the browser exported never leaves this module.
Standard library only.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

MAX_BYTES = 1024 * 1024
HEADER = "# Netscape HTTP Cookie File"
KEPT_DOMAINS = ("youtube.com", "google.com")
# At most one email a day while YouTube keeps refusing the box
ALERT_EVERY_SECONDS = 24 * 3600
# What a signed-in session carries; without them YouTube treats the downloads as anonymous
SIGNED_IN = ("SAPISID", "__Secure-3PSID", "LOGIN_INFO")


class CookiesRejected(ValueError):
    """The upload isn't usable; the message says why, for the owner."""


def default_path(gateway_db: Path) -> Path:
    return gateway_db.with_name("youtube_cookies.txt")


def set_cookies(path: Path, text: str) -> dict:
    """Stores the youtube.com / google.com cookies of a Netscape cookies.txt. Replaces any earlier ones."""
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise CookiesRejected("The file is larger than 1 MB: export the cookies of youtube.com only.")
    rows = _rows(text)
    if not rows:
        raise CookiesRejected("That is not a cookies.txt file (Netscape format, as cookie export extensions save it).")
    kept = [r for r in rows if _domain_kept(r[0])]
    if not any(r[0].lstrip(".").endswith("youtube.com") for r in kept):
        raise CookiesRejected("The file has no youtube.com cookies: export them while youtube.com is open.")
    body = HEADER + "\n" + "".join("\t".join(r) + "\n" for r in kept)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(prefix=".youtube_cookies-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(body)
        os.chmod(partial, 0o600)
        os.replace(partial, path)
        _refused_stamp(path).unlink(missing_ok=True)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    return status(path)


def take_back(path: Path, text: str) -> bool:
    """
    The jar yt-dlp saved after a download: YouTube refreshes session cookies as it is used, so keeping them is what
    keeps the sign-in alive. Taken only when it is still a signed-in YouTube session (never a broken or signed-out
    jar over a good one). Returns whether it replaced the stored cookies.
    """
    try:
        rows = _rows(text)
    except Exception:  # noqa: BLE001 - whatever came back, the stored cookies stay
        return False
    if not any(r[5] in SIGNED_IN and _domain_kept(r[0]) for r in rows):
        return False
    try:
        set_cookies(path, text)
    except CookiesRejected:
        return False
    return True


def mark_refused(path: Path, clock=time.time) -> bool:
    """
    Records that YouTube refused a download as a bot (missing or stale cookies). Returns whether this is news, i.e.
    nothing was recorded in the last day: the owner is emailed once, not for every batch.
    """
    stamp = _refused_stamp(path)
    try:
        last = float(stamp.read_text().strip())
    except (OSError, ValueError):
        last = None
    now = clock()
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"{now:.0f}\n")
    return last is None or now - last >= ALERT_EVERY_SECONDS


def _refused_stamp(path: Path) -> Path:
    return path.with_name(path.name + ".refused")


def status(path: Path) -> dict:
    """
    {"present", "cookies", "signed_in", "expires" (earliest sign-in cookie, epoch s or None), "expired", "updated_at",
    "refused_at" (when YouTube last refused a download as a bot, or None)}.
    """
    try:
        refused_at = int(float(_refused_stamp(path).read_text().strip()))
    except (OSError, ValueError):
        refused_at = None
    try:
        text = path.read_text(encoding="utf-8")
        updated = path.stat().st_mtime
    except OSError:
        return {"present": False, "cookies": 0, "signed_in": False, "expires": None, "updated_at": None,
                "expired": False, "refused_at": refused_at}
    rows = _rows(text)
    session = [r for r in rows if r[5] in SIGNED_IN]
    expiries = [int(r[4]) for r in session if r[4].isdigit() and int(r[4]) > 0]
    return {
        "present": True, "cookies": len(rows), "signed_in": bool(session),
        "expires": min(expiries) if expiries else None, "updated_at": int(updated),
        "expired": bool(expiries) and min(expiries) < time.time(),
        # YouTube refused a download as a bot since these were uploaded or last refreshed: re-export them
        "refused_at": refused_at,
    }


def clear(path: Path) -> None:
    path.unlink(missing_ok=True)
    _refused_stamp(path).unlink(missing_ok=True)


def _rows(text: str) -> List[List[str]]:
    """The cookie lines: domain, include-subdomains, path, secure, expiry, name, value."""
    rows = []
    for line in text.splitlines():
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]  # curl's marker for HttpOnly cookies, still a cookie
        elif not line.strip() or line.startswith("#"):
            continue
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) == 7 and fields[0] and fields[5]:
            rows.append(fields)
    return rows


def _domain_kept(domain: str) -> bool:
    bare = domain.lstrip(".").lower()
    return any(bare == d or bare.endswith("." + d) for d in KEPT_DOMAINS)
