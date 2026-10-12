"""
A cited source, served to the browser that cited it.

The noVNC demo's agent (deploy/instance/agent/) already solved the hard parts: which files a browser can show, the
vault check that refuses anything outside the vault or behind a symlink, and the ffmpeg cuts and transcodes behind a
5 GB cache. They are reused here as they are. What differs is where the request comes from: not an "offer" the
desktop app posted, but a chunk id the user's own answer cited, re-resolved through the firm's own engine, so a
user can only ever open a file of their own firm's vault.

Two halves, on two machines. The box's API knows the chunk (`locate`: its file's path and the cited span); the
gateway on the VPS holds the vault and serves the file (`offer_for_chunk` and the rest). The gateway re-checks the
path against the firm's own vault, so not even the box can make it serve anything else.
"""

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote

from deploy.instance.agent.sources import (
    Offer, OfferRequest, SourceKind, SourceRejected, Variant, classify, resolve_vault_path, variants_for,
)

__all__ = ["SourceRejected", "Variant", "locate", "offer_for_chunk", "file_headers", "describe"]


def locate(chunk: Any) -> Dict[str, Any]:
    """What the gateway needs of a cited chunk (box side): plain values, no file is touched."""
    anchor = getattr(getattr(chunk, "anchor_kind", None), "value", getattr(chunk, "anchor_kind", None)) or "time"
    extracted = getattr(chunk, "extracted_file_id", None)
    return {
        "physical_path": str(getattr(chunk, "physical_path", None) or ""),
        "anchor_kind": str(anchor),
        "extracted_file_id": str(extracted) if extracted else None,
        "start_sec": float(getattr(chunk, "start_sec", None) or 0.0),
        "end_sec": float(getattr(chunk, "end_sec", None) or 0.0),
        "page_start": getattr(chunk, "page_start", None),
        "page_end": getattr(chunk, "page_end", None),
    }


def offer_for_chunk(location: Dict[str, Any], vault_root: Path) -> Offer:
    """
    What can be served for a located chunk (gateway side). Raises SourceRejected when its file is not in the firm's
    vault: a path outside it, behind a symlink, or not there yet (an ingestion whose results are still being pushed).
    """
    path = resolve_vault_path(str(location.get("physical_path") or ""), vault_root)
    kind, mime = classify(path)
    anchor = str(location.get("anchor_kind") or "time")
    request = OfferRequest(
        path=str(path),
        media_kind="clip" if location.get("extracted_file_id") else ("full_source" if anchor == "time" else anchor),
        anchor_kind=anchor,
        start_sec=float(location.get("start_sec") or 0.0),
        end_sec=float(location.get("end_sec") or 0.0),
        page_start=_page(location.get("page_start")),
        page_end=_page(location.get("page_end")),
    )
    return Offer(
        id=secrets.token_urlsafe(16), seq=0, path=path, kind=kind, mime=mime, request=request, created_at=0.0,
        created_wall=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        variants=variants_for(kind, path, request),
    )


def _page(value: Any):
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def describe(offer: Offer, doc_label: str, location: str) -> dict:
    public = offer.public()
    name = _download_name(doc_label, location, offer.path.suffix)
    return {
        "kind": public["kind"], "clip": public["clip"], "variants": public["variants"],
        "anchor_kind": public["anchor_kind"], "start_sec": public["start_sec"], "end_sec": public["end_sec"],
        "page_start": public["page_start"], "page_end": public["page_end"],
        "excerpt_start_sec": public["excerpt_start_sec"], "size_bytes": public["size_bytes"],
        "download_name": name,
    }


def _download_name(doc_label: str, location: str, suffix: str) -> str:
    from deploy.instance.agent.sources import download_name
    return download_name(doc_label[:64], location[:300], suffix)


def file_headers(offer: Offer, variant: Variant, mime: str, download_name: str) -> dict:
    """
    Every served file: its type from the table and never sniffed, never cached, and sandboxed so nothing in a client's
    file can script the firm's origin. A PDF shown inline goes without the sandbox (Chrome refuses to render one
    under it); its viewer runs no document script in the page.
    """
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"}
    if not (mime == "application/pdf" and variant is Variant.MEDIA):
        headers["Content-Security-Policy"] = "sandbox"
    if variant is Variant.ORIGINAL or offer.kind is SourceKind.DOCUMENT:
        fallback = download_name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
        headers["Content-Disposition"] = f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(download_name)}"
    else:
        headers["Content-Disposition"] = "inline"
    return headers
