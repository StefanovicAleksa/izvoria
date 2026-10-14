"""
A firm's own glossaries, one per language: what its recordings are transcribed and corrected with.

The glossary is what Whisper is given as hotwords and what phonetic correction matches against
(app/features/domain_adaptation, app/features/domain_phonetic_correction). Without one, a firm's recordings get the
shared general glossary (data/glossaries/<lang>/general.json), as every firm did before this module.

A firm's glossary lives beside its other data on the VPS:

    <tenants>/<firm>/glossaries/<lang>/<industry>.json     the file the ingestion child reads
    <tenants>/<firm>/glossaries/<lang>/meta.json           where it came from, its size, the validator's warnings

At most one glossary per language. The ingestion child is pointed at it with GLOSSARY_PATH=<tenants>/<firm>/glossaries
and APPLIANCE_INDUSTRY=<industry> (ingest_worker/supervisor.py), which is exactly how the appliance reads its own
data/glossaries/<lang>/<industry>.json, so nothing in app/ changes. Set by the owner only: the owner console and
admin_cli. Standard library plus scripts/validate_glossary.py, which is standard library too.
"""

import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from scripts import validate_glossary
from web_deploy.common.tenancy import firm_root

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = REPO_ROOT / "data" / "glossaries"
MAX_BYTES = 2 * 1024 * 1024
META = "meta.json"
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class GlossaryRejected(ValueError):
    """The file can't be a glossary; the message (or `errors`) says why, for the person who uploaded it."""

    def __init__(self, message: str, errors: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def served_languages() -> List[str]:
    return validate_glossary.served_languages()


def builtins() -> Dict[str, List[str]]:
    """The glossaries shipped with Izvoria, per served language (e.g. {"en": ["finance", "general", ...]})."""
    return {code: sorted(p.stem for p in (BUILTIN_ROOT / code).glob("*.json")) for code in served_languages()}


def folder(tenants_root: Path, firm: str, language: str) -> Path:
    _language(language)
    return firm_root(tenants_root, firm) / "glossaries" / language


def status(tenants_root: Path, firm: str) -> Dict[str, Optional[dict]]:
    """Per served language: the firm's glossary as described by its meta, or None (the shared general one)."""
    out: Dict[str, Optional[dict]] = {}
    for code in served_languages():
        out[code] = _read_meta(folder(tenants_root, firm, code))
    return out


def active_file(tenants_root: Path, firm: str, language: str) -> Optional[Path]:
    """The glossary file an ingestion in this language should read, or None for the shared general one."""
    try:
        files = [p for p in folder(tenants_root, firm, language).glob("*.json") if p.name != META]
    except GlossaryRejected:
        return None
    return files[0] if len(files) == 1 else None


# ---------- setting one ----------

def set_uploaded(tenants_root: Path, firm: str, language: str, glossary: dict, original_name: str = "") -> dict:
    """
    The firm's own glossary for one language. Refused (GlossaryRejected) when the validator finds errors; its warnings
    are kept and shown. Replaces whatever the firm had for that language.
    """
    _language(language)
    if not isinstance(glossary, dict):
        raise GlossaryRejected("A glossary is a JSON object (base_terms, sub_domains, …).")
    if len(json.dumps(glossary, ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
        raise GlossaryRejected(f"The glossary is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not glossary.get("base_terms") and not glossary.get("sub_domains"):
        raise GlossaryRejected("The glossary has no terms: it needs base_terms or sub_domains.")
    findings = validate_glossary.check({language: glossary})
    if findings.errors:
        raise GlossaryRejected(f"The glossary has {len(findings.errors)} error(s).", findings.errors)
    key = industry_key(glossary.get("industry") or Path(original_name).stem)
    meta = _describe(glossary, language, key, source="upload", original_name=original_name,
                     warnings=findings.warnings)
    _install(folder(tenants_root, firm, language), key, glossary, meta)
    return meta


def set_builtin(tenants_root: Path, firm: str, language: str, name: str) -> Optional[dict]:
    """One of Izvoria's own glossaries for this firm and language. "general" is the default: it clears instead."""
    _language(language)
    if name not in builtins().get(language, []):
        raise GlossaryRejected(f"There is no built-in '{name}' glossary for '{language}'.")
    if name == "general":
        clear(tenants_root, firm, language)
        return None
    glossary = json.loads((BUILTIN_ROOT / language / f"{name}.json").read_text(encoding="utf-8"))
    key = industry_key(name)
    meta = _describe(glossary, language, key, source="builtin", builtin=name,
                     warnings=validate_glossary.check({language: glossary}).warnings)
    _install(folder(tenants_root, firm, language), key, glossary, meta)
    return meta


def clear(tenants_root: Path, firm: str, language: str) -> None:
    """Back to the shared general glossary."""
    shutil.rmtree(folder(tenants_root, firm, language), ignore_errors=True)


def import_claude_answer(tenants_root: Path, firm: str, answer: dict, original_name: str = "") -> Dict[str, dict]:
    """
    The glossary creator prompt's answer (docs/glossary_authoring), which holds every language at once: each
    language's glossary is set for the firm. All or nothing: nothing is written if any language is refused.
    """
    if not isinstance(answer, dict):
        raise GlossaryRejected("Expected Claude's JSON answer: an object with one glossary per language.")
    glossaries = validate_glossary.split_answer(answer)
    unknown = sorted(set(glossaries) - set(served_languages()))
    if unknown or not glossaries:
        raise GlossaryRejected(f"The answer's languages must be among {', '.join(served_languages())}"
                               + (f" (not {', '.join(unknown)})" if unknown else "") + ".")
    findings = validate_glossary.check(glossaries)
    if findings.errors:
        raise GlossaryRejected(f"The answer has {len(findings.errors)} error(s).", findings.errors)
    for glossary in glossaries.values():
        industry_key(glossary.get("industry") or Path(original_name).stem)  # refused before anything is written
    return {code: set_uploaded(tenants_root, firm, code, glossary, original_name)
            for code, glossary in glossaries.items()}


def industry_key(raw: str) -> str:
    """The file name the loader looks for (APPLIANCE_INDUSTRY lowercased): letters, digits, underscores."""
    key = re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower()).strip("_")
    if not _KEY.match(key):
        raise GlossaryRejected("The glossary needs an industry name (its \"industry\" key, e.g. \"legal_disputes\").")
    return key


# ---------- plumbing ----------

def _language(language: str) -> None:
    if language not in served_languages():
        raise GlossaryRejected(f"'{language}' is not a served language ({', '.join(served_languages())}).")


def _describe(glossary: dict, language: str, key: str, source: str, original_name: str = "", builtin: str = "",
              warnings: Optional[List[str]] = None) -> dict:
    sub_domains = glossary.get("sub_domains") or {}
    return {
        "language": language,
        "industry": key,
        "source": source,
        "builtin": builtin or None,
        "original_name": original_name or None,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "terms": len(set(validate_glossary.listed_terms(glossary))),
        "sub_domains": sorted(sub_domains) if isinstance(sub_domains, dict) else [],
        "warnings": list(warnings or [])[:50],
    }


def _install(target: Path, key: str, glossary: dict, meta: dict) -> None:
    """Writes the language's folder anew beside the old one, then swaps: a reader never sees half of it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o2770)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / f"{key}.json").write_text(json.dumps(glossary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
        (staging / META).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for path in [staging, *staging.iterdir()]:
            path.chmod(0o2770 if path.is_dir() else 0o640)  # the box's staging and the gateway share the group
        old = target.with_name(f".{target.name}-old")
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.replace(old)
        staging.replace(target)
        shutil.rmtree(old, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_meta(path: Path) -> Optional[dict]:
    try:
        meta = json.loads((path / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if any(p.name != META for p in path.glob("*.json")) else None
