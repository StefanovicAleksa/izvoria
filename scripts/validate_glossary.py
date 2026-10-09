# Path: scripts/validate_glossary.py
"""
Checks glossaries written with docs/glossary_authoring/glossary_creator_prompt.md, and splits Claude's answer into
the per-language files Onyx reads (data/glossaries/<lang>/<industry>.json).

The loaders read only `base_terms` and `sub_domains.<LABEL>.{anchor_words, terms}`; `description`, `term_details`
and `review` ride along for people and for later code (Serbian `forms`), so a file written here stays a drop-in
glossary.

Run from the repo root (no GPU, no database):
    venv/bin/python -m scripts.validate_glossary <industry> [--languages en sr] [--dir data/glossaries]
    venv/bin/python -m scripts.validate_glossary --from-claude answer.json [--write]

Exits 1 when there are errors. Warnings (e.g. a hand-written glossary with no term_details) don't fail it.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "glossaries"

_LABEL = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Languages whose glossaries must be Latin script (the transcript is normalised to Latin)
LATIN_ONLY = {"sr"}
MAX_ANCHORS = 8


@dataclass
class Findings:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def served_languages() -> List[str]:
    """One per pack in data/languages/ (legacy packs are not served)."""
    return sorted(path.stem for path in (REPO_ROOT / "data" / "languages").glob("*.toml"))


def listed_terms(glossary: dict) -> List[str]:
    """Every term the loaders would read: base terms, then each sub-domain's anchors and terms."""
    terms = list(glossary.get("base_terms", []))
    for entry in glossary.get("sub_domains", {}).values():
        if isinstance(entry, dict):
            terms += entry.get("anchor_words", []) + entry.get("terms", [])
        else:
            terms += list(entry)
    return terms


def split_answer(answer: dict) -> Dict[str, dict]:
    """Claude's answer -> {lang: glossary}, each language's review items moved into its own file."""
    glossaries = {code: dict(glossary) for code, glossary in answer.items() if code != "review"}
    for item in answer.get("review", []):
        glossary = glossaries.get(item.get("language"))
        if glossary is not None:
            glossary.setdefault("review", []).append(item)
    return glossaries


def merge(existing: dict, new: dict) -> dict:
    """
    Adds a run's glossary to what is already on disk, so per-sub-domain batches accumulate. A sub-domain in `new`
    replaces the stored one; base terms and review items are unioned; term details of `new` win.
    """
    merged = dict(existing)
    for key in ("industry", "language"):
        if new.get(key):
            merged[key] = new[key]
    merged["base_terms"] = _unique(list(existing.get("base_terms", [])) + list(new.get("base_terms", [])))
    merged["sub_domains"] = {**existing.get("sub_domains", {}), **new.get("sub_domains", {})}
    merged["term_details"] = {**existing.get("term_details", {}), **new.get("term_details", {})}
    review = list(existing.get("review", []))
    review += [item for item in new.get("review", []) if item not in review]
    if review:
        merged["review"] = review
    return merged


def check(glossaries: Dict[str, dict]) -> Findings:
    findings = Findings()
    for code, glossary in glossaries.items():
        _check_one(code, glossary, findings)
    _check_across(glossaries, findings)
    return findings


def _check_one(code: str, glossary: dict, findings: Findings) -> None:
    err, warn = findings.errors.append, findings.warnings.append

    if not _is_term_list(glossary.get("base_terms", [])):
        err(f"[{code}] base_terms must be a list of non-empty strings")
    sub_domains = glossary.get("sub_domains", {})
    if not isinstance(sub_domains, dict):
        err(f"[{code}] sub_domains must be an object")
        return

    anchor_owner: Dict[str, str] = {}
    for label, entry in sub_domains.items():
        if not _LABEL.match(label):
            err(f"[{code}] sub-domain label '{label}' is not UPPER_SNAKE_CASE")
        if not isinstance(entry, dict):
            warn(f"[{code}] {label}: a bare term list (no anchor_words), so it adds no Whisper hotwords")
            continue
        anchors, terms = entry.get("anchor_words", []), entry.get("terms", [])
        for name, values in (("anchor_words", anchors), ("terms", terms)):
            if not _is_term_list(values):
                err(f"[{code}] {label}.{name} must be a list of non-empty strings")
            for duplicate in _duplicates(values):
                err(f"[{code}] {label}.{name} lists '{duplicate}' twice")
        if not anchors:
            warn(f"[{code}] {label}: no anchor_words, so the router and Whisper see only its terms")
        if len(anchors) > MAX_ANCHORS:
            warn(f"[{code}] {label}: {len(anchors)} anchor words (> {MAX_ANCHORS}); the router reads 10 entries "
                 f"and all routed sub-domains share 40 hotwords")
        for anchor in anchors:
            key = _fold(anchor)
            if key in anchor_owner and anchor_owner[key] != label:
                err(f"[{code}] anchor '{anchor}' is in both {anchor_owner[key]} and {label}")
            anchor_owner[key] = label
            if key in {_fold(term) for term in terms}:
                err(f"[{code}] {label}: '{anchor}' is both an anchor word and a term")

    terms = listed_terms(glossary)
    details = glossary.get("term_details")
    if details is None:
        warn(f"[{code}] no term_details (hand-written glossary?): shared/forms can't be checked")
        details = {}
    else:
        for term in sorted(set(terms) - set(details)):
            err(f"[{code}] '{term}' has no term_details entry")
        for term, detail in details.items():
            forms = detail.get("forms", [])
            if not _is_term_list(forms):
                err(f"[{code}] term_details['{term}'].forms must be a list of non-empty strings")
            elif term in forms:
                err(f"[{code}] term_details['{term}'].forms repeats the term itself")

    if code in LATIN_ONLY:
        spoken = terms + [form for detail in details.values() for form in detail.get("forms", [])]
        for term in spoken:
            if isinstance(term, str) and _CYRILLIC.search(term):
                err(f"[{code}] '{term}' is in Cyrillic; {code} glossaries must be Latin script")

    listed = {_fold(term) for term in terms if isinstance(term, str)}
    for item in glossary.get("review", []):
        if _fold(item.get("term", "")) in listed:
            err(f"[{code}] '{item['term']}' is under review but also in the glossary")


def _check_across(glossaries: Dict[str, dict], findings: Findings) -> None:
    if len(glossaries) < 2:
        return
    labels = {code: set(g.get("sub_domains", {})) for code, g in glossaries.items()}
    every = set().union(*labels.values())
    for code, own in labels.items():
        for label in sorted(every - own):
            findings.errors.append(f"[{code}] sub-domain {label} is missing (labels must match in every language)")

    listed = {code: set(listed_terms(g)) for code, g in glossaries.items()}
    for code, glossary in glossaries.items():
        for term, detail in glossary.get("term_details", {}).items():
            if not detail.get("shared"):
                continue
            for other in glossaries:
                if other != code and term not in listed[other]:
                    findings.errors.append(f"[{code}] '{term}' is shared but [{other}] doesn't list it "
                                           f"with identical spelling")


def summary(glossaries: Dict[str, dict]) -> List[str]:
    lines = []
    for code, glossary in glossaries.items():
        sub_domains = glossary.get("sub_domains", {})
        details = glossary.get("term_details", {})
        shared = sum(1 for d in details.values() if d.get("shared"))
        with_forms = sum(1 for d in details.values() if d.get("forms"))
        lines.append(f"[{code}] {len(set(listed_terms(glossary)))} unique terms, "
                     f"{len(glossary.get('base_terms', []))} base, {len(sub_domains)} sub-domains, "
                     f"{shared} shared, {with_forms} with forms, {len(glossary.get('review', []))} under review")
        for label, entry in sub_domains.items():
            if isinstance(entry, dict):
                lines.append(f"    {label}: {len(entry.get('anchor_words', []))} anchors, "
                             f"{len(entry.get('terms', []))} terms")
    return lines


def _is_term_list(values) -> bool:
    return isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values)


def _fold(term: str) -> str:
    return " ".join(term.split()).casefold() if isinstance(term, str) else ""


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _duplicates(values) -> List[str]:
    if not isinstance(values, list):
        return []
    seen, repeated = set(), []
    for value in values:
        key = _fold(value)
        if key in seen and value not in repeated:
            repeated.append(value)
        seen.add(key)
    return repeated


def _read(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _report(glossaries: Dict[str, dict]) -> int:
    findings = check(glossaries)
    print("\n".join(summary(glossaries)))
    for warning in findings.warnings:
        print(f"WARNING {warning}")
    for error in findings.errors:
        print(f"ERROR   {error}")
    print(f"{len(findings.errors)} error(s), {len(findings.warnings)} warning(s)")
    return 1 if findings.errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("industry", nargs="?", help="industry key, i.e. the file name without .json")
    parser.add_argument("--languages", nargs="+", default=None, help="default: every served language")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="glossary root (holds <lang>/ folders)")
    parser.add_argument("--from-claude", type=Path, default=None, help="Claude's JSON answer to check/split")
    parser.add_argument("--write", action="store_true", help="with --from-claude: merge it into the glossary files")
    args = parser.parse_args(argv)

    if args.from_claude:
        glossaries = split_answer(json.loads(args.from_claude.read_text(encoding="utf-8")))
        status = _report(glossaries)
        if args.write:
            if status:
                print("Not written: fix the errors first.")
                return status
            for code, glossary in glossaries.items():
                path = args.dir / code / f"{glossary['industry'].lower()}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                merged = merge(_read(path) or {}, glossary)
                path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"Wrote {path}")
        return status

    if not args.industry:
        parser.error("give an industry, or --from-claude")
    glossaries = {}
    for code in args.languages or served_languages():
        glossary = _read(args.dir / code / f"{args.industry.lower()}.json")
        if glossary is None:
            print(f"ERROR   [{code}] no glossary at {args.dir / code / (args.industry.lower() + '.json')}")
            return 1
        glossaries[code] = glossary
    return _report(glossaries)


if __name__ == "__main__":
    sys.exit(main())
