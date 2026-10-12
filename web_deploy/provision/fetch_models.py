"""
Fetches the appliance's models onto the GPU box from Hugging Face, instead of uploading 51 GB from home.

Every model that exists on Hugging Face is pinned here to the exact commit the dev box downloaded, and every file is
checked against models.lock.json (size and sha256 of the dev box's own copy), so the box runs byte-identical models.
What exists only on the dev box (the locally 4-bit-quantised Qwen2-VL, the NeMo VAD and TitaNet files, the RapidOCR
models: ~6 GB) still travels from there: web_deploy/provision/sync_to_box.sh ... --local-only.

Standard library only, so it runs on the box's system Python the moment the code is there, beside provision.sh:

  python3 /workspace/onyx_v3/web_deploy/provision/fetch_models.py              # fetch, resume, verify
  python3 /workspace/onyx_v3/web_deploy/provision/fetch_models.py --verify     # check what is there
  venv/bin/python -m web_deploy.provision.fetch_models --write-lock            # dev box: record the hashes

A download that stops resumes where it stopped (HTTP Range into a .part file); a file whose hash is wrong is fetched
again once, then reported.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
LOCK = HERE / "models.lock.json"
DEFAULT_MODELS_DIR = HERE.parents[1] / "models"
HASH_CHUNK = 8 * 1024 * 1024

# name under models/ -> (repo, commit, file inside the repo for a single file, or None for the whole repo)
HUGGING_FACE: Dict[str, Tuple[str, str, Optional[str]]] = {
    "Qwen3.5-9B-Q4_K_M.gguf": ("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5",
                               "Qwen3.5-9B-Q4_K_M.gguf"),
    "google_gemma-4-E4B-it-Q4_K_M.gguf": ("bartowski/google_gemma-4-E4B-it-GGUF",
                                          "022bbda30622c46d3467f567ad7c7064bfe37d0d",
                                          "google_gemma-4-E4B-it-Q4_K_M.gguf"),
    "llama-3.2-3b-instruct-q8_0.gguf": ("hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
                                        "7ef7efff7d2c14e5d6161a0c7006e1f2fea6ec79",
                                        "llama-3.2-3b-instruct-q8_0.gguf"),
    "qwen2.5-3b-instruct-q4_k_m.gguf": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "7dabda4d13d513e3e842b20f0d435c732f172cbe",
                                        "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "Qwen2.5-7B-Instruct-Q4_K_M.gguf": ("bartowski/Qwen2.5-7B-Instruct-GGUF",
                                        "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
                                        "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "bge-m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", None),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", None),
    "deberta-v3-base-zeroshot": ("cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7", None),
    "faster-whisper-large-v3": ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", None),
    "flanec-base-cd": ("morenolq/flanec-base-cd", "e3c8a1ea702dfb2e47ce77733a31123ea18dbe56", None),
    "fullstop-punctuation-multilingual-base": ("oliverguhr/fullstop-punctuation-multilingual-base",
                                               "fe99fb0dc8d8be674e8bb4e39833605f9a1af651", None),
    "Phi-3.5-mini-instruct": ("microsoft/Phi-3.5-mini-instruct", "2fe192450127e6a83f7441aef6e3ca586c338b77", None),
    "whisper-tiny-en": ("Systran/faster-whisper-tiny.en", "0d3d19a32d3338f10357c0889762bd8d64bbdeba", None),
}
# Only on the dev box: sent by sync_to_box.sh --local-only
LOCAL_ONLY = ["qwen2-vl-7b-hf-bnb4bit", "vad_multilingual_marblenet.nemo", "titanet_large.nemo", "rapidocr"]


# ---------- the lock: what every file must be ----------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def model_files(models_dir: Path, name: str) -> List[Path]:
    """A model's files under models/ (a single file, or a folder's files minus Hugging Face's own cache)."""
    root = models_dir / name
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)


def write_lock(models_dir: Path) -> dict:
    lock = {}
    for name in [*HUGGING_FACE, *LOCAL_ONLY]:
        files = {}
        for path in model_files(models_dir, name):
            files[str(path.relative_to(models_dir))] = {"size": path.stat().st_size, "sha256": sha256_of(path)}
            print(f"  {path.relative_to(models_dir)}", flush=True)
        if not files:
            raise SystemExit(f"models/{name} is missing on this machine.")
        lock[name] = files
    LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    return lock


def read_lock() -> dict:
    return json.loads(LOCK.read_text())


# ---------- fetching ----------

def url_for(name: str, relative: str) -> str:
    """The download address of one locked file: its path inside the repo, at the pinned commit."""
    repo, commit, single = HUGGING_FACE[name]
    inside = single if single else relative.split("/", 1)[1]
    return f"https://huggingface.co/{repo}/resolve/{commit}/{urllib.request.quote(inside)}"


def fetch_file(url: str, target: Path, size: int, attempts: int = 8) -> None:
    """Downloads to target.part, resuming with Range after a drop; renames once the size is right."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            partial.unlink()
            have = 0
        if have < size:
            request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if have and response.status != 206:
                        have = 0  # the server ignored the range: start over
                    with partial.open("ab" if have else "wb") as handle:
                        while True:
                            chunk = response.read(HASH_CHUNK)
                            if not chunk:
                                break
                            handle.write(chunk)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                if attempt == attempts:
                    raise RuntimeError(f"{url}: {exc}") from exc
                time.sleep(min(60, 2 ** attempt))
                continue
        if partial.stat().st_size == size:
            partial.replace(target)
            return
    raise RuntimeError(f"{url}: still {partial.stat().st_size} of {size} bytes after {attempts} attempts")


def is_good(path: Path, expected: dict) -> bool:
    return path.is_file() and path.stat().st_size == expected["size"] and sha256_of(path) == expected["sha256"]


def fetch(models_dir: Path, lock: dict, names: List[str], workers: int = 6) -> List[str]:
    """Fetches every locked file of `names` not already there and right. Returns the files still wrong."""
    jobs = [(name, relative, expected) for name in names for relative, expected in sorted(lock[name].items())]
    total = sum(e["size"] for _, _, e in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.1f} GB, from Hugging Face", flush=True)
    started = time.monotonic()

    def one(name: str, relative: str, expected: dict) -> Optional[str]:
        target = models_dir / relative
        if target.is_file() and target.stat().st_size == expected["size"] and sha256_of(target) == expected["sha256"]:
            return None
        for _ in range(2):  # a file with the wrong hash is fetched once more from scratch
            fetch_file(url_for(name, relative), target, expected["size"])
            if sha256_of(target) == expected["sha256"]:
                print(f"  ok  {relative} ({expected['size'] / 1e9:.2f} GB)", flush=True)
                return None
            target.unlink()
        return relative

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, *job): job[1] for job in jobs}
        for future in as_completed(futures):
            try:
                bad = future.result()
            except Exception as exc:
                print(f"  FAILED {futures[future]}: {exc}", flush=True)
                bad = futures[future]
            if bad:
                failed.append(bad)
    print(f"Done in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return failed


def verify(models_dir: Path, lock: dict, names: List[str]) -> List[str]:
    return [relative for name in names for relative, expected in sorted(lock[name].items())
            if not is_good(models_dir / relative, expected)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the appliance's models from Hugging Face, verified.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--write-lock", action="store_true", help="dev box: record every model file's hash")
    parser.add_argument("--verify", action="store_true", help="only check what is there, all models")
    parser.add_argument("--only", nargs="*", help="these models only (names under models/)")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    if args.write_lock:
        write_lock(args.models_dir)
        print(f"Wrote {LOCK}")
        return 0
    lock = read_lock()
    if args.verify:
        wrong = verify(args.models_dir, lock, list(lock))
        for relative in wrong:
            print(f"missing or different: {relative}")
        print("All models are present and identical." if not wrong else f"{len(wrong)} file(s) need fetching.")
        return 1 if wrong else 0
    names = args.only or list(HUGGING_FACE)
    unknown = [n for n in names if n not in HUGGING_FACE]
    if unknown:
        print(f"error: not on Hugging Face (send them with sync_to_box.sh --local-only): {', '.join(unknown)}",
              file=sys.stderr)
        return 1
    failed = fetch(args.models_dir, lock, names, args.workers)
    if failed:
        print(f"{len(failed)} file(s) could not be fetched or did not match; run this again.", file=sys.stderr)
        return 1
    missing = [n for n in LOCAL_ONLY if verify(args.models_dir, lock, [n])]
    if missing:
        print(f"Still to come from the dev box: {', '.join(missing)}  (sync_to_box.sh <target> <port> --local-only)")
    else:
        print("Every model is here and identical to the dev box's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
