# Path: scripts/download_ocr_models.py

"""
Fetches the RapidOCR models document ingestion uses into DOC_OCR_MODEL_DIR, once, at build time.

The appliance never downloads anything at runtime: RapidOCR fetches missing models from modelscope.cn on first
use unless it is handed explicit model paths, so the OCR adapter always passes paths into this directory and
treats missing files as "this appliance has no OCR". URLs and SHA256 sums come from the installed rapidocr's own
default_models.yaml, so they stay in step with the pinned rapidocr version.

    venv/bin/python -m scripts.download_ocr_models
"""

import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

import rapidocr
from app.core.config import settings
from app.features.document_conversion.data.rapidocr_engine import OCR_MODEL_FILES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _catalogue() -> dict:
    """{file name: (url, sha256)} for every onnxruntime model rapidocr knows about."""
    with open(Path(rapidocr.__file__).parent / "default_models.yaml", encoding="utf-8") as f:
        models = yaml.safe_load(f)["onnxruntime"]
    catalogue = {}
    for version in models.values():
        for task in version.values():
            for info in task.values():
                url = info["model_dir"]
                catalogue[url.rsplit("/", 1)[-1]] = (url, info["SHA256"])
    return catalogue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    target = settings.BASE_DIR / settings.DOC_OCR_MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)
    catalogue = _catalogue()

    for name in sorted(set(OCR_MODEL_FILES.values())):
        url, expected = catalogue[name]
        path = target / name
        if path.exists() and _sha256(path) == expected:
            logger.info(f"{name}: already present")
            continue

        logger.info(f"{name}: downloading {url}")
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink()
            logger.error(f"{name}: SHA256 mismatch (expected {expected}, got {actual}); nothing installed")
            return 1
        partial.replace(path)
        logger.info(f"{name}: verified and installed")

    logger.info(f"OCR models ready in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
