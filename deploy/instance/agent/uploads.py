import re
from pathlib import Path

# Mirrors ALLOWED_EXTENSIONS in app/features/scanner/utils/filters.py. Copied, not imported: the agent runs in its
# own venv without the appliance's dependencies. A test parses filters.py to keep the two in step.
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".alac", ".aiff", ".amr", ".opus", ".mka",
}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx", ".csv", ".html", ".htm", ".rtf", ".epub"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
ALLOWED_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES | TEXT_SUFFIXES | IMAGE_SUFFIXES

_BATCH = re.compile(r"^\d{8}-\d{6}$")
_UNSAFE = re.compile(r"[^\w.\- ]", re.UNICODE)
MAX_NAME_LENGTH = 180


class UploadRejected(ValueError):
    pass


def sanitize_filename(raw: str) -> str:
    """Keeps the name the tester will recognise in the Ingest view, minus anything that could leave the folder."""
    name = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = _UNSAFE.sub("_", name).lstrip(".")
    suffix = Path(name).suffix.lower()
    if not name or not suffix:
        raise UploadRejected(f"'{raw}' has no file extension.")
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(f"'{suffix}' files can't be ingested.")
    if len(name) > MAX_NAME_LENGTH:
        name = Path(name).stem[: MAX_NAME_LENGTH - len(suffix)] + suffix
    return name


def upload_path(root: Path, language: str, batch: str, raw_filename: str) -> Path:
    if not _BATCH.match(batch):
        raise UploadRejected(f"'{batch}' is not a batch id.")
    folder = (root / language / batch).resolve()
    target = (folder / sanitize_filename(raw_filename)).resolve()
    if target.parent != folder or not folder.is_relative_to(root.resolve()):
        raise UploadRejected(f"'{raw_filename}' escapes the upload folder.")
    return target
