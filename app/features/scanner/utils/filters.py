import os
import re
from pathlib import Path
from typing import Iterable

from app.core.config import settings

# Izvoria Supported Extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}
# Every format the audio pipeline decodes: a suffix missing here is dropped by the scan before any pipeline sees it
AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma', '.alac', '.aiff', '.amr', '.opus', '.mka'
}
TEXT_EXTENSIONS = {'.txt', '.md', '.markdown', '.pdf', '.docx', '.csv', '.html', '.htm', '.rtf', '.epub'}

# Still images. No HEIC: reading it needs pillow-heif, a native dependency the appliance doesn't ship.
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}

ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | TEXT_EXTENSIONS | IMAGE_EXTENSIONS

IGNORED_DIRS = {'.git', '__pycache__', '.vscode', '.idea', 'venv', 'env', 'node_modules'}

def is_extension_allowed(file_path: Path) -> bool:
    """Checks if the file has a supported extension."""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS

def is_directory_ignored(dir_name: str) -> bool:
    """Checks if the directory should be skipped."""
    return dir_name in IGNORED_DIRS

_YEAR_SHARD = re.compile(r"^\d{4}$")


def storage_root() -> Path:
    """The vault's root, resolved exactly as storage's FileManager resolves it."""
    return Path(os.getenv("ONYX_STORAGE_PATH", "onyx_data")).resolve()


def izvoria_output_dirs() -> Iterable[Path]:
    """Directories Izvoria writes its own files into; never ingested back."""
    return (settings.REPORTS_DIR.resolve(),)


def is_onyx_output(path: Path) -> bool:
    """
    A directory holding Izvoria's own output rather than the client's material: the vault's date shards
    (`<storage root>/YYYY/...`: video frames, clips, extracted audio, copies of ingested files) and the finished
    reports. Ingesting them would index frames as images, clips as videos, and -- worst -- the model-written
    reports as source documents, citing Izvoria's own prose as evidence. Anything else under the storage root (the
    language benchmark's media) is still scanned.
    """
    resolved = path.resolve()
    if any(resolved == root or root in resolved.parents for root in izvoria_output_dirs()):
        return True
    try:
        relative = resolved.relative_to(storage_root())
    except ValueError:
        return False
    return bool(relative.parts) and bool(_YEAR_SHARD.match(relative.parts[0]))
