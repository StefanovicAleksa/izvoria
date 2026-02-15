from pathlib import Path

# Onyx Supported Extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
TEXT_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.csv'}

ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | TEXT_EXTENSIONS

IGNORED_DIRS = {'.git', '__pycache__', '.vscode', '.idea', 'venv', 'env', 'node_modules'}

def is_extension_allowed(file_path: Path) -> bool:
    """Checks if the file has a supported extension."""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS

def is_directory_ignored(dir_name: str) -> bool:
    """Checks if the directory should be skipped."""
    return dir_name in IGNORED_DIRS