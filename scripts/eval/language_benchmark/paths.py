# Path: scripts/eval/language_benchmark/paths.py

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = REPO_ROOT / "onyx_data" / "benchmarks"


@dataclass(frozen=True)
class LanguageDirs:
    """Where one language's benchmark lives: raw downloads, ingestible media, references, results."""
    code: str

    @property
    def root(self) -> Path:
        return BENCH_ROOT / self.code

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def refs(self) -> Path:
        return self.root / "refs"

    @property
    def vault(self) -> Path:
        return self.root / "vault"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def labels(self) -> Path:
        return self.root / "labels.json"

    def make(self) -> "LanguageDirs":
        for path in (self.raw, self.media, self.refs, self.vault, self.results):
            path.mkdir(parents=True, exist_ok=True)
        return self


def database_name(code: str) -> str:
    """Each language is ingested into its own database, so its numbers are measured on its material alone."""
    return f"onyx_bench_{code}"


def served_languages() -> list:
    """
    The languages the appliance serves: one per pack in data/languages/ (read from disk, so nothing reads
    `settings` before a stage's child process sets its environment). Legacy packs (legacy/) are not measured.
    """
    return sorted(path.stem for path in (REPO_ROOT / "data" / "languages").glob("*.toml"))
