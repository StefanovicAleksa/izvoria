# Path: app/core/language/data/toml_pack_repository.py

import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.language.domain.entities import LanguagePack
from app.core.language.domain.interfaces import ILanguagePackRepository
from app.core.language.exceptions import LanguagePackNotFoundError
from app.core.language.utils.error_mapper import map_language_error


class TomlLanguagePackRepository(ILanguagePackRepository):
    """
    Reads `<code>.toml` packs from one directory (`data/languages/` in production).

    TOML rather than JSON so a pack can carry a comment next to every rule saying why it is there, and rather
    than YAML because `tomllib` ships with Python: the appliance has no package manager to install a parser.
    """

    def __init__(self, packs_dir: Path):
        self._packs_dir = Path(packs_dir)

    def load(self, code: str) -> LanguagePack:
        path = self._packs_dir / f"{code}.toml"
        try:
            if not path.is_file():
                raise LanguagePackNotFoundError(
                    f"No language pack for '{code}' in {self._packs_dir}. "
                    f"Available: {', '.join(self.available_codes()) or 'none'}."
                )
            with open(path, "rb") as f:
                data = tomllib.load(f)
            data.setdefault("text", {})["stopwords"] = self._resolve_stopwords(data.get("text", {}))
            return LanguagePack.model_validate(data)
        except Exception as e:
            raise map_language_error(e, context=str(path)) from e

    def available_codes(self) -> List[str]:
        if not self._packs_dir.is_dir():
            return []
        return sorted(path.stem for path in self._packs_dir.glob("*.toml"))

    @staticmethod
    def _resolve_stopwords(text_table: Dict[str, Any]) -> Tuple[str, ...]:
        """The pack's inline stopwords plus, when it names one, an NLTK stopword corpus."""
        words = list(text_table.get("stopwords", []))
        corpus = text_table.pop("nltk_stopwords", None)
        if corpus:
            from nltk.corpus import stopwords
            words.extend(stopwords.words(corpus))
        return tuple(sorted({word.lower() for word in words}))
