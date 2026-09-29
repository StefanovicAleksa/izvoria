# Path: app/features/rag_retrieval/utils/sparse_query.py

"""
The keyword half of hybrid search, across languages.

One index holds every language, each chunk stemmed by its own text search configuration. A question is therefore
turned into one query per configuration, and each chunk is matched only against the query built with its own:
an English question still finds an English chunk by its English stems, and the same words find a Serbian chunk
by Serbian stems if they happen to be Serbian.

A configuration that doesn't stem (`simple`, for a language Postgres has no stemmer for; the legacy Slovenian pack
used it) also keeps stop words, so "the", "a" or "je" from a question in any language would match nearly every
chunk it indexed. Those configurations get the question
without any shipped language's stop words. Stemmed configurations drop their own stop words already.
"""

import re
from functools import lru_cache
from typing import FrozenSet, List, Tuple

from app.core.config import settings
from app.core.language.service.language_api import get_language_pack

# Postgres' text search configurations that neither stem nor drop stop words
UNSTEMMED_CONFIGS = frozenset({"simple"})

_WORD = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=8)
def _all_stopwords(codes: Tuple[str, ...]) -> FrozenSet[str]:
    return frozenset(word.lower() for code in codes for word in get_language_pack(code).text.stopwords)


def _without_stopwords(query_text: str) -> str:
    stopwords = _all_stopwords(tuple(settings.supported_languages()))
    return " ".join(word for word in _WORD.findall(query_text) if word.lower() not in stopwords)


def sparse_query_texts(query_text: str) -> Tuple[List[str], List[str]]:
    """Every text search configuration in the index, and the text to build each one's query from, aligned."""
    configs = settings.SEARCH_FTS_CONFIGS
    texts = [_without_stopwords(query_text) if config in UNSTEMMED_CONFIGS else query_text for config in configs]
    return configs, texts
