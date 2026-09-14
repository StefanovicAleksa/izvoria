# Path: tests/features/rag_retrieval/domain/test_entities.py

import pytest
from pydantic import ValidationError

from app.features.rag_retrieval.domain.entities import SearchQuery


class TestSearchQuery:
    @pytest.mark.parametrize("blank_prompt", ["", "   ", "\n\t"])
    def test_blank_prompt_is_rejected(self, blank_prompt):
        with pytest.raises(ValidationError):
            SearchQuery(prompt=blank_prompt)

    def test_prompt_is_stripped(self):
        assert SearchQuery(prompt="  what did the witness say?  ").prompt == "what did the witness say?"
