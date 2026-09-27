# Path: tests/features/qa_generation/data/test_prompts.py

from app.features.deep_research_report.data.prompts import (
    OUTLINE_SYSTEM_PROMPT, SECTION_DOCUMENTS_SYSTEM_PROMPT, SECTION_SYSTEM_PROMPT, outline_system_prompt,
)
from app.features.qa_generation.data.prompts import QA_DOCUMENTS_SYSTEM_PROMPT, QA_SYSTEM_PROMPT, qa_system_prompt
from app.features.qa_reasoning.data.prompts import REASONER_DOCUMENTS_SYSTEM_PROMPT, REASONER_SYSTEM_PROMPT


class TestImageExcerptsInPrompts:
    def test_recordings_only_prompts_never_mention_images(self):
        for prompt in (QA_SYSTEM_PROMPT, REASONER_SYSTEM_PROMPT, SECTION_SYSTEM_PROMPT, OUTLINE_SYSTEM_PROMPT):
            assert "image" not in prompt.lower()
        assert qa_system_prompt(False) is QA_SYSTEM_PROMPT

    def test_answer_prompt_says_an_image_excerpt_is_not_the_clients_wording(self):
        assert qa_system_prompt(True) is QA_DOCUMENTS_SYSTEM_PROMPT
        assert "There are three kinds" in QA_DOCUMENTS_SYSTEM_PROMPT
        assert "machine-written description of a picture" in QA_DOCUMENTS_SYSTEM_PROMPT
        assert '"the image shows"' in QA_DOCUMENTS_SYSTEM_PROMPT
        # The first line no longer promises that every block is verbatim
        assert "Each context block is a verbatim excerpt" not in QA_DOCUMENTS_SYSTEM_PROMPT

    def test_reasoner_and_report_prompts_describe_image_excerpts(self):
        assert "machine-written description of the picture" in REASONER_DOCUMENTS_SYSTEM_PROMPT
        assert "machine-written description of a picture" in SECTION_DOCUMENTS_SYSTEM_PROMPT
        assert '"image"' in outline_system_prompt(True)
