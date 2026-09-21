# Path: tests/orchestration/utils/test_follow_up.py

import pytest

from app.orchestration.utils.follow_up import answer_question


class TestAnswerQuestion:
    @pytest.mark.parametrize("standalone", [None, "", "Who voted?", "  Who voted? "])
    def test_question_is_unchanged_without_a_different_standalone_form(self, standalone):
        assert answer_question("Who voted?", standalone) == "Who voted?"

    def test_follow_up_carries_its_standalone_form_after_the_users_words(self):
        assert answer_question("And then?", "What happened after the vote?") == (
            "And then?\n(Standalone form of this follow-up question: What happened after the vote?)"
        )
