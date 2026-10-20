# Path: tests/features/source_profile/utils/test_profile_text.py

import pytest

from app.features.source_profile.domain.entities import MAX_TITLE_CHARS, MAX_TOPIC_CHARS
from app.features.source_profile.utils.profile_text import clip, clipped_draft, first_sentence, title_from_name


class TestTitleFromName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Lecture 4: Conditional Probability | Statistics 110 [P7NE4WF9j-Q].mp4",
             "Lecture 4: Conditional Probability | Statistics 110"),
            ("module_01.mp4", "module 01"),
            ("contract_final_v3.pdf", "contract final v3"),
            ("Board meeting 3 May", "Board meeting 3 May"),
            ("notes.backup.txt", "notes backup"),
            ("", "Untitled"),
        ]
    )
    def test_title_from_name(self, name, expected):
        assert title_from_name(name) == expected

    def test_long_names_are_cut_to_the_title_limit(self):
        assert len(title_from_name("word " * 100 + ".mp4")) <= MAX_TITLE_CHARS


class TestClip:
    def test_cuts_at_a_word_boundary(self):
        assert clip("alpha beta gamma", 12) == "alpha beta"

    def test_short_text_is_unchanged_but_one_line(self):
        assert clip("alpha\n beta", 50) == "alpha beta"

    def test_clipped_draft_never_rejects_an_overlong_reply(self):
        draft = clipped_draft("t " * 200, "x " * 400)
        assert len(draft.title) <= MAX_TITLE_CHARS and len(draft.topic) <= MAX_TOPIC_CHARS


class TestFirstSentence:
    def test_takes_the_first_sentence(self):
        assert first_sentence("A whiteboard with a formula. Two people stand.") == "A whiteboard with a formula."

    def test_without_punctuation_the_whole_text(self):
        assert first_sentence("a photo of a car") == "a photo of a car"


class TestTidyTitle:
    def test_a_title_copied_from_the_file_name_loses_extension_and_id(self):
        draft = clipped_draft("Lecture 28 Inequalities Statistics 110 [UtXK_EQ3Pow].mp4", "")
        assert draft.title == "Lecture 28 Inequalities Statistics 110"

    def test_a_version_number_is_not_an_extension(self):
        assert clipped_draft("Policy v2.5", "").title == "Policy v2.5"
