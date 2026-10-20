# Path: tests/features/source_profile/utils/test_grounding.py

from app.features.source_profile.utils.grounding import OPENING_CHARS, dropped_numbers, invented_numbers

LECTURE_20 = "Lecture 20 Multinomial and Cauchy Statistics 110 [xiVWNkQUqKk].mp4"


class TestInventedNumbers:
    def test_a_title_copied_from_another_lecture_is_caught(self):
        assert invented_numbers("Statistics 110, Lecture 4", [LECTURE_20], ["we saw 4 cases"]) == ["4"]

    def test_the_file_names_numbers_are_allowed(self):
        assert invented_numbers("Statistics 110, Lecture 20", [LECTURE_20], []) == []

    def test_numbers_in_a_numbered_file_name_rule_over_the_transcript(self):
        assert invented_numbers("Unit 7, Lecture 4", ["UNIT 7 – FOMC.mp4"], ["lecture 4 of the course"]) == ["4"]

    def test_a_name_without_numbers_lets_the_opening_supply_them(self):
        assert invented_numbers("Contract Law, Module 3", ["intro.mp4"], ["Welcome to module 3 of contract law"]) == []

    def test_only_the_opening_counts(self):
        texts = ["x " * OPENING_CHARS + "module 9"]
        assert invented_numbers("Module 9", ["intro.mp4"], texts) == ["9"]

    def test_leading_zeros_compare_as_numbers(self):
        assert invented_numbers("Module 1", ["module_01.mp4"], []) == []

    def test_a_title_without_numbers_is_fine(self):
        assert invented_numbers("Bank of Canada Meeting", [LECTURE_20], []) == []


class TestDroppedNumbers:
    def test_a_title_without_the_lecture_number_drops_it(self):
        assert dropped_numbers("Exponential Distribution Statistics 110",
                               ["Lecture 16 Exponential Distribution Statistics 110 [bM6nFDjvEns].mp4"]) == ["16"]

    def test_a_title_keeping_every_number(self):
        assert dropped_numbers("Statistics 110, Lecture 16", ["Lecture 16 Statistics 110 [bM6nFDjvEns].mp4"]) == []

    def test_the_youtube_id_and_extension_are_not_numbers(self):
        assert dropped_numbers("Talk", ["Talk [12345678-ab].mp4"]) == []

    def test_a_name_without_numbers_drops_nothing(self):
        assert dropped_numbers("Bank of Canada", ["Course Status.mp4"]) == []
