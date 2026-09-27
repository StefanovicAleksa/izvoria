# Path: tests/features/image_description/utils/test_text_verification.py

import pytest

from app.features.image_description.utils.text_verification import (
    UNCONFIRMED_NUMBER, UNCONFIRMED_TEXT, OcrReading, select_ocr_lines, strip_unconfirmed, verify_items,
)

OCR = "Q3 LAUNCH\nBudget review - 14 March 2025\nOwner: Ana Kovač\nRoom 204"


class TestOcrReadingConfirms:
    @pytest.mark.parametrize("item", [
        "Q3 launch",                       # case differs
        "Budget review – 14 March 2025",   # dash differs
        "Owner: Ana Kovač",
        "Room 204",
        "204",
        "March",
    ])
    def test_text_ocr_also_read_is_confirmed(self, item):
        assert OcrReading(OCR).confirms(item, 0.8)

    @pytest.mark.parametrize("item", [
        "Room 205",                # a number OCR didn't read is never confirmed, whatever the words
        "Budget review 15 March 2025",
        "Owner: Marko Petrović",   # most words missing
        "Confidential",            # one word, not there
        "",
    ])
    def test_text_ocr_did_not_read_is_not_confirmed(self, item):
        assert not OcrReading(OCR).confirms(item, 0.8)

    def test_coverage_threshold_allows_one_misread_word_in_five(self):
        reading = OcrReading("the quick brown fox jumps")

        assert reading.confirms("the quick brown fax jumps", 0.8)
        assert not reading.confirms("the quick brown fax jumps", 0.9)

    def test_serbian_cyrillic_matches_latin_ocr_in_the_packs_script(self):
        reading = OcrReading("Ugovor o zakupu", script="latin")

        assert reading.confirms("Уговор о закупу", 0.8)

    def test_empty_ocr_confirms_nothing(self):
        assert not OcrReading("").confirms("anything", 0.8)


class TestVerifyItems:
    def test_splits_in_the_vision_models_order(self):
        confirmed, dropped = verify_items(["Room 204", "Room 999", "Q3 launch"], OcrReading(OCR), 0.8)

        assert confirmed == ["Room 204", "Q3 launch"]
        assert dropped == ["Room 999"]


class TestStripUnconfirmed:
    def test_unconfirmed_quotes_and_numbers_are_marked_not_silently_removed(self):
        phrase = 'A whiteboard reading "Q3 launch" and "Go live 2026", with room 204 and 17 sticky notes'

        stripped = strip_unconfirmed(phrase, OcrReading(OCR), 0.8)

        assert stripped == (f'A whiteboard reading "Q3 launch" and {UNCONFIRMED_TEXT}, with room 204 and '
                            f'{UNCONFIRMED_NUMBER} sticky notes')

    def test_single_digits_and_plain_words_are_left_alone(self):
        assert strip_unconfirmed("Three people around 1 table", OcrReading(""), 0.8) == "Three people around 1 table"

    def test_none_stays_none(self):
        assert strip_unconfirmed(None, OcrReading(OCR), 0.8) is None


def _select(lines, vision_text=""):
    return select_ocr_lines(lines, OcrReading(vision_text), min_coverage=0.8, min_confidence=0.7,
                            unreliable_share=0.5, unreliable_min_lines=5, unreliable_confidence=0.9)


class TestSelectOcrLines:
    """Scores are real RapidOCR scores from the images that exposed the problem."""

    def test_confident_lines_stand_alone(self):
        assert _select([("How to reach Resorts on this Line", 1.0), ("Hartf d", 0.79)]) == [True, True]

    def test_low_confidence_misreads_nothing_confirms_are_withheld(self):
        assert _select([("ATTENDANCE", 0.99), ("-", 0.19), ("585", 0.37)]) == [True, False, False]

    def test_low_confidence_text_the_vision_model_also_read_is_kept(self):
        lines = [("STOP", 1.0), ("GIMAr", 0.71), ("MOntaz", 0.65), ("MAutino", 0.76)]
        assert _select(lines, "STOP\nGIMAR MONTAZ MAUTINO") == [True, True, True, True]

    def test_on_handwriting_even_middling_lines_need_confirmation(self):
        # Most lines under 0.7: OCR is guessing throughout, so 0.8 no longer stands alone
        lines = [("Analbeig aftea long apok", 0.64), ("bank of amorth teopt", 0.61), ("ansholy eerTunking", 0.58),
                 ("to he wotheyp", 0.66), ("ne wondep the eg", 0.63), ("tih. I had Me pelieily", 0.81)]
        assert _select(lines) == [False] * 6

    def test_a_few_low_lines_do_not_make_a_printed_page_unreliable(self):
        lines = [("Line one", 0.99), ("Line two", 0.98), ("E", 0.12), ("Line four", 0.99), ("Line five", 0.8)]
        assert _select(lines) == [True, True, False, True, True]
