# Path: tests/features/image_description/utils/test_text_verification.py

import pytest

from app.features.image_description.utils.text_verification import (
    UNCONFIRMED_NUMBER, UNCONFIRMED_TEXT, OcrReading, strip_unconfirmed, verify_items,
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
