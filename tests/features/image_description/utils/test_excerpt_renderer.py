# Path: tests/features/image_description/utils/test_excerpt_renderer.py

import uuid

import pytest

from app.features.image_description.domain.entities import ImageDescription
from app.features.image_description.utils.excerpt_renderer import render_context, render_excerpt


def _description(**fields):
    return ImageDescription(source_id=uuid.uuid4(), file_name="board.jpg", **fields)


class TestRenderExcerpt:
    def test_description_then_ocr_text(self, language_pack):
        excerpt = render_excerpt(_description(
            shows="a photographed whiteboard with a timeline", marks='an arrow at "Q3"',
            ocr_text="Q3 launch\nBudget review", ocr_checked=True, verified_items=["Q3 launch"],
        ), language_pack("en").images)

        assert excerpt == ('Image (machine description): A photographed whiteboard with a timeline. '
                           'Marked: an arrow at "Q3".\n'
                           'Text on the image (read by OCR):\nQ3 launch\nBudget review')

    def test_without_ocr_the_vision_models_text_is_labelled_unverified(self, language_pack):
        excerpt = render_excerpt(_description(shows="Skenirani ugovor", unverified_items=["Ugovor o zakupu", "2025"]),
                                 language_pack("sr").images)

        assert excerpt == ('Slika (mašinski opis): Skenirani ugovor.\n'
                           'Tekst koji je pročitao model za vid (neprovereno): "Ugovor o zakupu"; "2025"')

    def test_text_only_image_has_no_description_line(self, language_pack):
        excerpt = render_excerpt(_description(ocr_text="EXIT", ocr_checked=True), language_pack("sl").images)

        assert excerpt == "Besedilo na sliki (prebral OCR):\nEXIT"

    @pytest.mark.parametrize("code", ["en", "sr", "sl"])
    def test_every_pack_labels_the_description_as_machine_written(self, language_pack, code):
        excerpt = render_excerpt(_description(shows="a dog"), language_pack(code).images)

        assert excerpt.startswith(language_pack(code).images.description_label + ": A dog.")


class TestWithheldText:
    def test_an_image_whose_handwriting_was_withheld_says_so(self, language_pack):
        excerpt = render_excerpt(_description(shows="a handwritten letter", ocr_checked=True,
                                              ocr_withheld_lines=["Analbeig aftea long apok"]),
                                 language_pack("en").images)

        assert excerpt == ("Image (machine description): A handwritten letter.\n"
                           "Some text on this image could not be read reliably and is not included.")
        assert "Analbeig" not in excerpt

    def test_withheld_symbols_alone_are_not_missing_text(self, language_pack):
        excerpt = render_excerpt(_description(shows="a bar chart", ocr_text="Total Male Female", ocr_checked=True,
                                              ocr_withheld_lines=["■", "-"]),
                                 language_pack("en").images)

        assert "could not be read" not in excerpt

    @pytest.mark.parametrize("code", ["en", "sr", "sl"])
    def test_every_pack_has_the_note(self, language_pack, code):
        assert language_pack(code).images.withheld_text_note


class TestRenderContext:
    def test_name_date_and_size(self):
        assert render_context(_description(taken_at="2025-03-14 10:02", width=4032, height=3024)) == \
            "board.jpg · 2025-03-14 10:02 · 4032×3024"

    def test_only_what_is_known(self):
        assert render_context(_description()) == "board.jpg"
