# Path: tests/features/document_conversion/data/test_rapidocr_engine.py

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.features.document_conversion.data.rapidocr_engine import (
    CLASSIFIER_FILE, DETECTOR_FILE, RECOGNIZER_FILES, RapidOcrEngine,
)
from app.features.document_conversion.exceptions import OcrModelsMissingError


def _box(x, y, width=200, height=30):
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=float)


@pytest.fixture
def model_dir(tmp_path):
    for name in (DETECTOR_FILE, CLASSIFIER_FILE, RECOGNIZER_FILES["latin"]):
        (tmp_path / name).write_bytes(b"onnx")
    return tmp_path


class TestModelFiles:
    def test_missing_files_raise_instead_of_downloading(self, tmp_path):
        engine = RapidOcrEngine(str(tmp_path), "latin")

        with pytest.raises(OcrModelsMissingError, match="download_ocr_models"):
            engine.recognize(Image.new("RGB", (10, 10)))

    def test_unknown_recognition_model_raises(self, model_dir):
        with pytest.raises(OcrModelsMissingError):
            RapidOcrEngine(str(model_dir), "cyrillic").recognize(Image.new("RGB", (10, 10)))

    @patch("rapidocr.RapidOCR")
    def test_every_model_path_is_explicit_and_no_line_is_score_filtered(self, rapidocr_cls, model_dir):
        rapidocr_cls.return_value.return_value = SimpleNamespace(boxes=None)

        RapidOcrEngine(str(model_dir), "latin").recognize(Image.new("RGB", (10, 10)))

        params = rapidocr_cls.call_args.kwargs["params"]
        assert params["Det.model_path"] == str(model_dir / DETECTOR_FILE)
        assert params["Cls.model_path"] == str(model_dir / CLASSIFIER_FILE)
        assert params["Rec.model_path"] == str(model_dir / RECOGNIZER_FILES["latin"])
        assert params["Global.text_score"] == 0.0


def _words(*items):
    """RapidOCR word results for one line: (text, score, four corner points)."""
    return tuple((text, 0.99, _box(x, y, width).tolist()) for text, x, y, width in items)


class TestRecognize:
    @patch("rapidocr.RapidOCR")
    def test_counts_detected_lines_separately_from_lines_read(self, rapidocr_cls, model_dir):
        ocr = rapidocr_cls.return_value
        detection = SimpleNamespace(boxes=[_box(0, 0), _box(0, 40), _box(0, 80)])
        # RapidOCR dropped the middle line, whose recognition came back empty
        recognition = SimpleNamespace(boxes=[_box(0, 0), _box(0, 80)], txts=("First line", "Third line"),
                                      scores=(0.99, 0.95), word_results=None)
        ocr.side_effect = [detection, recognition]

        page = RapidOcrEngine(str(model_dir), "latin").recognize(Image.new("RGB", (10, 10)))

        assert (page.lines_detected, page.lines_read) == (3, 2)
        # The dropped line leaves a gap wide enough to read as a paragraph break
        assert page.text == "First line\n\nThird line"
        assert page.mean_confidence == 0.97
        # RapidOCR remembers use_* flags between calls, so both calls must name all three
        assert ocr.call_args_list[0].kwargs == {"use_det": True, "use_cls": False, "use_rec": False}
        assert ocr.call_args_list[1].kwargs == {"use_det": True, "use_cls": True, "use_rec": True,
                                                "return_word_box": True}

    @patch("rapidocr.RapidOCR")
    def test_word_boxes_let_a_scanned_table_keep_its_rows(self, rapidocr_cls, model_dir):
        """The detector boxed each row as one line; its word boxes still show the columns."""
        rows = [("Item", "Qty", "Amount"), ("Parking", "2", "80.00"), ("Cleaning", "4", "100.00")]
        ocr = rapidocr_cls.return_value
        line_boxes = [_box(0, 40 * i, width=500) for i in range(3)]
        words = [_words((a, 0, 40 * i, 100), (b, 250, 40 * i, 40), (c, 400, 40 * i, 100))
                 for i, (a, b, c) in enumerate(rows)]
        ocr.side_effect = [SimpleNamespace(boxes=line_boxes),
                           SimpleNamespace(boxes=line_boxes, txts=tuple(" ".join(r) for r in rows),
                                           scores=(0.99,) * 3, word_results=tuple(words))]

        page = RapidOcrEngine(str(model_dir), "latin").recognize(Image.new("RGB", (10, 10)))

        assert page.text == ("| Item | Qty | Amount |\n| --- | --- | --- |\n"
                             "| Parking | 2 | 80.00 |\n| Cleaning | 4 | 100.00 |")
        assert (page.lines_detected, page.lines_read) == (3, 3)

    @patch("rapidocr.RapidOCR")
    def test_page_with_no_detected_text_skips_recognition(self, rapidocr_cls, model_dir):
        rapidocr_cls.return_value.return_value = SimpleNamespace(boxes=None)

        page = RapidOcrEngine(str(model_dir), "latin").recognize(Image.new("RGB", (10, 10)))

        assert (page.text, page.lines_detected, page.lines_read) == ("", 0, 0)
        assert rapidocr_cls.return_value.call_count == 1

    @patch("rapidocr.RapidOCR")
    def test_models_load_once_and_close_releases_them(self, rapidocr_cls, model_dir):
        rapidocr_cls.return_value.return_value = SimpleNamespace(boxes=None)
        engine = RapidOcrEngine(str(model_dir), "latin")

        engine.recognize(Image.new("RGB", (10, 10)))
        engine.recognize(Image.new("RGB", (10, 10)))
        engine.close()
        engine.recognize(Image.new("RGB", (10, 10)))

        assert rapidocr_cls.call_count == 2
