# Path: app/features/document_conversion/data/rapidocr_engine.py

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from app.features.document_conversion.domain.entities import OcrPage
from app.features.document_conversion.domain.interfaces import IOcrEngine, LineSelector
from app.features.document_conversion.exceptions import OcrModelsMissingError
from app.features.document_conversion.utils.layout import Box, render_page

logger = logging.getLogger(__name__)

# The files scripts/download_ocr_models.py fetches. PP-OCRv5's detector and the text-direction classifier are
# language-independent; the recognition model is chosen by the language pack's `documents.ocr_rec_model`.
DETECTOR_FILE = "ch_PP-OCRv5_det_mobile.onnx"
CLASSIFIER_FILE = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
RECOGNIZER_FILES = {
    # Its alphabet covers English and Slovenian (č š ž, and ć đ for names). It has no Cyrillic, which is why the
    # Serbian pack names no model.
    "latin": "latin_PP-OCRv5_rec_mobile.onnx",
}
OCR_MODEL_FILES = {"det": DETECTOR_FILE, "cls": CLASSIFIER_FILE, **RECOGNIZER_FILES}



class RapidOcrEngine(IOcrEngine):
    """
    RapidOCR (PP-OCRv5 on onnxruntime, CPU).

    Two things about RapidOCR decide how this adapter is written:
      * With no explicit model paths it downloads its models from modelscope.cn on first use. Every path is
        given here, and a missing file raises `OcrModelsMissingError` instead.
      * It silently drops every line whose recognised text is empty (rapidocr/main.py). A detection-only pass
        counts the lines first, so a page that lost lines is reported as incomplete rather than looking whole.
    """

    def __init__(self, model_dir: str, rec_model: str):
        self._model_dir = Path(model_dir)
        self._rec_model = rec_model
        self._engine = None

    def _paths(self) -> dict:
        if self._rec_model not in RECOGNIZER_FILES:
            raise OcrModelsMissingError(f"No OCR recognition model named {self._rec_model!r}.")
        paths = {
            "det": self._model_dir / DETECTOR_FILE,
            "cls": self._model_dir / CLASSIFIER_FILE,
            "rec": self._model_dir / RECOGNIZER_FILES[self._rec_model],
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise OcrModelsMissingError(
                f"OCR model files missing: {', '.join(missing)}. Run `python -m scripts.download_ocr_models`."
            )
        return paths

    def _load(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            paths = self._paths()
            self._engine = RapidOCR(params={
                # Keep every recognised line; confidence is reported, not used to discard text
                "Global.text_score": 0.0,
                "Global.log_level": "warning",
                "Det.model_path": str(paths["det"]),
                "Cls.model_path": str(paths["cls"]),
                "Rec.model_path": str(paths["rec"]),
            })
            logger.info(f"Loaded RapidOCR ({self._rec_model}) from {self._model_dir}")
        return self._engine

    def recognize(self, image: Image.Image, select_lines: Optional[LineSelector] = None) -> OcrPage:
        engine = self._load()
        pixels = np.asarray(image.convert("RGB"))

        detected = engine(pixels, use_det=True, use_cls=False, use_rec=False)
        lines_detected = 0 if detected.boxes is None else len(detected.boxes)
        if lines_detected == 0:
            return OcrPage()

        # RapidOCR remembers the use_* flags of the previous call, so both calls name all three. Word boxes let
        # a scanned table be found the same way as a born-digital one: the detector often boxes two neighbouring
        # cells as one line
        result = engine(pixels, use_det=True, use_cls=True, use_rec=True, return_word_box=True)
        boxes = [] if result.boxes is None else [np.asarray(box) for box in result.boxes]
        texts = list(result.txts or [])
        scores = list(result.scores or [])
        words = list(result.word_results or [None] * len(texts))

        read = [(box, text, line_words, float(score))
                for box, text, line_words, score in zip(boxes, texts, words, scores or [1.0] * len(texts))
                if text.strip()]
        keep = select_lines([(text, score) for _, text, _, score in read]) if select_lines else [True] * len(read)

        page_boxes: List[Box] = []
        withheld: List[str] = []
        lines_read = len(read)
        for (box, text, line_words, _), kept in zip(read, keep):
            if not kept:
                withheld.append(text.strip())
                continue
            if line_words:
                page_boxes.extend(_box(np.asarray(points), word) for word, _, points in line_words if word.strip())
            else:
                page_boxes.append(_box(box, text))
        # Same layout rules as a text layer: rows in reading order, and a scanned table as a Markdown table
        page_text, _ = render_page(page_boxes)
        return OcrPage(
            text=page_text,
            mean_confidence=round(float(np.mean(scores)), 4) if scores else None,
            lines_detected=lines_detected,
            lines_read=lines_read,
            withheld_lines=withheld,
        )

    def close(self) -> None:
        self._engine = None


def _box(points: np.ndarray, text: str) -> Box:
    return Box(float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()),
               float(points[:, 1].max()), text.strip())
