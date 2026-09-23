# Path: app/features/document_conversion/data/pdfplumber_layout.py

import io
from typing import List, Optional

import pdfplumber

from app.features.document_conversion.domain.interfaces import IPageLayoutExtractor
from app.features.document_conversion.utils.layout import Box, RenderedTable, markdown_table, render_page

# Tables drawn with ruling lines. The text strategy is not used: on ordinary prose it cuts sentences into
# "cells". Borderless tables are found from word positions instead (utils/layout.py).
_RULED = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
# Characters closer than this fraction of their size belong to one word (a space is ~0.25 of it)
_X_TOLERANCE_RATIO = 0.15


class PdfplumberLayoutExtractor(IPageLayoutExtractor):
    """
    Text of a born-digital page with its tables as Markdown tables, via pdfplumber (MIT; the library MarkItDown's
    own PDF converter uses). Returns None for a page without tables, which then keeps MarkItDown's text.
    """

    def extract(self, page_pdf: bytes) -> Optional[str]:
        with pdfplumber.open(io.BytesIO(page_pdf)) as pdf:
            page = pdf.pages[0]
            ruled = [t for t in page.find_tables(_RULED) if _is_grid(t.extract())]
            tables = [RenderedTable(top=t.bbox[1], bottom=t.bbox[3], markdown=markdown_table(t.extract()))
                      for t in ruled]

            boxes: List[Box] = []
            # Tolerances relative to each character's size: pdfplumber's fixed 3-point default joins the words
            # of two cells whenever the gutter is narrower than 3 points, which a small font makes common
            for word in page.extract_words(x_tolerance_ratio=_X_TOLERANCE_RATIO, y_tolerance_ratio=0.3,
                                           keep_blank_chars=False, use_text_flow=False):
                if any(_inside(word, t.bbox) for t in ruled):
                    continue
                boxes.append(Box(word["x0"], word["x1"], word["top"], word["bottom"], word["text"]))

        text, table_count = render_page(boxes, tables)
        return text if table_count else None


def _is_grid(rows) -> bool:
    """At least two rows and two columns with text: a single ruled box (a signature frame) is not a table."""
    rows = [row for row in rows or [] if any(cell and cell.strip() for cell in row)]
    return len(rows) >= 2 and max(len(row) for row in rows) >= 2


def _inside(word: dict, bbox) -> bool:
    x0, top, x1, bottom = bbox
    middle_x, middle_y = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
    return x0 <= middle_x <= x1 and top <= middle_y <= bottom
