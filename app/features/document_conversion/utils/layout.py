# Path: app/features/document_conversion/utils/layout.py

"""
Page text from positioned boxes -- PDF words or OCR lines -- with tables kept as Markdown tables.

Neither MarkItDown nor OCR keeps a table's rows: MarkItDown reads a borderless invoice column by column ("Parking
place", "Cleaning service", "2", "4", "EUR 40.00", ...), and OCR reads each cell as a separate phrase. A chunk that
has lost which amount belongs to which item cannot answer "what did parking cost", and neither can the SLM that
writes its context. Here, boxes are grouped into lines, and consecutive lines that leave the same vertical strips
empty (gutters), with words on both sides of them, become a Markdown table. Shared gutters rather than wide gaps:
OCR word spacing jitters too much for any one gap width to separate cells from words, but prose never keeps its
word gaps in the same place from line to line.

Pure functions over `Box`es, so every rule is testable without a PDF. Coordinates grow downwards (top < bottom).
"""

import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Boxes on one line: vertical centres closer than this fraction of a line height
_SAME_LINE = 0.5
# A column gutter: a vertical strip at least this many line heights wide that no word of any row touches
_MIN_GUTTER = 0.35
# A vertical gap larger than this many line heights starts a new paragraph
_PARAGRAPH_GAP = 0.7
# A borderless table needs at least this many rows (header included), every one with at least two cells. Two
# aligned lines are too easily a coincidence -- a date beside a place name -- to be called a table.
MIN_TABLE_ROWS = 3
# A railway timetable or a financial statement easily has 15 columns; with a cap of 8 such a table stayed flat text,
# and an answer read "10.06" from the wrong column of a row that had lost its cells. Prose is kept out by the gutter
# and cell-length rules, not by this cap.
MAX_TABLE_COLUMNS = 16
# Side-by-side prose (a bilingual contract) also leaves a shared gutter; table cells are short
MAX_MEAN_WORDS_PER_CELL = 6.0
# A table the detector found in two pieces -- its header, a few lines it couldn't place (a lone "P. M.", a row a
# rotated caption crosses), then the body -- is joined when at most this many lines separate the pieces
MAX_LINES_BETWEEN_TABLE_PIECES = 3
# A gutter must be clearly wider than the gaps between the words inside the cells. Word gaps in justified prose
# can line up across a few lines by chance, but such a "gutter" is only as wide as a word gap.
MIN_GUTTER_TO_WORD_GAP = 1.5


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    top: float
    bottom: float
    text: str

    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class Line:
    boxes: List[Box]

    @property
    def top(self) -> float:
        return min(b.top for b in self.boxes)

    @property
    def bottom(self) -> float:
        return max(b.bottom for b in self.boxes)

    @property
    def text(self) -> str:
        return " ".join(box.text for box in self.boxes)


@dataclass(frozen=True)
class RenderedTable:
    """A table found by other means (ruling lines), placed on the page by its top edge."""
    top: float
    bottom: float
    markdown: str


def render_page(boxes: Sequence[Box], tables: Sequence[RenderedTable] = ()) -> Tuple[str, int]:
    """
    Page text in reading order, with borderless tables detected among `boxes` and `tables` (already rendered)
    placed where they sit. Returns the text and how many tables it contains.
    """
    lines = group_lines(boxes)
    items: List[Tuple[float, float, str, bool]] = []  # (top, bottom, text, is_table)

    height = _line_height(boxes)
    spans: List[Tuple[int, int, str]] = []  # (first line, line after the last, Markdown)
    index = 0
    while index < len(lines):
        found = _table_at(lines, index, height)
        if found is not None:
            spans.append((index, found[0], found[1]))
            index = found[0]
        else:
            index += 1
    spans = _join_table_pieces(lines, spans, height)

    table_count = len(tables) + len(spans)
    index = 0
    for start, end, table in spans:
        items.extend((line.top, line.bottom, line.text, False) for line in lines[index:start])
        items.append((lines[start].top, lines[end - 1].bottom, table, True))
        index = end
    items.extend((line.top, line.bottom, line.text, False) for line in lines[index:])

    items.extend((t.top, t.bottom, t.markdown, True) for t in tables)
    items.sort(key=lambda item: item[0])

    out: List[str] = []
    previous: Optional[Tuple[float, float, str, bool]] = None
    for item in items:
        if previous is not None and (item[3] or previous[3] or item[0] - previous[1] > _PARAGRAPH_GAP * height):
            out.append("")
        out.append(item[2])
        previous = item
    return "\n".join(out), table_count


def group_lines(boxes: Sequence[Box]) -> List[Line]:
    """Boxes grouped into lines top to bottom, each line's boxes left to right."""
    if not boxes:
        return []
    height = _line_height(boxes)
    lines: List[Line] = []
    for box in sorted(boxes, key=lambda b: (b.middle, b.x0)):
        if lines and abs(box.middle - statistics.mean(b.middle for b in lines[-1].boxes)) < _SAME_LINE * height:
            lines[-1].boxes.append(box)
        else:
            lines.append(Line(boxes=[box]))
    for line in lines:
        line.boxes.sort(key=lambda b: b.x0)
    return lines


def _table_at(lines: List[Line], start: int, height: float) -> Optional[Tuple[int, str]]:
    """
    The longest table starting at `lines[start]`: (index after its last row, Markdown), or None.

    Rows are added while the lines keep at least one shared gutter -- a strip no word of any of them touches --
    and every row still has words in two or more of the columns those gutters make. A prose line crosses every
    gutter; a lone heading or a wrapped cell has only one cell; either ends the table.
    """
    best: Optional[Tuple[int, str]] = None
    end = start + 1
    while end <= len(lines):
        rows = lines[start:end]
        columns, narrowest_gutter = _columns(rows, height)
        if len(columns) < 2:
            break
        grid = [_split_row(row, columns) for row in rows]
        if any(sum(1 for value in values if value) < 2 for values in grid):
            break
        if (len(rows) >= MIN_TABLE_ROWS and len(columns) <= MAX_TABLE_COLUMNS and _short_cells(grid)
                and narrowest_gutter >= MIN_GUTTER_TO_WORD_GAP * _word_gap(rows, columns)):
            best = (end, markdown_table(grid))
        end += 1
    return best


def _join_table_pieces(lines: List[Line], spans: List[Tuple[int, int, str]],
                       height: float) -> List[Tuple[int, int, str]]:
    """
    Joins a table found in pieces into one, cut along the columns of its later piece (the body, usually the
    widest). Without it a timetable's header ("Sunday only", "Berk. Exp.") and its rows ended up in two tables
    whose columns didn't line up, and an answer read "10.06" from the wrong column. Joined only when no word of
    the lines in between runs across a whole gutter, as a title or a sentence would.
    """
    joined: List[Tuple[int, int, str]] = []
    for start, end, table in spans:
        if joined and start - joined[-1][1] <= MAX_LINES_BETWEEN_TABLE_PIECES:
            first = joined[-1][0]
            columns, _ = _columns(lines[start:end], height)
            if not any(_spans_a_gutter(line, columns, height) for line in lines[first:end]):
                joined[-1] = (first, end, markdown_table([_split_row(row, columns) for row in lines[first:end]]))
                continue
        joined.append((start, end, table))
    return joined


def _spans_a_gutter(line: Line, columns: List[Tuple[float, float]], height: float) -> bool:
    """
    The line runs across the columns like a title or a sentence: its words bridge -- leave no gutter-sized hole
    in -- at least half of the gutters they reach (one box per word, so no single box need cover one). A header
    has cells wider than the body's, and two of them may nearly touch over one gutter ("M'tain |Sunday", 16 px
    apart), but it stays open over the rest. Punctuation doesn't count -- OCR boxed one bracket 230 px wide.
    """
    words = [box for box in line.boxes if any(ch.isalnum() for ch in box.text)]
    if not words:
        return False
    left, right = min(box.x0 for box in words), max(box.x1 for box in words)
    reached = bridged = 0
    for g0, g1 in ((a[1], b[0]) for a, b in zip(columns, columns[1:])):
        if g1 <= g0 or g1 <= left or g0 >= right:
            continue
        reached += 1
        covered = sorted((max(box.x0, g0), min(box.x1, g1)) for box in words if box.x1 > g0 and box.x0 < g1)
        reach, hole = g0, 0.0
        for x0, x1 in covered:
            hole = max(hole, x0 - reach)
            reach = max(reach, x1)
        hole = max(hole, g1 - reach)
        bridged += hole < _MIN_GUTTER * height
    return bridged > 0 and bridged >= reached / 2


def _columns(rows: List[Line], height: float) -> Tuple[List[Tuple[float, float]], float]:
    """
    Column extents between the gutters the rows share -- the strips no word of any row covers -- and the width
    of the narrowest gutter.
    """
    boxes = [box for row in rows for box in row.boxes]
    left, right = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    columns: List[Tuple[float, float]] = []
    narrowest = float("inf")
    column_start, reach = left, left
    for x0, x1 in sorted((b.x0, b.x1) for b in boxes):
        if x0 - reach >= _MIN_GUTTER * height and reach > column_start:
            columns.append((column_start, reach))
            narrowest = min(narrowest, x0 - reach)
            column_start = x0
        reach = max(reach, x1)
    columns.append((column_start, right))
    return columns, narrowest


def _word_gap(rows: List[Line], columns: List[Tuple[float, float]]) -> float:
    """Median gap between neighbouring words inside the same cell; 0 when every cell is one word."""
    gaps = []
    for row in rows:
        for previous, box in zip(row.boxes, row.boxes[1:]):
            if _column_of(previous, columns) == _column_of(box, columns):
                gaps.append(max(0.0, box.x0 - previous.x1))
    return statistics.median(gaps) if gaps else 0.0


def _column_of(box: Box, columns: List[Tuple[float, float]]) -> int:
    """The column holding the box's centre, or, for a centre in a gutter, the nearest column (not the last)."""
    centre = (box.x0 + box.x1) / 2
    return min(range(len(columns)),
               key=lambda i: 0.0 if columns[i][0] <= centre <= columns[i][1]
               else min(abs(centre - columns[i][0]), abs(centre - columns[i][1])))


def _split_row(row: Line, columns: List[Tuple[float, float]]) -> List[str]:
    values: List[List[str]] = [[] for _ in columns]
    for box in row.boxes:
        values[_column_of(box, columns)].append(box.text)
    return [" ".join(words) for words in values]


def _short_cells(grid: List[List[str]]) -> bool:
    cells = [value for values in grid for value in values if value]
    return statistics.mean(len(value.split()) for value in cells) <= MAX_MEAN_WORDS_PER_CELL


def markdown_table(grid: Sequence[Sequence[Optional[str]]]) -> str:
    """Rows of cell text as a Markdown table; the first row is the header."""
    rows = [[_cell_text(value) for value in row] for row in grid]
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def _cell_text(value: Optional[str]) -> str:
    return " ".join((value or "").split()).replace("|", "\\|")


def _line_height(boxes: Sequence[Box]) -> float:
    heights = [b.height for b in boxes if b.height > 0]
    return statistics.median(heights) if heights else 1.0
