# Path: tests/features/document_conversion/utils/test_layout.py

from app.features.document_conversion.utils.layout import Box, RenderedTable, markdown_table, render_page

HEIGHT = 10


def line(y, *cells):
    """A line of words; each cell is (x, text), its words 3 units apart (a normal space at this height)."""
    boxes = []
    for x, text in cells:
        for word in text.split():
            width = 6 * len(word)
            boxes.append(Box(x, x + width, y, y + HEIGHT, word))
            x += width + 3
    return boxes


def page(*lines):
    return [box for boxes in lines for box in boxes]


class TestBorderlessTables:
    def test_rows_with_shared_gutters_become_a_markdown_table(self):
        boxes = page(
            line(0, (0, "Invoice INV-7"),),
            line(20, (0, "Item"), (200, "Qty"), (300, "Amount")),
            line(32, (0, "Parking place"), (200, "2"), (300, "EUR 80.00")),
            line(44, (0, "Cleaning"), (200, "4"), (300, "EUR 100.00")),
            line(70, (0, "Payable within thirty days."),),
        )

        text, tables = render_page(boxes)

        assert tables == 1
        assert text == ("Invoice INV-7\n\n| Item | Qty | Amount |\n| --- | --- | --- |\n"
                        "| Parking place | 2 | EUR 80.00 |\n| Cleaning | 4 | EUR 100.00 |\n\n"
                        "Payable within thirty days.")

    def test_right_aligned_numbers_share_a_column(self):
        boxes = page(
            line(0, (0, "Service"), (200, "Amount")),
            line(12, (0, "Rent"), (218, "1,250.00")),
            line(24, (0, "Parking"), (230, "80.00")),
        )

        text, _ = render_page(boxes)

        assert "| Rent | 1,250.00 |" in text and "| Parking | 80.00 |" in text

    def test_a_heading_line_is_not_absorbed_as_a_row(self):
        boxes = page(
            line(0, (0, "Payment details:"),),
            line(12, (0, "IBAN"), (150, "SI56 1910 0000")),
            line(24, (0, "Due date"), (150, "20 November")),
            line(36, (0, "Total"), (150, "EUR 1,430.00")),
        )

        text, _ = render_page(boxes)

        assert text.startswith("Payment details:\n\n| IBAN | SI56 1910 0000 |")

    def test_empty_cells_stay_empty(self):
        boxes = page(
            line(0, (0, "Item"), (200, "Qty"), (300, "Amount")),
            line(12, (0, "Deposit"), (300, "500.00")),
            line(24, (0, "Rent"), (200, "1"), (300, "1,250.00")),
        )

        text, _ = render_page(boxes)

        assert "| Deposit |  | 500.00 |" in text

    def test_two_aligned_lines_are_not_a_table(self):
        boxes = page(line(0, (0, "Ljubljana"), (200, "5 November 2025")),
                     line(12, (0, "Acme Ltd"), (200, "Registration 123")))

        text, tables = render_page(boxes)

        assert tables == 0 and "|" not in text


    def test_a_wide_timetable_is_still_a_table(self):
        """15 columns, like the 1900 railway timetable whose Norfolk row stayed flat text under a cap of 8."""
        header = line(0, *[(60 * i, f"C{i}") for i in range(15)])
        rows = [line(12 * (r + 1), *[(60 * i, f"{r}.{i:02d}") for i in range(15)]) for r in range(3)]

        text, tables = render_page(page(header, *rows))

        assert tables == 1
        assert text.splitlines()[0] == "| " + " | ".join(f"C{i}" for i in range(15)) + " |"
        assert text.splitlines()[2] == "| " + " | ".join(f"0.{i:02d}" for i in range(15)) + " |"


class TestTableInPieces:
    """Found on a 1900 railway timetable: its header and body were detected as two tables that didn't line up."""

    COLUMNS = (0, 100, 200, 300)

    def _row(self, y, *texts):
        return line(y, *[(x, text) for x, text in zip(self.COLUMNS, texts) if text])

    def test_header_marker_line_and_body_become_one_table(self):
        boxes = page(
            self._row(0, "Sunday", "Berk.", "STATIONS", "Day"),
            self._row(12, "only.", "Exp.", "", "Exp."),
            self._row(24, "A.M.", "P.M.", "Leave.", "A.M."),
            self._row(36, "P.M."),                                   # a lone marker line breaks detection
            self._row(48, "4.22", "7.23", "NORFOLK", "1.02"),
            self._row(60, "4.46", "7.45", "WINSTED", "1.24"),
            self._row(72, "5.01", "8.00", "HARTFORD", "1.38"),
        )

        text, tables = render_page(boxes)

        assert tables == 1
        assert text.splitlines()[0] == "| Sunday | Berk. | STATIONS | Day |"
        assert "| 4.22 | 7.23 | NORFOLK | 1.02 |" in text.splitlines()

    def test_a_title_across_the_columns_keeps_the_pieces_apart(self):
        title = line(36, (0, "Trains to New York stop at every station"))
        boxes = page(
            self._row(0, "Sunday", "Berk.", "STATIONS", "Day"),
            self._row(12, "only.", "Exp.", "", "Exp."),
            self._row(24, "A.M.", "P.M.", "Leave.", "A.M."),
            title,
            self._row(48, "4.22", "7.23", "NORFOLK", "1.02"),
            self._row(60, "4.46", "7.45", "WINSTED", "1.24"),
            self._row(72, "5.01", "8.00", "HARTFORD", "1.38"),
        )

        text, tables = render_page(boxes)

        assert tables == 2
        assert "Trains to New York stop at every station" in text.splitlines()

    def test_a_word_in_a_gutter_goes_to_the_nearest_column_not_the_last(self):
        from app.features.document_conversion.utils.layout import _column_of
        columns = [(0, 50), (100, 150), (200, 250), (300, 350)]
        assert _column_of(Box(55, 65, 0, 10, "d"), columns) == 0
        assert _column_of(Box(180, 195, 0, 10, "W"), columns) == 2


class TestProseIsNotATable:
    def test_justified_prose_whose_word_gaps_line_up_by_chance(self):
        """
        All three lines leave x=96..100 empty: a shared strip wider than the minimum gutter, with short cells on
        both sides. Only its width gives it away -- it is no wider than the gaps between the words around it.
        """
        def prose_line(y, left_words, right_words):
            boxes, x = [], 96
            for word in reversed(left_words):          # left cell ends exactly at x=96
                boxes.insert(0, Box(x - 6 * len(word), x, y, y + HEIGHT, word))
                x -= 6 * len(word) + 4
            x = 100
            for word in right_words:                   # right cell starts exactly at x=100
                boxes.append(Box(x, x + 6 * len(word), y, y + HEIGHT, word))
                x += 6 * len(word) + 4
            return boxes

        boxes = page(
            prose_line(0, ["shall", "keep"], ["the", "Premises", "in", "repair"]),
            prose_line(12, ["and", "shall"], ["permit", "the", "Landlord", "to"]),
            prose_line(24, ["at", "all"], ["reasonable", "times", "on", "notice"]),
        )

        _, tables = render_page(boxes)

        assert tables == 0

    def test_the_same_layout_with_a_real_gutter_is_a_table(self):
        """The control for the test above: widen the shared strip well past the word gaps and it is a table."""
        boxes = page(
            line(0, (0, "shall keep"), (160, "the Premises")),
            line(12, (0, "and shall"), (160, "permit the")),
            line(24, (0, "at all"), (160, "reasonable times")),
        )

        _, tables = render_page(boxes)

        assert tables == 1

    def test_side_by_side_paragraphs_with_long_cells(self):
        english = "The Tenant shall pay the rent monthly in advance to the account"
        slovenian = "Najemnik plačuje najemnino mesečno vnaprej na račun najemodajalca"
        boxes = page(*(line(12 * i, (0, english), (500, slovenian)) for i in range(4)))

        _, tables = render_page(boxes)

        assert tables == 0


class TestRenderPage:
    def test_ruled_tables_are_placed_where_they_sit(self):
        boxes = page(line(0, (0, "Article 9 Rent"),), line(80, (0, "Binding for the term."),))
        ruled = RenderedTable(top=30, bottom=60, markdown="| Year | Rent |\n| --- | --- |\n| 2026 | 1,300 |")

        text, tables = render_page(boxes, [ruled])

        assert tables == 1
        assert text == "Article 9 Rent\n\n| Year | Rent |\n| --- | --- |\n| 2026 | 1,300 |\n\nBinding for the term."

    def test_paragraph_gap_starts_a_new_paragraph(self):
        boxes = page(line(0, (0, "First line"),), line(12, (0, "continues here."),),
                     line(40, (0, "New paragraph."),))

        text, _ = render_page(boxes)

        assert text == "First line\ncontinues here.\n\nNew paragraph."

    def test_no_boxes_is_empty(self):
        assert render_page([]) == ("", 0)


class TestMarkdownTable:
    def test_escapes_pipes_collapses_whitespace_and_pads_short_rows(self):
        table = markdown_table([["Term", "Value"], ["A | B", "  two\nlines "], ["Only one"]])

        assert table == ("| Term | Value |\n| --- | --- |\n| A \\| B | two lines |\n| Only one |  |")
