# Path: tests/features/report_export/utils/test_line_layout.py

from app.features.report_export.utils.line_layout import MAX_STRETCH, Atom, break_lines

SPACE = Atom(width=2.0, glue=True)


def word(width, ascent=7.0, descent=2.0, name=None):
    return Atom(width=width, ascent=ascent, descent=descent, payload=name)


def names(line):
    return [atom.payload for _, atom in line.placed]


class TestBreakLines:
    def test_words_fill_a_line_then_wrap(self):
        atoms = [word(10, name="a"), SPACE, word(10, name="b"), SPACE, word(10, name="c")]

        lines = break_lines(atoms, max_width=23)

        assert [names(line) for line in lines] == [["a", "b"], ["c"]]
        assert [x for x, _ in lines[0].placed] == [0, 12]

    def test_touching_pieces_never_split(self):
        """A formula and the comma after it, or "(" and a formula, stay on one line together."""
        atoms = [word(10, name="a"), SPACE, word(4, name="("), word(8, name="math"), word(3, name=")")]

        lines = break_lines(atoms, max_width=20)

        assert [names(line) for line in lines] == [["a"], ["(", "math", ")"]]

    def test_a_line_is_as_tall_as_its_tallest_piece(self):
        (line,) = break_lines([word(5), SPACE, word(5, ascent=14.0, descent=6.0)], max_width=100)

        assert (line.ascent, line.descent) == (14.0, 6.0)

    def test_justified_lines_stretch_their_spaces_but_the_last_does_not(self):
        atoms = [word(10, name="a"), SPACE, word(10, name="b"), SPACE, word(10, name="c")]

        first, last = break_lines(atoms, max_width=26, justify=True)

        assert [x for x, _ in first.placed] == [0, 16]
        assert first.width == 26
        assert [x for x, _ in last.placed] == [0]

    def test_a_line_that_would_stretch_too_far_is_left_ragged(self):
        atoms = [word(10, name="a"), SPACE, word(10, name="b"), SPACE, word(80, name="long")]

        first = break_lines(atoms, max_width=22 + 2 * MAX_STRETCH + 10, justify=True)[0]

        assert [x for x, _ in first.placed] == [0, 12]

    def test_a_newline_forces_a_break_and_is_not_justified(self):
        atoms = [word(10, name="a"), Atom(width=0, newline=True), word(10, name="b")]

        lines = break_lines(atoms, max_width=100, justify=True)

        assert [names(line) for line in lines] == [["a"], ["b"]]

    def test_a_piece_wider_than_the_line_gets_a_line_of_its_own(self):
        lines = break_lines([word(5, name="a"), SPACE, word(500, name="huge")], max_width=100)

        assert [names(line) for line in lines] == [["a"], ["huge"]]

    def test_no_atoms_is_one_empty_line(self):
        assert [names(line) for line in break_lines([], 100)] == [[]]
