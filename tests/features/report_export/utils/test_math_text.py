# Path: tests/features/report_export/utils/test_math_text.py

import pytest

from app.features.report_export.utils.math_text import (
    PLACEHOLDER_CLOSE, PLACEHOLDER_OPEN, find_math, normalize_for_display, protect,
)


def _found(text):
    return [(span.latex, span.display) for _, _, span in find_math(text)]


class TestFindMath:
    @pytest.mark.parametrize("text, expected", [
        ("elasticity ($E_p$) is", [("E_p", False)]),
        ("as $(\\Delta Q / Q) / (\\Delta P / P)$ [Doc 21].", [("(\\Delta Q / Q) / (\\Delta P / P)", False)]),
        ("inline \\(x^2\\) here", [("x^2", False)]),
        ("display \\[\\frac{a}{b}\\] here", [("\\frac{a}{b}", True)]),
        ("$$\\bar{P}$$", [("\\bar{P}", True)]),
        ("the mean $\\bar{Q}$[Doc 27]", [("\\bar{Q}", False)]),
        ("$Q_S = 15.90 + 0.72P_G$ and $Q_D$", [("Q_S = 15.90 + 0.72P_G", False), ("Q_D", False)]),
    ])
    def test_finds_every_delimiter_style(self, text, expected):
        assert _found(text) == expected

    @pytest.mark.parametrize("text", [
        "equilibrium is $6.40 per thousand cubic feet; a ceiling of $3.00 per mcf",
        "costs $5 and $10",
        "an escaped \\$5 and \\$7",
        "a lone $ sign",
        "spaced $ x $ is not math",
        "`$code$` is left alone",
        "```\n$fenced$\n```",
    ])
    def test_money_code_and_stray_dollars_are_not_math(self, text):
        assert _found(text) == []

    def test_a_single_dollar_formula_never_spans_lines(self):
        assert _found("$a\nb$") == []


class TestProtect:
    def test_formulas_become_numbered_placeholders(self):
        protected, spans = protect("a $E_p$ and $$x$$ b")

        assert protected == f"a {PLACEHOLDER_OPEN}0{PLACEHOLDER_CLOSE} and {PLACEHOLDER_OPEN}1{PLACEHOLDER_CLOSE} b"
        assert [(s.latex, s.display) for s in spans] == [("E_p", False), ("x", True)]

    def test_text_without_math_is_unchanged(self):
        assert protect("from $6.40 to $3.00") == ("from $6.40 to $3.00", [])


class TestNormalizeForDisplay:
    @pytest.mark.parametrize("markdown, expected", [
        ("elasticity ($E_p$) is", "elasticity ($$E_p$$) is"),
        ("inline \\(x^2\\)", "inline $$x^2$$"),
        ("from $6.40 to $3.00", "from $6.40 to $3.00"),
        ("text\n\\[\\frac{a}{b}\\]\nmore", "text\n$$\n\\frac{a}{b}\n$$\nmore"),
        ("inside \\[a\\] a line", "inside $$a$$ a line"),
        ("$$\n  x +\n  y\n$$", "$$\nx + y\n$$"),
    ])
    def test_every_formula_is_written_the_way_the_web_plugin_reads_it(self, markdown, expected):
        assert normalize_for_display(markdown) == expected
