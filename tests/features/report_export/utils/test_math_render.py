# Path: tests/features/report_export/utils/test_math_render.py

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.features.report_export.exceptions.rendering import MathRenderError
from app.features.report_export.utils import math_render


@pytest.fixture
def ziamath(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "ziamath", fake)
    math_render.render.cache_clear()
    yield fake
    math_render.render.cache_clear()


class TestRender:
    def test_measures_the_drawing_from_its_viewbox_with_the_baseline_at_zero(self, ziamath):
        ziamath.Latex.return_value.svg.return_value = (
            '<svg width="92.5" height="15" viewBox="-0.37 -11.3 92.5 15.0"><path d="M 0 0"/></svg>')

        rendered = math_render.render("E_p", 11.0)

        ziamath.Latex.assert_called_once_with("E_p", size=11.0, inline=True, color="#1f1d1a", margin=0.6)
        assert (rendered.width, rendered.height, rendered.ascent) == (92.5, 15.0, 11.3)
        assert rendered.descent == pytest.approx(3.7)
        assert rendered.svg.startswith(b"<svg")

    def test_a_display_formula_is_set_in_display_style(self, ziamath):
        ziamath.Latex.return_value.svg.return_value = '<svg viewBox="0 -20 40 30"></svg>'

        math_render.render("\\frac{a}{b}", 11.0, display=True)

        assert ziamath.Latex.call_args.kwargs["inline"] is False

    def test_latex_it_cannot_set_is_a_typed_error(self, ziamath):
        ziamath.Latex.side_effect = ValueError("Undefined control sequence")

        with pytest.raises(MathRenderError):
            math_render.render("\\nonsense{", 11.0)

    def test_a_drawing_without_a_viewbox_is_a_typed_error(self, ziamath):
        ziamath.Latex.return_value.svg.return_value = "<svg></svg>"

        with pytest.raises(MathRenderError):
            math_render.render("x", 11.0)

    def test_the_same_formula_is_typeset_once(self, ziamath):
        ziamath.Latex.return_value.svg.return_value = '<svg viewBox="0 -8 10 10"></svg>'

        math_render.render("x", 11.0)
        math_render.render("x", 11.0)

        assert ziamath.Latex.call_count == 1
