# Path: app/features/report_export/utils/math_render.py

"""
LaTeX typeset as SVG, with the measurements needed to set it on a line of text.

ziamath does the typesetting: pure Python, with the STIX Two Math font bundled in its package, so a formula looks
the same on every appliance and nothing is fetched. Its SVG draws glyphs as paths, which fpdf2 embeds as vectors.
"""

import re
from functools import lru_cache

from app.features.report_export.domain.entities import RenderedMath
from app.features.report_export.exceptions.rendering import MathRenderError

_VIEWBOX = re.compile(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"')


@lru_cache(maxsize=512)
def render(latex: str, size: float, display: bool = False, color: str = "#1f1d1a") -> RenderedMath:
    """
    `latex` at `size` points. Display style for a formula on its own line, text style inside a sentence (smaller
    fractions and limits, so a line's height barely changes). Raises MathRenderError for LaTeX it cannot set.
    """
    try:
        import ziamath
        math = ziamath.Latex(latex, size=size, inline=not display, color=color, margin=0.6)
        svg = math.svg()
    except Exception as e:
        raise MathRenderError(f"Could not typeset '{latex}': {e}", original_exception=e) from e

    match = _VIEWBOX.search(svg)
    if match is None:
        raise MathRenderError(f"Typesetting '{latex}' produced no measurable drawing.")
    _x, top, width, height = (float(v) for v in match.groups())
    # The baseline is y = 0 in the drawing's own coordinates, so whatever lies above it is the ascent
    return RenderedMath(svg=svg.encode("utf-8"), width=width, height=height, ascent=max(0.0, -top))
