# Path: app/features/report_export/utils/math_text.py

"""
Finds the formulas the report writer put in its Markdown, and tells them apart from money.

The writer emits LaTeX unprompted (`$E_p$`, `$(\\Delta Q / Q) / (\\Delta P / P)$`), in every delimiter style, and the
same prose quotes prices ("from $6.40 … to $3.00"). One rule decides which is which, used by the PDF and by the web
Report page, so the two never disagree:

- `$$…$$`, `\\[…\\]` and `\\(…\\)` are always math;
- `$…$` is math only under pandoc's rule: the opening `$` is followed by neither a space nor a digit, and the
  closing `$` is preceded by no space and followed by no digit. "$6.40 … $3.00" fails on both ends.

Code spans and fences are left alone. Nothing here touches the stored Markdown: the Markdown download keeps the
writer's LaTeX as it was written.
"""

import re
from typing import List, Tuple

from app.features.report_export.domain.entities import MathSpan

# Stands in for a formula while Markdown is parsed, so its `_`, `\\` and `*` are not read as emphasis or escapes.
# Private-use characters: they occur in no report and survive the Markdown parser untouched.
PLACEHOLDER_OPEN = ""
PLACEHOLDER_CLOSE = ""
PLACEHOLDER = re.compile(f"{PLACEHOLDER_OPEN}(\\d+){PLACEHOLDER_CLOSE}")

_CODE = re.compile(r"```.*?```|~~~.*?~~~|(`+)(?:(?!\1).)+?\1", re.S)
_MATH = re.compile(
    r"\$\$(?P<double>.+?)\$\$"
    r"|\\\[(?P<bracket>.+?)\\\]"
    r"|\\\((?P<paren>.+?)\\\)"
    # pandoc's single-dollar rule; a single-dollar formula never spans lines
    r"|(?<![\\$\w])\$(?=[^\s\d$])(?P<single>[^$\n]+?)(?<=[^\s\\])\$(?![\d$])",
    re.S,
)


def find_math(text: str) -> List[Tuple[int, int, MathSpan]]:
    """Every formula in `text`, as (start, end, span), in order. `display` is True for `$$` and `\\[` forms."""
    found: List[Tuple[int, int, MathSpan]] = []
    position = 0
    for code in [*_CODE.finditer(text), None]:
        stop = code.start() if code is not None else len(text)
        for match in _MATH.finditer(text, position, stop):
            latex = next(group for group in match.group("double", "bracket", "paren", "single") if group is not None)
            latex = latex.strip()
            if not latex:
                continue
            display = match.group("double") is not None or match.group("bracket") is not None
            found.append((match.start(), match.end(), MathSpan(latex=latex, display=display)))
        if code is not None:
            position = code.end()
    return found


def protect(text: str) -> Tuple[str, List[MathSpan]]:
    """`text` with each formula replaced by a numbered placeholder, and the formulas in placeholder order."""
    spans: List[MathSpan] = []
    parts: List[str] = []
    position = 0
    for start, end, span in find_math(text):
        parts.append(text[position:start])
        parts.append(f"{PLACEHOLDER_OPEN}{len(spans)}{PLACEHOLDER_CLOSE}")
        spans.append(span)
        position = end
    parts.append(text[position:])
    return "".join(parts), spans


def normalize_for_display(markdown: str) -> str:
    """
    The Markdown with every formula in the two forms the web page's math plugin reads with single dollars off:
    `$$…$$` inside a line, and a `$$` block for a display formula standing on its own line. A dollar amount keeps
    its single `$` and so stays text.
    """
    parts: List[str] = []
    position = 0
    for start, end, span in find_math(markdown):
        parts.append(markdown[position:start])
        line_start = markdown.rfind("\n", 0, start) + 1
        line_end = markdown.find("\n", end)
        line_end = len(markdown) if line_end == -1 else line_end
        alone = not markdown[line_start:start].strip() and not markdown[end:line_end].strip()
        latex = " ".join(span.latex.split())
        parts.append(f"$$\n{latex}\n$$" if span.display and alone else f"$${latex}$$")
        position = end
    parts.append(markdown[position:])
    return "".join(parts)
