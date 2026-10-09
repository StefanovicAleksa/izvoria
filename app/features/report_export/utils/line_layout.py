# Path: app/features/report_export/utils/line_layout.py

"""
Breaks a paragraph's measured pieces into lines.

fpdf2 flows plain text well but cannot put a picture inside a line, and a report's sentences carry formulas and
citation tags. So the writer measures every piece itself (a word in its font, a tag, a typeset formula) and this
decides where the lines break: greedy, never between pieces that touch (a formula and the comma after it), and
justified except on a paragraph's last line. Pure arithmetic, so it is tested without a PDF.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class Atom:
    """One piece of a line. `glue` is a breakable space; `newline` forces a break after what came before it."""
    width: float
    ascent: float = 0.0
    descent: float = 0.0
    payload: Any = None
    glue: bool = False
    newline: bool = False


@dataclass
class Line:
    # Each atom with its x offset from the line's start
    placed: List[Tuple[float, Atom]] = field(default_factory=list)
    width: float = 0.0
    ascent: float = 0.0
    descent: float = 0.0


# A justified line may stretch each space by at most this many points; a line that would need more (a long word or
# a formula pushed to the next line) is set ragged instead of being pulled into rivers of white
MAX_STRETCH = 7.0


def break_lines(atoms: List[Atom], max_width: float, justify: bool = False) -> List[Line]:
    groups = _groups(atoms)
    lines: List[List[Tuple[Optional[Atom], List[Atom]]]] = []
    current: List[Tuple[Optional[Atom], List[Atom]]] = []
    width = 0.0
    forced: List[bool] = []

    for glue, members, newline in groups:
        if newline:
            lines.append(current)
            forced.append(True)
            current, width = [], 0.0
            continue
        group_width = sum(a.width for a in members)
        gap = glue.width if (glue is not None and current) else 0.0
        if current and width + gap + group_width > max_width:
            lines.append(current)
            forced.append(False)
            current, width = [], 0.0
            gap = 0.0
        current.append((glue if gap else None, members))
        width += gap + group_width

    if current or not lines:
        lines.append(current)
        forced.append(True)

    return [_place(groups_on_line, max_width, justify and not is_forced)
            for groups_on_line, is_forced in zip(lines, forced)]


def _groups(atoms: List[Atom]) -> List[Tuple[Optional[Atom], List[Atom], bool]]:
    """Runs of touching atoms, each with the glue before it; a newline is a group of its own."""
    groups: List[Tuple[Optional[Atom], List[Atom], bool]] = []
    pending_glue: Optional[Atom] = None
    current: List[Atom] = []
    for atom in atoms:
        if atom.newline:
            if current:
                groups.append((pending_glue, current, False))
            groups.append((None, [], True))
            current, pending_glue = [], None
        elif atom.glue:
            if current:
                groups.append((pending_glue, current, False))
                current = []
                pending_glue = atom
            elif pending_glue is None:
                pending_glue = atom
        else:
            current.append(atom)
    if current:
        groups.append((pending_glue, current, False))
    return groups


def _place(groups: List[Tuple[Optional[Atom], List[Atom]]], max_width: float, justify: bool) -> Line:
    natural = sum((g.width if g is not None else 0.0) + sum(a.width for a in members) for g, members in groups)
    gaps = sum(1 for g, _ in groups if g is not None)
    stretch = (max_width - natural) / gaps if justify and gaps and natural < max_width else 0.0
    if stretch > MAX_STRETCH:
        stretch = 0.0
    line = Line()
    x = 0.0
    for glue, members in groups:
        if glue is not None:
            x += glue.width + stretch
        for atom in members:
            line.placed.append((x, atom))
            line.ascent = max(line.ascent, atom.ascent)
            line.descent = max(line.descent, atom.descent)
            x += atom.width
    line.width = x
    return line
