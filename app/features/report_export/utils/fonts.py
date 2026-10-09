# Path: app/features/report_export/utils/fonts.py

"""
The typefaces an exported report is set in, shipped with the appliance in data/fonts (SIL Open Font License).

Source Serif 4 for the text and headings and Inter for the small labels: the web app's own pair, so a PDF reads
like the page it was saved from. Both cover Serbian Latin and Cyrillic. Symbols neither has (≥, →, ∑ in running
text) fall back to STIX Two Math, which ziamath ships for its formulas. Bundled rather than found on the system:
an appliance's system fonts are whatever its OS image had, and a report must not change its look between boxes.
"""

from pathlib import Path
from typing import Dict, Tuple

from app.core.config import settings
from app.features.report_export.exceptions.rendering import FontMissingError

SERIF = "SourceSerif"
SANS = "Inter"
SYMBOLS = "STIXTwoMath"

# (family, style) -> file in data/fonts. Semibold stands in for bold: Source Serif's Bold is heavier than the
# app's headings.
FACES: Dict[Tuple[str, str], str] = {
    (SERIF, ""): "SourceSerif4-Regular.ttf",
    (SERIF, "I"): "SourceSerif4-It.ttf",
    (SERIF, "B"): "SourceSerif4-Semibold.ttf",
    (SERIF, "BI"): "SourceSerif4-SemiboldIt.ttf",
    (SANS, ""): "Inter-Regular.ttf",
    (SANS, "B"): "Inter-SemiBold.ttf",
}


def font_dir() -> Path:
    return Path(settings.BASE_DIR) / "data" / "fonts"


def symbols_font() -> Path:
    import ziamath
    return Path(ziamath.__file__).parent / "fonts" / "STIXTwoMath-Regular.ttf"


def register(pdf) -> None:
    """Adds every face to an FPDF document, and the symbol fallback. Raises FontMissingError if one is absent."""
    folder = font_dir()
    for (family, style), name in FACES.items():
        path = folder / name
        if not path.is_file():
            raise FontMissingError(f"The report font {name} is missing from {folder}.")
        pdf.add_font(family, style, str(path))
    symbols = symbols_font()
    if symbols.is_file():
        pdf.add_font(SYMBOLS, "", str(symbols))
        pdf.set_fallback_fonts([SYMBOLS], exact_match=False)
