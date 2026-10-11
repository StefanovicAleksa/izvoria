# Canvas: desktop_theme_v4
# Path: frontend/izvoria_desktop/ui/theme.py

"""
The app's look: one design (editorial, professional-services) in a light or a dark mode.

Warm neutrals, serif headings, near-square corners, outlined citation tags, an ink sidebar in both modes and a
deep green accent. Widgets read colours as module attributes (`theme.CITATION`, `theme.WARNING`, ...) while
they build, so `use()` rebinds those attributes from the chosen palette and must run before the first widget
exists; a switch at runtime goes through `MainWindow.set_mode()`, which also redraws what they baked in.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Qt drops its native combo and spin-box arrows once those widgets are styled, so they are drawn from here.
# One mid-grey stroke reads on every palette's surfaces.
_ASSETS = (Path(__file__).resolve().parent / "assets").as_posix()

SANS_STACK = '"Inter", "Segoe UI", "Cantarell", "Noto Sans", "Helvetica Neue", sans-serif'
SERIF_STACK = '"Source Serif 4", "Noto Serif", "Charter", "Bitstream Charter", "DejaVu Serif", Georgia, serif'
MONO_STACK = '"JetBrains Mono", "DejaVu Sans Mono", "Menlo", monospace'

MODES = ("dark", "light")


@dataclass(frozen=True)
class Palette:
    background: str        # the window canvas
    surface: str           # cards, inputs, documents
    surface_raised: str    # buttons, chips, hover rows
    border: str
    border_strong: str     # hover outlines
    text: str
    text_muted: str
    text_faint: str        # disabled text
    accent: str
    accent_hover: str
    accent_soft: str       # tinted fills: selection, the active list row
    on_accent: str         # text on a solid accent fill
    citation: str
    warning: str
    warning_soft: str
    danger: str
    danger_soft: str
    success: str
    scrollbar: str
    scrollbar_hover: str
    log_text: str
    # The sidebar has its own colours: it stays ink-dark even in light mode.
    sidebar: str
    sidebar_raised: str    # the appliance card, the hovered nav row
    sidebar_input: str
    sidebar_border: str
    sidebar_text: str
    sidebar_muted: str
    sidebar_danger: str    # the Stop button's red, readable on the sidebar's own background
    nav_active_bg: str
    status_bar: str


DARK = Palette(
    background="#151412", surface="#1c1b18", surface_raised="#25231f",
    border="#2f2c27", border_strong="#433f38",
    text="#ece7df", text_muted="#9d978c", text_faint="#5e5a53",
    accent="#6fbfa3", accent_hover="#86cdb3", accent_soft="#1d2d27", on_accent="#0f1f19",
    citation="#7fcfb2",
    warning="#dcae5a", warning_soft="#2b2416", danger="#e27d6f", danger_soft="#2e1b18", success="#7fbf7a",
    scrollbar="#35322c", scrollbar_hover="#48443c", log_text="#b9b2a6",
    sidebar="#0e0d0c", sidebar_raised="#1a1917", sidebar_input="#232220", sidebar_border="#2a2823",
    sidebar_text="#ece7df", sidebar_muted="#8a857c", sidebar_danger="#e8897b", nav_active_bg="#1f1e1b",
    status_bar="#0e0d0c",
)

LIGHT = Palette(
    background="#faf8f5", surface="#ffffff", surface_raised="#f2efea",
    border="#e4dfd6", border_strong="#cbc4b7",
    text="#1f1d1a", text_muted="#6b665e", text_faint="#aea89e",
    accent="#2d6a58", accent_hover="#235646", accent_soft="#e4eee9", on_accent="#ffffff",
    citation="#27705b",
    warning="#8f5d06", warning_soft="#fbf1dd", danger="#a4332a", danger_soft="#f8e6e3", success="#2f7a4b",
    scrollbar="#d8d2c8", scrollbar_hover="#c1b9ac", log_text="#3f3a33",
    sidebar="#1d2126", sidebar_raised="#272c33", sidebar_input="#1d2126", sidebar_border="#353b43",
    sidebar_text="#f2efe9", sidebar_muted="#9aa1a9", sidebar_danger="#f0998c", nav_active_bg="#2c323a",
    status_bar="#f2efea",
)

PALETTES: Dict[str, Palette] = {"dark": DARK, "light": LIGHT}
DEFAULT_MODE = "dark"

# Geometry. Near-square corners are most of what makes the design read as a document rather than an app.
RADIUS_CONTROL = 3
CARD_RADIUS = 4
PROGRESS_HEIGHT = 3


def build_stylesheet(p: Palette) -> str:
    bar_radius = PROGRESS_HEIGHT // 2
    return f"""
QWidget {{
    background-color: {p.background};
    color: {p.text};
    font-family: {SANS_STACK};
    font-size: 14px;
}}
/* Text never paints its own box, so it sits on whatever card or panel holds it. */
QLabel, QCheckBox, QRadioButton {{
    background-color: transparent;
}}
QToolTip {{
    background-color: {p.surface_raised};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {RADIUS_CONTROL}px;
    padding: 6px 8px;
}}

QLabel[role="heading"] {{
    font-family: {SERIF_STACK};
    font-size: 23px;
    font-weight: 600;
}}
QLabel[role="subtle"] {{
    color: {p.text_muted};
    font-size: 12px;
}}
QLabel[role="metric"] {{
    font-family: {SERIF_STACK};
    font-size: 26px;
    font-weight: 600;
}}
QLabel[role="strong"] {{
    font-weight: 600;
}}
QLabel[role="speaker"] {{
    color: {p.text_muted};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QLabel[role="warning-text"] {{
    color: {p.warning};
}}
QLabel[role="trace"] {{
    color: {p.text_muted};
    font-size: 12px;
    border-left: 2px solid {p.border};
    padding-left: 10px;
}}
QTextBrowser#ReportDocument {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {CARD_RADIUS}px;
    padding: 24px;
    font-size: 14px;
}}
QToolButton#ModeToggle {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS_CONTROL}px;
    padding: 5px;
}}
QToolButton#ModeToggle:hover {{
    background-color: {p.sidebar_raised};
    border-color: {p.sidebar_border};
}}
/* The display-language button shares the toggle's box; its text is the language code ("EN") */
QToolButton#ModeToggle {{
    color: {p.sidebar_muted};
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QToolButton#ModeToggle:hover {{
    color: {p.sidebar_text};
}}
QToolButton#ModeToggle::menu-indicator {{
    image: none;
    width: 0px;
}}
QMenu {{
    background-color: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {RADIUS_CONTROL}px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 2px;
}}
QMenu::item:selected {{
    background-color: {p.accent_soft};
    color: {p.text};
}}

/* ---- sidebar ---- */
QFrame#Sidebar {{
    background-color: {p.sidebar};
    border-right: 1px solid {p.sidebar_border};
}}
QWidget#SidebarBlock {{
    background-color: transparent;
}}
QFrame#Sidebar QLabel {{
    color: {p.sidebar_text};
}}
QFrame#Sidebar QLabel[role="subtle"], QFrame#Sidebar QLabel[role="section"] {{
    color: {p.sidebar_muted};
}}
QLabel[role="section"] {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel[role="brand"] {{
    font-family: {SERIF_STACK};
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0px;
}}
QPushButton[role="nav"] {{
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0px;
    padding: 10px 20px;
    text-align: left;
    color: {p.sidebar_muted};
    font-size: 15px;
    font-weight: 500;
}}
QPushButton[role="nav"]:hover {{
    background-color: {p.sidebar_raised};
    color: {p.sidebar_text};
}}
QPushButton[role="nav"]:checked {{
    background-color: {p.nav_active_bg};
    border-left: 3px solid {p.accent};
    color: {p.sidebar_text};
    font-weight: 600;
}}
QFrame#ApplianceCard {{
    background-color: {p.sidebar_raised};
    border: 1px solid {p.sidebar_border};
    border-radius: {CARD_RADIUS}px;
}}
QFrame#Sidebar QComboBox {{
    background-color: {p.sidebar_input};
    border: 1px solid {p.sidebar_border};
    color: {p.sidebar_text};
}}
QFrame#Sidebar QComboBox:disabled {{
    color: {p.sidebar_muted};
}}
QFrame#Sidebar QPushButton[role="danger-outline"] {{
    border-color: {p.sidebar_border};
    color: {p.sidebar_danger};
}}
QFrame#Sidebar QPushButton[role="danger-outline"]:hover:enabled {{
    background-color: transparent;
    border-color: {p.sidebar_danger};
}}
QFrame#Sidebar QPushButton:disabled {{
    background-color: transparent;
    border: 1px solid {p.sidebar_border};
    color: {p.sidebar_muted};
}}

/* ---- buttons ---- */
QPushButton {{
    background-color: transparent;
    border: 1px solid {p.border_strong};
    border-radius: {RADIUS_CONTROL}px;
    padding: 7px 14px;
    min-height: 20px;
    color: {p.text};
}}
QPushButton:hover:enabled {{
    border-color: {p.border_strong};
    background-color: {p.surface_raised};
}}
QPushButton:disabled {{
    color: {p.text_faint};
    border-color: {p.border};
    background-color: transparent;
}}
QPushButton[role="primary"] {{
    background-color: {p.accent};
    border: 1px solid {p.accent};
    color: {p.on_accent};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover:enabled {{
    background-color: {p.accent_hover};
    border-color: {p.accent_hover};
}}
QPushButton[role="primary"]:disabled {{
    background-color: {p.surface_raised};
    border-color: {p.border};
    color: {p.text_faint};
}}
QPushButton[role="danger-outline"] {{
    background-color: transparent;
    border: 1px solid {p.border_strong};
    color: {p.danger};
    font-weight: 600;
}}
QPushButton[role="danger-outline"]:hover:enabled {{
    background-color: {p.danger_soft};
    border-color: {p.danger};
}}
QPushButton[role="link"] {{
    background: transparent;
    border: none;
    color: {p.citation};
    text-decoration: underline;
    padding: 2px 4px;
}}

/* ---- inputs ---- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {RADIUS_CONTROL}px;
    padding: 7px 10px;
    selection-background-color: {p.accent_soft};
    selection-color: {p.text};
}}
QLineEdit, QSpinBox, QComboBox {{
    min-height: 20px;
}}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QComboBox:hover {{
    border-color: {p.border_strong};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {p.accent};
}}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color: {p.text_faint};
}}
QComboBox {{
    padding-right: 28px;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 26px;
}}
QComboBox::down-arrow {{
    image: url({_ASSETS}/chevron-down.svg);
    width: 12px;
    height: 12px;
}}
QComboBox QAbstractItemView {{
    background-color: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: {RADIUS_CONTROL}px;
    padding: 4px;
    selection-background-color: {p.accent_soft};
    selection-color: {p.text};
    outline: none;
}}
QSpinBox {{
    padding-right: 24px;
}}
QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    width: 22px;
    border: none;
    background: transparent;
}}
QSpinBox::up-button {{ subcontrol-position: top right; margin-top: 3px; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 3px; }}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background-color: {p.surface_raised};
    border-radius: 3px;
}}
QSpinBox::up-arrow {{ image: url({_ASSETS}/chevron-up.svg); width: 10px; height: 10px; }}
QSpinBox::down-arrow {{ image: url({_ASSETS}/chevron-down.svg); width: 10px; height: 10px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px;
    height: 15px;
}}

/* ---- lists ---- */
QListWidget, QListView, QTreeView, QTableView {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {CARD_RADIUS}px;
    padding: 4px;
    outline: none;
}}
QListWidget::item, QListView::item {{
    padding: 7px 8px;
    border-radius: 1px;
}}
QListWidget::item:hover, QListView::item:hover {{
    background-color: {p.surface_raised};
}}
QListWidget::item:selected, QListView::item:selected {{
    background-color: {p.accent_soft};
    color: {p.text};
}}

/* ---- cards ---- */
QFrame[role="card"] {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {CARD_RADIUS}px;
}}
/* The user's turn reads as a quotation rather than a bubble */
QFrame[role="bubble-user"] {{
    background-color: transparent;
    border: none;
    border-left: 3px solid {p.accent};
    border-radius: 0px;
}}
QFrame[role="bubble-assistant"] {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: {CARD_RADIUS}px;
}}

/* ---- banners ---- */
QLabel[role="banner-warning"] {{
    background-color: {p.warning_soft};
    color: {p.warning};
    border-left: 3px solid {p.warning};
    padding: 9px 12px;
    border-radius: {RADIUS_CONTROL}px;
}}
QLabel[role="banner-danger"] {{
    background-color: {p.danger_soft};
    color: {p.danger};
    border-left: 3px solid {p.danger};
    padding: 9px 12px;
    border-radius: {RADIUS_CONTROL}px;
}}
/* Reassurance, not a warning: the accent rule marks it as information */
QLabel[role="notice"] {{
    background-color: {p.surface_raised};
    color: {p.text_muted};
    border-left: 3px solid {p.accent};
    padding: 10px 14px;
    border-radius: {RADIUS_CONTROL}px;
}}
QLabel[role="chip"] {{
    background-color: transparent;
    color: {p.citation};
    border: 1px solid {p.citation};
    border-radius: 2px;
    padding: 1px 6px;
    font-weight: 600;
}}

/* ---- misc ---- */
QScrollArea {{ border: none; background-color: transparent; }}
QScrollArea > QWidget > QWidget {{ background-color: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {p.scrollbar};
    border-radius: 3px;
    min-height: 30px;
    min-width: 30px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {p.scrollbar_hover}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QProgressBar {{
    background-color: {p.surface_raised};
    border: none;
    border-radius: {bar_radius}px;
    max-height: {PROGRESS_HEIGHT}px;
    min-height: {PROGRESS_HEIGHT}px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {p.accent};
    border-radius: {bar_radius}px;
}}

/* A step's bar is the main thing on screen while material is added, so it is thicker, with a visible track */
QProgressBar#StageBar {{
    background-color: {p.border};
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
}}
QProgressBar#StageBar::chunk {{
    border-radius: 3px;
}}

QFrame#StatusBar {{
    background-color: {p.status_bar};
    border-top: 1px solid {p.border};
}}
QFrame#StatusBar QLabel {{
    font-size: 12px;
}}

QSplitter::handle {{ background-color: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QMessageBox, QDialog {{ background-color: {p.background}; }}
"""


def _status_colours(p: Palette) -> Dict[str, str]:
    """Status dot colours, keyed by ApplianceStatus value."""
    return {
        "offline": p.text_muted,
        "booting": p.warning,
        "ready": p.success,
        "working": p.accent,
        "ingesting": p.warning,
        "researching": p.accent,
        "error": p.danger,
    }


def use(mode: str = DEFAULT_MODE) -> Palette:
    """
    Makes `mode` the app's look by rebinding this module's names. Raises ValueError for a mode that does not
    exist, so a typo on the command line fails at startup rather than falling back.
    """
    if mode not in PALETTES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of: {', '.join(MODES)}.")
    palette = PALETTES[mode]

    global MODE, PALETTE, BACKGROUND, SURFACE, SURFACE_RAISED, BORDER, TEXT, TEXT_MUTED, ACCENT
    global ACCENT_DIM, WARNING, DANGER, SUCCESS, CITATION, STYLESHEET, STATUS_COLOURS
    MODE = mode
    PALETTE = palette
    BACKGROUND = palette.background
    SURFACE = palette.surface
    SURFACE_RAISED = palette.surface_raised
    BORDER = palette.border
    TEXT = palette.text
    TEXT_MUTED = palette.text_muted
    ACCENT = palette.accent
    ACCENT_DIM = palette.accent_soft
    WARNING = palette.warning
    DANGER = palette.danger
    SUCCESS = palette.success
    # Citation links and the [Doc N] chips share this so a claim and its evidence read as one thing.
    CITATION = palette.citation
    STYLESHEET = build_stylesheet(palette)
    STATUS_COLOURS = _status_colours(palette)
    return palette


def apply_to(app) -> None:
    """
    Pushes the current look onto a running QApplication. The stylesheet covers the widgets; the palette's
    link colour covers what a stylesheet cannot reach -- links inside a QTextBrowser's document.
    """
    from PySide6.QtGui import QColor, QPalette

    palette = app.palette()
    for role in (QPalette.ColorRole.Link, QPalette.ColorRole.LinkVisited):
        palette.setColor(role, QColor(CITATION))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


use()
