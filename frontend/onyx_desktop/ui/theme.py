# Canvas: desktop_theme_v1
# Path: frontend/onyx_desktop/ui/theme.py

"""One palette, applied as a stylesheet. Dark, low-chroma, built for long reading sessions."""

BACKGROUND = "#14161a"
SURFACE = "#1c1f25"
SURFACE_RAISED = "#23272f"
BORDER = "#2e333c"
TEXT = "#e6e8ec"
TEXT_MUTED = "#9aa1ad"
ACCENT = "#4c8dff"
ACCENT_DIM = "#2f5fa8"
WARNING = "#e0a33e"
DANGER = "#e05c5c"
SUCCESS = "#4fb477"

# Citation links and the [Doc N] chips share this so a claim and its evidence read as one thing.
CITATION = "#7cc4ff"

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 14px;
}}

QLabel[role="heading"] {{
    font-size: 19px;
    font-weight: 600;
}}
QLabel[role="subtle"] {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}
QLabel[role="metric"] {{
    font-size: 22px;
    font-weight: 600;
}}

/* ---- sidebar ---- */
QFrame#Sidebar {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}
QPushButton[role="nav"] {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: left;
    color: {TEXT_MUTED};
    font-size: 14px;
}}
QPushButton[role="nav"]:hover {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
}}
QPushButton[role="nav"]:checked {{
    background-color: {ACCENT_DIM};
    color: #ffffff;
    font-weight: 600;
}}

/* ---- buttons ---- */
QPushButton {{
    background-color: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 14px;
}}
QPushButton:hover:enabled {{
    border-color: {ACCENT};
}}
QPushButton:disabled {{
    color: #5c626d;
    border-color: #262a31;
}}
QPushButton[role="primary"] {{
    background-color: {ACCENT};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover:enabled {{
    background-color: #5d9aff;
}}
QPushButton[role="primary"]:disabled {{
    background-color: #2b3546;
    color: #7b8394;
}}
QPushButton[role="link"] {{
    background: transparent;
    border: none;
    color: {CITATION};
    text-decoration: underline;
    padding: 2px 4px;
}}

/* ---- inputs ---- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QComboBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 9px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}

/* ---- cards ---- */
QFrame[role="card"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame[role="bubble-user"] {{
    background-color: {ACCENT_DIM};
    border-radius: 10px;
}}
QFrame[role="bubble-assistant"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

/* ---- banners ---- */
QLabel[role="banner-warning"] {{
    background-color: #3a2d14;
    color: {WARNING};
    border-left: 3px solid {WARNING};
    padding: 8px 10px;
    border-radius: 4px;
}}
QLabel[role="banner-danger"] {{
    background-color: #3a1c1c;
    color: {DANGER};
    border-left: 3px solid {DANGER};
    padding: 8px 10px;
    border-radius: 4px;
}}
QLabel[role="chip"] {{
    background-color: {SURFACE_RAISED};
    color: {CITATION};
    border-radius: 4px;
    padding: 2px 7px;
    font-weight: 600;
}}

/* ---- misc ---- */
QScrollArea {{ border: none; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #333944;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #3e4653; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

QPlainTextEdit#LogPane {{
    font-family: "DejaVu Sans Mono", "Monospace";
    font-size: 12px;
    color: #b9c0cc;
}}

QProgressBar {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

QFrame#StatusBar {{
    background-color: {SURFACE};
    border-top: 1px solid {BORDER};
}}

QSplitter::handle {{ background-color: {BORDER}; }}
"""

# Status dot colours, keyed by ApplianceStatus value.
STATUS_COLOURS = {
    "offline": TEXT_MUTED,
    "booting": WARNING,
    "ready": SUCCESS,
    "working": ACCENT,
    "ingesting": WARNING,
    "error": DANGER,
}
