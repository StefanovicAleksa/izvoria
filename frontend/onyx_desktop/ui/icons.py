# Canvas: desktop_icons_v1
# Path: frontend/onyx_desktop/ui/icons.py

"""
Small monochrome line icons (the light/dark toggle), drawn from inline SVG so nothing ships as an image file.

QtSvg is part of PySide6-Essentials, so this adds no dependency. Each icon is rendered in two colours: muted at
rest, full text colour on hover.
"""

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

# 24x24 strokes, Feather-style. `{c}` is the stroke colour.
_PATHS = {
    "sun": (
        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.9 4.9l1.4 1.4"/>'
        '<path d="M17.7 17.7l1.4 1.4"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M4.9 19.1l1.4-1.4"/>'
        '<path d="M17.7 6.3l1.4-1.4"/>'
    ),
    "moon": '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
}

ICON_SIZE = QSize(18, 18)


def _pixmap(name: str, colour: str, size: QSize) -> QPixmap:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{colour}" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_PATHS[name]}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    # Rendered at 2x so it stays crisp on a HiDPI screen
    pixmap = QPixmap(size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size.width(), size.height()))
    painter.end()
    return pixmap


def nav_icon(name: str, muted: str, active: str, size: QSize = ICON_SIZE) -> QIcon:
    """`muted` when the button is unchecked, `active` when checked (the view being shown)."""
    icon = QIcon()
    icon.addPixmap(_pixmap(name, muted, size), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_pixmap(name, active, size), QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(_pixmap(name, active, size), QIcon.Mode.Active, QIcon.State.Off)
    icon.addPixmap(_pixmap(name, active, size), QIcon.Mode.Active, QIcon.State.On)
    return icon
