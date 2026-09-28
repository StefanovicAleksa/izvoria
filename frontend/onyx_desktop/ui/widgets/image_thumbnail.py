# Path: frontend/onyx_desktop/ui/widgets/image_thumbnail.py

"""
A small preview of a cited image. An image's excerpt is a machine description, so the picture itself is the
quickest check a reader has that the description is right -- it is shown beside the excerpt, not behind a click.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import QLabel

THUMBNAIL_MAX_SIDE = 180


def load_thumbnail(path: Optional[str], max_side: int = THUMBNAIL_MAX_SIDE) -> Optional[QPixmap]:
    """
    The image at `path` scaled to fit `max_side`, or None when it can't be decoded. Only the scaled size is
    decoded (QImageReader.setScaledSize), so a 48 MP photo never lands in memory at full size; EXIF rotation is
    applied so a phone photo isn't shown on its side.
    """
    if not path:
        return None
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_side:
        reader.setScaledSize(size.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio))
    image = reader.read()
    if image.isNull():
        return None
    pixmap = QPixmap.fromImage(image)
    if max(pixmap.width(), pixmap.height()) > max_side:
        pixmap = pixmap.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
    return pixmap


def set_thumbnail(label: QLabel, path: Optional[str], max_side: int = THUMBNAIL_MAX_SIDE) -> bool:
    """Shows the preview in `label` (hidden when there is none). Returns whether a preview is shown."""
    pixmap = load_thumbnail(path, max_side)
    if pixmap is None:
        label.clear()
        label.setVisible(False)
        return False
    label.setPixmap(pixmap)
    label.setToolTip(path)
    label.setVisible(True)
    return True
