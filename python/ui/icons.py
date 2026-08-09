from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap


def _make_icon(size, draw_fn):
    pix = QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    draw_fn(painter, size)
    painter.end()
    return QIcon(pix)


def _trim_transparent(pix: QPixmap) -> QPixmap:
    """Crop away fully-transparent padding around an icon's glyph, so
    source assets with built-in margins render at the same visual size
    as the hand-drawn icons (which already fill their canvas edge to edge)."""
    img = pix.toImage().convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    if w == 0 or h == 0:
        return pix

    ptr = img.bits()
    ptr.setsize(h * img.bytesPerLine())
    stride = img.bytesPerLine() // 4
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, stride, 4))[:, :w, :]

    opaque = arr[:, :, 3] > 0
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    if not rows.any() or not cols.any():
        return pix

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    cropped = img.copy(int(left), int(top), int(right - left + 1), int(bottom - top + 1))
    return QPixmap.fromImage(cropped)


def _icon_from_file(filename):
    icon_path = Path(__file__).resolve().parent.parent / "icons" / filename
    if not icon_path.exists():
        return None
    pix = QPixmap(str(icon_path))
    if pix.isNull():
        return None
    icon = QIcon(_trim_transparent(pix))
    if icon.isNull():
        return None
    return icon


def icon_pencil(app):
    icon = _icon_from_file("annotate.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 2, QtCore.Qt.SolidLine,
                      QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        p.drawLine(3, s - 3, s - 3, 3)
        p.setPen(QPen(QColor("#d28b36"), 2, QtCore.Qt.SolidLine,
                      QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        p.drawLine(5, s - 5, s - 5, 5)
    return _make_icon(16, draw)


def icon_eraser(app):
    icon = _icon_from_file("eraser.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.5))
        p.setBrush(QColor("#f2b07b"))
        p.drawRoundedRect(3, 6, 10, 6, 2, 2)
        p.setPen(QPen(QColor("#ffffff"), 1.2))
        p.drawLine(4, 7, 12, 11)
    return _make_icon(16, draw)


def icon_repair(app):
    icon = _icon_from_file("repair.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.8, QtCore.Qt.SolidLine,
                      QtCore.Qt.RoundCap))
        p.drawEllipse(2, 2, 6, 6)
        p.drawLine(7, 7, 13, 13)
        p.drawLine(10, 12, 13, 9)
    return _make_icon(16, draw)


def icon_clone(app):
    icon = _icon_from_file("clone.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.5))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(3, 5, 8, 8)
        p.drawRect(6, 2, 8, 8)
    return _make_icon(16, draw)


def icon_palette(app):
    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.2))
        p.setBrush(QColor("#f0d1a0"))
        p.drawEllipse(2, 2, 12, 12)
        p.setBrush(QColor("#d9534f"))
        p.drawEllipse(5, 5, 2, 2)
        p.setBrush(QColor("#5bc0de"))
        p.drawEllipse(9, 5, 2, 2)
        p.setBrush(QColor("#5cb85c"))
        p.drawEllipse(7, 9, 2, 2)
    return _make_icon(16, draw)


def icon_contrast(app):
    icon = _icon_from_file("contrast.png")
    if icon is not None:
        return icon

    def draw(p, s):
        rect = QtCore.QRectF(2, 2, 12, 12)
        p.setPen(QPen(QColor("#2b2b2b"), 1.2))
        p.setBrush(QColor("#222222"))
        p.drawPie(rect, 90 * 16, 180 * 16)
        p.setBrush(QColor("#ffffff"))
        p.drawPie(rect, -90 * 16, 180 * 16)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(rect)
    return _make_icon(16, draw)


def icon_reset_view(app):
    icon = _icon_from_file("reset.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)


def icon_hist(app):
    icon = _icon_from_file("histogram.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.2))
        p.setBrush(QColor("#7aa7d9"))
        p.drawRect(3, 7, 3, 6)
        p.setBrush(QColor("#5cb85c"))
        p.drawRect(7, 4, 3, 9)
        p.setBrush(QColor("#f0ad4e"))
        p.drawRect(11, 6, 3, 7)
    return _make_icon(16, draw)


def icon_prev(app):
    icon = _icon_from_file("previous.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack)


def icon_next(app):
    icon = _icon_from_file("next.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward)


def icon_loop(app):
    icon = _icon_from_file("loop.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)


def icon_reset_contrast(app):
    icon = _icon_from_file("reset-contrast.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)


def icon_eye(app):
    icon = _icon_from_file("view.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(2, 5, 12, 6)
        p.setBrush(QColor("#2b2b2b"))
        p.drawEllipse(7, 7, 2, 2)
    return _make_icon(16, draw)


def icon_zoom(app, plus=True):
    if plus:
        icon = _icon_from_file("zoom-in.png")
    else:
        icon = _icon_from_file("zoom-out.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(2, 2, 9, 9)
        p.drawLine(9, 9, 14, 14)
        p.drawLine(5, 6, 9, 6)
        if plus:
            p.drawLine(7, 4, 7, 8)
    return _make_icon(16, draw)


def icon_revision(app):
    icon = _icon_from_file("revision.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_DriveFDIcon)


def icon_copy_arrow(app):
    icon = _icon_from_file("copy-arrow.png")
    if icon is not None:
        return icon

    def draw(p, s):
        p.setPen(QPen(QColor("#2b2b2b"), 1.5, QtCore.Qt.SolidLine,
                      QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        p.drawLine(2, 8, 12, 8)
        p.drawLine(8, 4, 12, 8)
        p.drawLine(8, 12, 12, 8)
    return _make_icon(16, draw)


def icon_textbox(app):
    icon = _icon_from_file("textbox.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)


def icon_save_comment(app):
    icon = _icon_from_file("save-comment.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)


def icon_export_excel(app):
    icon = _icon_from_file("export-excel.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)


def icon_append_json(app):
    icon = _icon_from_file("append-json.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)


def icon_clear_review(app):
    icon = _icon_from_file("clear-review.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)


def icon_restore_json(app):
    icon = _icon_from_file("restore-json.png")
    if icon is not None:
        return icon
    return app.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
