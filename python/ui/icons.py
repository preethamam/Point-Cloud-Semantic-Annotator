from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def _make_icon(size, draw_fn):
    pix = QPixmap(size, size)
    pix.fill(QtCore.Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    draw_fn(painter, size)
    painter.end()
    return QIcon(pix)


def _icon_from_file(filename):
    icon_path = Path(__file__).resolve().parent.parent / "icons" / filename
    if not icon_path.exists():
        return None
    icon = QIcon(str(icon_path))
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
