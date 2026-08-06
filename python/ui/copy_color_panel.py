from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

MENU_STYLE = """
    QMenu {
        background-color: #ffffff;
        border: 1px solid #d7d7dc;
        border-radius: 8px;
        padding: 0px;
    }
    """

PANEL_STYLE = """
    QWidget#copyPopupRoot {
        background: #ffffff;
        border-radius: 8px;
    }
    QLabel#copyTitle {
        font-size: 12px;
        font-weight: 600;
        color: #1a1a1a;
        background: transparent;
        border: none;
    }
    QLabel#copyHint {
        font-size: 10px;
        color: #7a7a80;
        background: transparent;
        border: none;
    }
    QSpinBox {
        background: #f3f4f6;
        border: 1px solid #d7d7dc;
        border-radius: 6px;
        padding: 3px 4px;
        color: #222;
        selection-background-color: #cfe4fb;
    }
    QSpinBox:hover {
        border: 1px solid #b7b7c0;
    }
    QSpinBox:focus {
        border: 1px solid #0078D4;
        background: #ffffff;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 14px;
        border: none;
        background: transparent;
    }
    QSpinBox::up-arrow {
        image: none;
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-bottom: 4px solid #666666;
        width: 0px;
        height: 0px;
    }
    QSpinBox::down-arrow {
        image: none;
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid #666666;
        width: 0px;
        height: 0px;
    }
    QPushButton#copyValuesBtn {
        background: #f4f4f4;
        color: #000000;
        border: 1px solid #0078D4;
        border-radius: 4px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    QPushButton#copyValuesBtn:hover {
        background: #e6f1fb;
    }
    QPushButton#copyValuesBtn:pressed {
        background: #d7ebfa;
    }
    QPushButton#bulkCopyBtn {
        background: #6b6b70;
        color: #ffffff;
        border: 1px solid #6b6b70;
        border-radius: 4px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 600;
    }
    QPushButton#bulkCopyBtn:hover {
        background: #58585c;
    }
    QPushButton#bulkCopyBtn:pressed {
        background: #47474a;
    }
    """


def build_copy_color_widget(app, on_applied=None) -> QtWidgets.QWidget:
    """Build the 'ignore RGB, copy the rest from the Original Point Cloud'
    panel. Shared by the ribbon Enhancement group and the Enhancement menu
    so both surfaces offer identical behavior and styling."""
    widget = QtWidgets.QWidget()
    widget.setObjectName("copyPopupRoot")
    widget.setStyleSheet(PANEL_STYLE)

    outer = QtWidgets.QVBoxLayout(widget)
    outer.setContentsMargins(14, 12, 14, 12)
    outer.setSpacing(8)

    title = QtWidgets.QLabel("Ignore RGB in annotated cloud")
    title.setObjectName("copyTitle")
    outer.addWidget(title)

    hint = QtWidgets.QLabel("Points with this color will be retained")
    hint.setObjectName("copyHint")
    hint.setWordWrap(True)
    outer.addWidget(hint)

    row = QtWidgets.QHBoxLayout()
    row.setSpacing(6)

    init_r, init_g, init_b = getattr(app, "_copy_ignore_rgb", (0, 0, 0))

    spin_r = QtWidgets.QSpinBox()
    spin_g = QtWidgets.QSpinBox()
    spin_b = QtWidgets.QSpinBox()
    for prefix, spin, init in (("R ", spin_r, init_r), ("G ", spin_g, init_g), ("B ", spin_b, init_b)):
        spin.setRange(0, 255)
        spin.setValue(init)
        spin.setPrefix(prefix)
        spin.setFixedWidth(64)
        spin.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(spin)

    swatch = QtWidgets.QLabel()
    swatch.setFixedSize(24, 24)

    def _update_swatch():
        swatch.setStyleSheet(
            f"background: rgb({spin_r.value()}, {spin_g.value()}, {spin_b.value()});"
            "border: 1px solid #d7d7dc; border-radius: 6px;"
        )
        # Remembered so the panel reopens with the last-used ignore color.
        app._copy_ignore_rgb = (spin_r.value(), spin_g.value(), spin_b.value())

    spin_r.valueChanged.connect(_update_swatch)
    spin_g.valueChanged.connect(_update_swatch)
    spin_b.valueChanged.connect(_update_swatch)
    _update_swatch()

    row.addWidget(swatch)
    outer.addLayout(row)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.setSpacing(6)

    btn_single = QtWidgets.QPushButton("Single File Copy")
    btn_single.setObjectName("copyValuesBtn")
    btn_single.setCursor(QtCore.Qt.PointingHandCursor)
    btn_single.setToolTip("Copy original colors into the current file only")
    btn_row.addWidget(btn_single)

    btn_bulk = QtWidgets.QPushButton("Bulk Copy")
    btn_bulk.setObjectName("bulkCopyBtn")
    btn_bulk.setCursor(QtCore.Qt.PointingHandCursor)
    btn_bulk.setToolTip("Copy original colors into every annotation file")
    btn_row.addWidget(btn_bulk)

    outer.addLayout(btn_row)

    widget.setMinimumWidth(260)

    def _do_single_copy():
        app.copy_color_from_original(spin_r.value(), spin_g.value(), spin_b.value())
        if on_applied is not None:
            on_applied()

    def _do_bulk_copy():
        r, g, b = spin_r.value(), spin_g.value(), spin_b.value()
        if on_applied is not None:
            on_applied()
        app.bulk_copy_colors(r, g, b)

    btn_single.clicked.connect(_do_single_copy)
    btn_bulk.clicked.connect(_do_bulk_copy)

    return widget
