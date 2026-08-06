from __future__ import annotations

from PyQt5 import QtWidgets


def build_review_panel(app) -> QtWidgets.QWidget:
    panel = QtWidgets.QWidget()
    panel.setStyleSheet("background:#efefef; border-top: 1px solid #cfcfcf;")

    v = QtWidgets.QVBoxLayout(panel)
    v.setContentsMargins(0, 4, 8, 6)
    v.setSpacing(2)

    label_row = QtWidgets.QWidget()
    label_row_layout = QtWidgets.QHBoxLayout(label_row)
    label_row_layout.setContentsMargins(0, 0, 0, 0)
    label_row_layout.setSpacing(6)

    label = QtWidgets.QLabel("Comment (this scene):")
    label.setStyleSheet("font-weight:bold; font-size:11px; color:#000000; background:transparent;")
    label_row_layout.addWidget(label)

    label_row_layout.addStretch(1)

    hint = QtWidgets.QLabel("Shift + ← / →  to navigate and comments are auto-saved")
    hint.setStyleSheet("font-size:10px; color:#666666; background:transparent;")
    label_row_layout.addWidget(hint)

    v.addWidget(label_row)

    box = QtWidgets.QPlainTextEdit()
    box.setPlaceholderText("Type review notes...")
    box.setFixedHeight(54)
    box.setStyleSheet("""
        QPlainTextEdit {
            background: white;
            border: 1px solid #cfcfcf;
            border-radius: 3px;
            font-size: 12px;
        }
    """)
    v.addWidget(box)

    app.review_textbox = box
    box.installEventFilter(app)
    box.textChanged.connect(app._on_review_comment_changed)

    return panel
