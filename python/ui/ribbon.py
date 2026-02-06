from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QToolButton


class RibbonGroup(QtWidgets.QFrame):
    """
    Compact ribbon group:
      - Controls are arranged in a grid (label left, control right)
      - Bottom title is plain text (no boxed footer)
      - add(widget): full-row widget (buttons, checkboxes, etc.)
      - add_row("Label", widget, trailing_widget=None): label+control row
    """
    def __init__(self, title: str, width=150, parent=None, title_position="bottom"):
        super().__init__(parent)

        self.setFixedWidth(width)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #f4f4f4;
                border: 1px solid #cfcfcf;
                border-radius: 4px;
            }
            QLabel {
                color: #222;
            }
        """)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(3)

        self.controls = QtWidgets.QWidget(self)
        self.grid = QtWidgets.QGridLayout(self.controls)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(3)
        self.grid.setColumnStretch(1, 1)
        self.title_lbl = QtWidgets.QLabel(title, self)
        self.title_lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.title_lbl.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #555;
                padding: 0px;
                background: transparent;
                border: none;
            }
        """)
        if title_position == "top":
            outer.addWidget(self.title_lbl, 0)
            outer.addWidget(self.controls, 1)
        else:
            outer.addWidget(self.controls, 1)
            outer.addWidget(self.title_lbl, 0)

        self._row = 0

    def add(self, w):
        """Full-width widget row."""
        self.grid.addWidget(w, self._row, 0, 1, 3)
        self._row += 1

    def add_row(self, label: str, w, trailing=None):
        """Label on left, control on right, optional trailing widget."""
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(
            "font-size: 11px; color: #222; background: transparent; border: none; padding: 0px;"
        )
        lbl.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)

        self.grid.addWidget(lbl, self._row, 0)

        try:
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass

        if trailing is None:
            self.grid.addWidget(w, self._row, 1, 1, 2)
        else:
            self.grid.addWidget(w, self._row, 1)

        if trailing is not None:
            try:
                trailing.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            except Exception:
                pass
            self.grid.addWidget(trailing, self._row, 2)

        self._row += 1


def _ribbon_button(icon, tooltip, checkable=False, icon_size=14, button_size=22):
    btn = QToolButton()
    btn.setIcon(icon)
    btn.setIconSize(QtCore.QSize(icon_size, icon_size))
    btn.setToolTip(tooltip)
    btn.setCheckable(checkable)
    btn.setAutoRaise(True)
    btn.setFixedSize(button_size, button_size)
    btn.setStyleSheet("""
        QToolButton { padding: 1px; }
        QToolButton:checked {
            background: #d0e7ff;
            border: 1px solid #7aa7d9;
            border-radius: 3px;
        }
    """)
    return btn


def build_ribbon(app) -> QtWidgets.QWidget:
    ribbon = QtWidgets.QWidget(app)
    ribbon.setFixedHeight(130)
    ribbon.setStyleSheet("background:#efefef;")

    h = QtWidgets.QHBoxLayout(ribbon)
    h.setContentsMargins(6, 4, 6, 4)
    h.setSpacing(6)
    h.setAlignment(QtCore.Qt.AlignTop)

    nav = RibbonGroup("Navigation", 130, title_position="top")

    btn_prev = _ribbon_button(app._icon_prev(), "Previous (Left Arrow)")
    btn_prev.clicked.connect(app.on_prev)

    btn_next = _ribbon_button(app._icon_next(), "Next (Right Arrow)")
    btn_next.clicked.connect(app.on_next)

    chk_loop = _ribbon_button(app._icon_loop(), "Loop playback", checkable=True)
    chk_loop.setChecked(app.act_loop.isChecked())
    chk_loop.toggled.connect(app.act_loop.setChecked)
    app.act_loop.toggled.connect(chk_loop.setChecked)
    
    btn_revision = _ribbon_button(app._icon_revision(), "Revise / Move To Folder (M)")
    btn_revision.clicked.connect(app.move_current_to_folder)

    delay = QtWidgets.QLineEdit()
    delay.setText(f"{app.loop_delay_sec:.2f}")
    delay.setFixedWidth(64)
    delay.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    delay_validator = QtGui.QDoubleValidator(0.1, 60.0, 2, delay)
    delay_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
    delay.setValidator(delay_validator)

    def _commit_delay():
        text = delay.text().strip()
        try:
            val = float(text)
        except ValueError:
            delay.setText(f"{app.loop_delay_sec:.2f}")
            delay.clearFocus()
            return
        val = max(0.1, min(60.0, val))
        app._set_loop_delay(val)
        delay.setText(f"{val:.2f}")
        delay.clearFocus()

    delay.editingFinished.connect(_commit_delay)

    btn_delay_menu = _ribbon_button(
        app.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
        "Delay presets"
    )
    btn_delay_menu.setPopupMode(QToolButton.InstantPopup)
    btn_delay_menu.setFixedSize(24, 22)

    delay_menu = QtWidgets.QMenu(btn_delay_menu)
    delay_menu.setStyleSheet("""
        QMenu {
            background-color: #f4f4f4;
            color: #222;
        }
        QMenu::item {
            background-color: transparent;
            color: #222;
            padding: 4px 20px 4px 20px;
        }
        QMenu::item:selected {
            background-color: #d0e7ff;
            color: #222;
        }
        """)

    def _set_delay(val):
        val = float(val)
        app._set_loop_delay(val)
        delay.setText(f"{val:.2f}")
        delay.clearFocus()

    for v in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        act = QtWidgets.QAction(f"{v:.1f} s", app)
        act.triggered.connect(lambda _, x=v: _set_delay(x))
        delay_menu.addAction(act)

    delay_menu.addSeparator()

    def _custom_delay():
        val, ok = QtWidgets.QInputDialog.getDouble(
            app,
            "Loop Delay",
            "Seconds:",
            app.loop_delay_sec,
            0.1,
            60.0,
            1,
        )
        if ok:
            _set_delay(val)

    act_custom = QtWidgets.QAction("Custom.", app)
    act_custom.triggered.connect(_custom_delay)
    delay_menu.addAction(act_custom)

    btn_delay_menu.setMenu(delay_menu)

    delay_row = QtWidgets.QWidget()
    hl = QtWidgets.QHBoxLayout(delay_row)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(2)
    hl.addWidget(delay)
    hl.addWidget(btn_delay_menu)

    nav_row = QtWidgets.QWidget()
    nav_row_layout = QtWidgets.QHBoxLayout(nav_row)
    nav_row_layout.setContentsMargins(0, 0, 0, 0)
    nav_row_layout.setSpacing(2)
    nav_row_layout.addWidget(btn_prev)
    nav_row_layout.addWidget(btn_next)    
    nav_row_layout.addWidget(chk_loop)
    nav_row_layout.addWidget(btn_revision)

    nav.add(nav_row)
    delay_container = QtWidgets.QWidget()
    delay_layout = QtWidgets.QHBoxLayout(delay_container)
    delay_layout.setContentsMargins(0, 0, 0, 0)
    delay_layout.setSpacing(2)

    delay_lbl = QtWidgets.QLabel("Delay:")
    delay_lbl.setStyleSheet(
        "font-size: 11px; color: #222; background: transparent; border: none; padding: 0px;"
    )
    delay_lbl.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)

    delay_layout.addWidget(delay_lbl)
    delay_layout.addWidget(delay_row)

    nav.add(delay_container)

    ann = RibbonGroup("Annotation", 200)

    s_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    s_alpha.setRange(0, 100)
    s_alpha.setValue(int(app.annotation_alpha * 100))
    s_alpha.setStyleSheet("""
        QSlider::groove:horizontal {
            background: #d0d0d0;
            height: 4px;
            border-radius: 2px;
        }
        QSlider::sub-page:horizontal {
            background: #8a8a8a;
            height: 4px;
            border-radius: 2px;
        }
        QSlider::add-page:horizontal {
            background: #d0d0d0;
            height: 4px;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #000000;
            border: 1px solid #000000;
            width: 10px;
            height: 10px;
            margin: -4px 0;
            border-radius: 5px;
        }
    """)

    lbl_alpha = QtWidgets.QLabel(f"{int(app.annotation_alpha * 100)}%")
    lbl_alpha.setFixedWidth(36)
    lbl_alpha.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    s_alpha.valueChanged.connect(app._on_ribbon_alpha)
    ann.add_row("Alpha (A +/-)", s_alpha, lbl_alpha)

    s_brush = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    s_brush.setRange(1, 200)
    s_brush.setValue(int(app.brush_size))
    s_brush.setStyleSheet(s_alpha.styleSheet())

    lbl_brush = QtWidgets.QLabel(f"{int(app.brush_size)} px")
    lbl_brush.setFixedWidth(36)
    lbl_brush.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    s_brush.valueChanged.connect(app._on_ribbon_brush)
    ann.add_row("Brush (B +/-)", s_brush, lbl_brush)

    s_point = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    s_point.setRange(1, 20)
    s_point.setValue(app.point_size)
    s_point.setStyleSheet(s_alpha.styleSheet())

    lbl_point = QtWidgets.QLabel(f"{app.point_size} px")
    lbl_point.setFixedWidth(36)
    lbl_point.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    s_point.valueChanged.connect(app._on_ribbon_point)
    ann.add_row("Point (D +/-)", s_point, lbl_point)

    app.ribbon_sliders = {
        "alpha": (s_alpha, lbl_alpha),
        "brush": (s_brush, lbl_brush),
        "point": (s_point, lbl_point),
    }

    col = RibbonGroup("Colors", 170)

    swatches = QtWidgets.QWidget()
    g = QtWidgets.QGridLayout(swatches)
    g.setContentsMargins(0, 0, 0, 0)
    g.setHorizontalSpacing(3)
    g.setVerticalSpacing(3)

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
        "#00FFFF", "#FF00FF", "#FFA500", "#800080",
        "#A52A2A", "#808080", "#000000", "#FFFFFF",
        "#008080", "#000080", "#808000", "#FFC0CB",
        "#C0C0C0", "#FFD700", "#4B0082", "#2E8B57",
        "#DC143C", "#4682B4", "#9ACD32", "#8B4513",
        "#7FFF00", "#00CED1", "#FF1493", "#708090", 
        "#FFDAB9",
    ]

    cols = 6
    swatch_group = QtWidgets.QButtonGroup(col)
    swatch_group.setExclusive(True)
    for i, c in enumerate(colors):
        b = QtWidgets.QPushButton()
        b.setCheckable(True)
        b.setAutoExclusive(False)
        b.setFixedSize(16, 16)
        b.setStyleSheet(f"""
            QPushButton {{
                background: {c};
                border: 1px solid #777;
                border-radius: 2px;
            }}
            QPushButton:checked {{
                border: 3px solid #00E5FF;
                padding: -1px;
                background: {c};
            }}
            QPushButton:hover {{
                border-color: #00E5FF;
            }}
        """)
        swatch_group.addButton(b)
        b.clicked.connect(lambda _, x=c: app.select_swatch(x))
        g.addWidget(b, i // cols, i % cols)

    pick_btn = QtWidgets.QPushButton()
    pick_btn.setFixedSize(16,16)
    pick_btn.setToolTip("Pick color")
    pick_btn.setFlat(True)
    pick_btn.setAutoFillBackground(False)
    pick_btn.setStyleSheet("""
        QPushButton { border: 1px solid transparent; border-radius: 2px; }
        QPushButton:hover { border-color: #777; }
        QPushButton:pressed { border-color: #00E5FF; }
    """)
    icon_path = Path(__file__).resolve().parent.parent / "icons" / "color-pick.png"
    if icon_path.exists():
        pick_btn.setIcon(QIcon(str(icon_path)))
        pick_btn.setIconSize(QtCore.QSize(16, 16))
    pick_btn.clicked.connect(app.pick_color)
    g.addWidget(pick_btn, len(colors) // cols, cols - 1)

    col.add(swatches)
    edit = RibbonGroup("Edit", 130)

    chk_ann = _ribbon_button(app._icon_pencil(), "Annotation mode (A)", checkable=True)
    chk_ann.setChecked(app.act_annotation_mode.isChecked())
    chk_ann.toggled.connect(app.act_annotation_mode.setChecked)
    app.act_annotation_mode.toggled.connect(chk_ann.setChecked)

    chk_eraser = _ribbon_button(app._icon_eraser(), "Eraser", checkable=True)
    chk_eraser.setChecked(app.act_eraser.isChecked())
    chk_eraser.toggled.connect(app.act_eraser.setChecked)
    app.act_eraser.toggled.connect(chk_eraser.setChecked)

    chk_repair = _ribbon_button(app._icon_repair(), "Repair", checkable=True)
    chk_repair.setChecked(app.act_repair.isChecked())
    chk_repair.toggled.connect(app.act_repair.setChecked)
    app.act_repair.toggled.connect(chk_repair.setChecked)

    chk_clone = _ribbon_button(app._icon_clone(), "Clone", checkable=True)
    chk_clone.setChecked(app.act_clone.isChecked())
    chk_clone.toggled.connect(app.act_clone.setChecked)
    app.act_clone.toggled.connect(chk_clone.setChecked)

    edit_row = QtWidgets.QWidget()
    edit_row_layout = QtWidgets.QHBoxLayout(edit_row)
    edit_row_layout.setContentsMargins(0, 0, 0, 0)
    edit_row_layout.setSpacing(2)
    edit_row_layout.addWidget(chk_ann)
    edit_row_layout.addWidget(chk_eraser)
    edit_row_layout.addWidget(chk_repair)
    edit_row_layout.addWidget(chk_clone)
    edit.add(edit_row)

    enh = RibbonGroup("Enhancement", 195, title_position="bottom")
    enh.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    enh.controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    enh.grid.setAlignment(QtCore.Qt.AlignTop)
    # Height alignment is handled after layout is assembled.

    s_gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    s_gamma.setRange(10, 300)
    s_gamma.setValue(100)
    s_gamma.valueChanged.connect(app.on_gamma_change)
    s_gamma.setStyleSheet(s_alpha.styleSheet())

    app.ribbon_gamma_slider = s_gamma
    app.ribbon_gamma_label = QtWidgets.QLabel("1.00")
    app.ribbon_gamma_label.setStyleSheet("font-size: 11px; color: #222;")
    app.ribbon_gamma_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

    btn_auto = _ribbon_button(app._icon_contrast(), "Auto contrast", icon_size=16, button_size=24)
    btn_auto.clicked.connect(app.apply_auto_contrast)

    btn_reset = _ribbon_button(app._icon_reset_contrast(), "Reset contrast", icon_size=16, button_size=24)
    btn_reset.clicked.connect(app.reset_contrast)

    btn_hist = _ribbon_button(app._icon_hist(), "Show histograms", icon_size=16, button_size=24)
    btn_hist.clicked.connect(app.show_histograms)

    enh.add_row("Gamma (G +/-)", s_gamma, app.ribbon_gamma_label)
    app.ribbon_sliders["gamma"] = (s_gamma, app.ribbon_gamma_label)
    enh_row = QtWidgets.QWidget()
    enh_row_layout = QtWidgets.QHBoxLayout(enh_row)
    enh_row_layout.setContentsMargins(0, 0, 0, 0)
    enh_row_layout.setSpacing(4)
    enh_row_layout.addWidget(btn_auto)
    enh_row_layout.addWidget(btn_reset)
    enh_row_layout.addWidget(btn_hist)
    enh.add(enh_row)

    view = RibbonGroup("View", 195, title_position="bottom")
    view.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    view.controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    view.grid.setAlignment(QtCore.Qt.AlignTop)
    # Height alignment is handled after layout is assembled.

    btn_reset = _ribbon_button(app._icon_reset_view(), "Reset view (R)")
    btn_reset.clicked.connect(app.reset_view)

    btn_zoom_in = _ribbon_button(app._icon_zoom(True), "Zoom in")
    btn_zoom_in.clicked.connect(app.on_zoom_in)

    btn_zoom_out = _ribbon_button(app._icon_zoom(False), "Zoom out")
    btn_zoom_out.clicked.connect(app.on_zoom_out)

    cmb_view = QtWidgets.QComboBox()
    cmb_view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    view_items = [
        ("Top View", 0),
        ("Bottom View", 1),
        ("Front View", 2),
        ("Back View", 3),
        ("Left View", 4),
        ("Right View", 5),
        ("SW Isometric", 6),
        ("SE Isometric", 7),
        ("NW Isometric", 8),
        ("NE Isometric", 9),
    ]

    for name, idx in view_items:
        cmb_view.addItem(name, idx)

    cmb_view.setCurrentIndex(app.current_view)
    app.ribbon_view_combo = cmb_view

    def _on_view_changed(i):
        app._set_view(cmb_view.itemData(i))
        app._release_view_combo_focus()

    cmb_view.currentIndexChanged.connect(_on_view_changed)

    chk_toggle = _ribbon_button(app._icon_eye(), "Show annotations", checkable=True)
    app.toggle_ann_chk = chk_toggle
    chk_toggle.setChecked(app.act_toggle_annotations.isChecked())
    chk_toggle.toggled.connect(app.act_toggle_annotations.setChecked)
    app.act_toggle_annotations.toggled.connect(chk_toggle.setChecked)

    view_row = QtWidgets.QWidget()
    view_row_layout = QtWidgets.QHBoxLayout(view_row)
    view_row_layout.setContentsMargins(0, 0, 0, 0)
    view_row_layout.setSpacing(4)
    view_row_layout.addWidget(btn_reset)
    view_row_layout.addWidget(btn_zoom_in)
    view_row_layout.addWidget(btn_zoom_out)
    view_row_layout.addWidget(chk_toggle)

    view.add(view_row)
    spacer = QtWidgets.QWidget()
    spacer.setFixedHeight(4)
    view.add(spacer)
    view.add(cmb_view)

    nav_edit = QtWidgets.QWidget()
    nav_edit_layout = QtWidgets.QVBoxLayout(nav_edit)
    nav_edit_layout.setContentsMargins(0, 0, 0, 0)
    nav_edit_layout.setSpacing(4)
    nav_edit_layout.addWidget(nav)
    nav_edit_layout.addWidget(edit)

    h.addWidget(nav_edit, 0, QtCore.Qt.AlignTop)
    for grp in (ann, col, enh, view):
        h.addWidget(grp, 0, QtCore.Qt.AlignTop)

    nav_edit.adjustSize()
    target_height = max(
        nav_edit.sizeHint().height(),
        ann.sizeHint().height(),
        col.sizeHint().height(),
        enh.sizeHint().height(),
        view.sizeHint().height(),
    )
    if target_height > 0:
        nav_height = nav.sizeHint().height()
        spacing = nav_edit_layout.spacing()
        edit_height = max(0, target_height - nav_height - spacing)
        edit.setFixedHeight(edit_height)
        nav_edit.setFixedHeight(target_height)
        for grp in (ann, col, enh, view):
            grp.setFixedHeight(target_height)

    h.addStretch(1)
    return ribbon
