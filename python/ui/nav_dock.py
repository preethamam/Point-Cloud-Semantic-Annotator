from __future__ import annotations

from PyQt5 import QtCore, QtWidgets, QtGui


def build_nav_dock(app) -> None:
    """Patch 2B: Left navigation dock (empty shell)."""
    app.nav_dock = QtWidgets.QDockWidget("Navigation", app)
    app.nav_dock.setMinimumWidth(110)
    app.nav_dock.setMaximumWidth(400)

    app.addDockWidget(QtCore.Qt.LeftDockWidgetArea, app.nav_dock)

    app.nav_dock.setObjectName("NavigationDock")
    app.nav_dock.setAllowedAreas(
        QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
    )
    app.nav_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
    app.nav_dock.setTitleBarWidget(QtWidgets.QWidget())

    container = QtWidgets.QWidget(app.nav_dock)
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(0)

    app.nav_search = QtWidgets.QLineEdit()
    app.nav_search.setPlaceholderText("Go to index or search filename...")
    app.nav_search.returnPressed.connect(app._on_nav_search_entered)
    app.nav_search.installEventFilter(app)

    layout.addWidget(app.nav_search)

    app.nav_status = QtWidgets.QLabel("")
    app.nav_status.setStyleSheet("color: gray; font-size: 10px; padding: 0px;")
    app.nav_status.setMaximumHeight(4)
    layout.addWidget(app.nav_status)

    app.nav_list = QtWidgets.QListWidget()
    app.nav_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    app.nav_list.setUniformItemSizes(True)
    app.nav_list.setAlternatingRowColors(True)
    app.nav_list.currentRowChanged.connect(app._on_nav_row_changed)

    app.nav_list.setSpacing(4)
    app.nav_list.setStyleSheet("""
    QListWidget::item {
        padding: 4px;
    }
    """)

    layout.addWidget(app.nav_list, 1)

    layout.addStretch(0)

    app.nav_dock.setWidget(container)

    app.addDockWidget(QtCore.Qt.LeftDockWidgetArea, app.nav_dock)
    app.nav_dock.setVisible(True)


def nav_display_name(app, name: str) -> str:
    if len(name) <= app.NAV_NAME_MAX:
        return name
    return name[:app.NAV_NAME_MAX - 1] + "."


def make_nav_item_widget(app, idx: int):
    """
    Thumbnail on top, index + filename below,
    with overlay state indicators (dirty / annotated).
    """
    w = QtWidgets.QWidget()
    w.setAttribute(QtCore.Qt.WA_StyledBackground, True)

    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(4, 4, 4, 4)
    lay.setSpacing(4)
    lay.setAlignment(QtCore.Qt.AlignCenter)

    thumb_container = QtWidgets.QFrame()
    thumb_container.setFixedSize(app.NAV_THUMB_SIZE, app.NAV_THUMB_SIZE)
    thumb_container.setStyleSheet("background: transparent;")
    thumb_container.setAttribute(QtCore.Qt.WA_StyledBackground, True)

    thumb_layout = QtWidgets.QStackedLayout(thumb_container)
    thumb_layout.setContentsMargins(0, 0, 0, 0)

    lbl_img = QtWidgets.QLabel()
    lbl_img.setAlignment(QtCore.Qt.AlignCenter)

    icon = app.thumbs.thumb_icon_for_index(idx)
    if icon is not None:
        lbl_img.setPixmap(icon.pixmap(app.NAV_THUMB_SIZE, app.NAV_THUMB_SIZE))

    thumb_layout.addWidget(lbl_img)

    dot_dirty = QtWidgets.QLabel(thumb_container)
    dot_dirty.setFixedSize(10, 10)
    dot_dirty.setStyleSheet("background:red; border-radius:5px;")
    dot_dirty.move(app.NAV_THUMB_SIZE - 10, 2)
    dot_dirty.hide()

    dot_annot = QtWidgets.QLabel(thumb_container)
    dot_annot.setFixedSize(10, 10)
    dot_annot.setStyleSheet("background:green; border-radius:5px;")
    dot_annot.move(app.NAV_THUMB_SIZE - 10, app.NAV_THUMB_SIZE - 10)
    dot_annot.hide()

    name = app.files[idx].name
    txt = f"{idx+1:04d}\n{nav_display_name(app, name)}"

    lbl_txt = QtWidgets.QLabel(txt)
    lbl_txt.setAlignment(QtCore.Qt.AlignCenter)
    lbl_txt.setWordWrap(True)
    lbl_txt.setStyleSheet("font-size:11px;")

    lay.addWidget(thumb_container)
    lay.addWidget(lbl_txt)

    app._nav_item_widgets[idx] = {
        "root": w,
        "img": lbl_img,
        "dirty": dot_dirty,
        "annotated": dot_annot,
    }

    decorate_nav_item(app, idx)

    return w


def decorate_nav_item(app, idx: int) -> None:
    if getattr(app, "_nav_fast_mode", False):
        if not hasattr(app, "nav_list"):
            return
        item = app.nav_list.item(idx)
        if item is None:
            return

        if idx in app._visited:
            item.setBackground(QtGui.QColor("#d0e7ff"))
        else:
            item.setBackground(QtGui.QBrush())

        flags = []
        if idx in app._dirty:
            flags.append("M")
        if idx in app._annotated:
            flags.append("A")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        item.setText(app._nav_row_text(idx) + suffix)
        return

    if not hasattr(app, "_nav_item_widgets"):
        return

    entry = app._nav_item_widgets.get(idx)
    if entry is None:
        return

    root = entry["root"]
    dot_dirty = entry["dirty"]
    dot_annot = entry["annotated"]

    if idx in app._visited:
        root.setStyleSheet("background:#d0e7ff;")
    else:
        root.setStyleSheet("")

    dot_dirty.setVisible(idx in app._dirty)
    dot_annot.setVisible(idx in app._annotated)
