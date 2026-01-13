from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QPixmap


def build_menubar(app) -> None:
    menubar = app.menuBar()

    file_menu = menubar.addMenu("&File")

    app.act_open_orig = QtWidgets.QAction("Open Original Folder", app)
    app.act_open_orig.triggered.connect(app.open_orig_folder)
    file_menu.addAction(app.act_open_orig)

    app.act_open_ann = QtWidgets.QAction("Open Annotation Folder", app)
    app.act_open_ann.triggered.connect(app.open_ann_folder)
    file_menu.addAction(app.act_open_ann)

    file_menu.addSeparator()

    app.act_save = QtWidgets.QAction("Save", app)
    app.act_save.setShortcut(QKeySequence.Save)
    app.act_save.triggered.connect(app.on_save)
    file_menu.addAction(app.act_save)

    app.act_autosave.setText("Autosave")
    app.act_autosave.setCheckable(True)
    app.act_autosave.setChecked(True)
    file_menu.addAction(app.act_autosave)

    file_menu.addSeparator()

    app.act_clear_thumbs = QtWidgets.QAction("Clear Thumbnail Cache", app)
    app.act_clear_thumbs.triggered.connect(app.thumbs.clear_thumbnail_cache)
    file_menu.addAction(app.act_clear_thumbs)

    file_menu.addSeparator()

    app.act_exit = QtWidgets.QAction("Exit", app)
    app.act_exit.triggered.connect(app.close)
    file_menu.addAction(app.act_exit)

    edit_menu = menubar.addMenu("&Edit")

    app.act_undo = QtWidgets.QAction("Undo", app)
    app.act_undo.setShortcut(QKeySequence.Undo)
    app.act_undo.triggered.connect(app.on_undo)
    edit_menu.addAction(app.act_undo)

    app.act_redo = QtWidgets.QAction("Redo", app)
    app.act_redo.setShortcut(QKeySequence.Redo)
    app.act_redo.triggered.connect(app.on_redo)
    edit_menu.addAction(app.act_redo)

    edit_menu.addSeparator()

    app.act_eraser.setText("Eraser")
    app.act_eraser.setShortcut(QKeySequence("E"))
    app.act_eraser.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    app.act_eraser.toggled.connect(app._on_eraser_toggled)
    edit_menu.addAction(app.act_eraser)

    app.act_repair.setText("Repair Mode")
    app.act_repair.setShortcut(QKeySequence("Shift+R"))
    app.act_repair.toggled.connect(lambda on: on and app.act_clone.setChecked(False))
    app.act_repair.toggled.connect(app.toggle_repair_mode)
    edit_menu.addAction(app.act_repair)

    app.act_clone.setText("Clone Mode")
    app.act_clone.setShortcut(QKeySequence("C"))
    app.act_clone.toggled.connect(lambda on: on and app.act_repair.setChecked(False))
    app.act_clone.toggled.connect(app.toggle_clone_mode)
    edit_menu.addAction(app.act_clone)

    edit_menu.addSeparator()

    color_menu = edit_menu.addMenu("Color")

    app.act_pick_color = QtWidgets.QAction("Pick Color.", app)
    app.act_pick_color.triggered.connect(app.pick_color)
    color_menu.addAction(app.act_pick_color)

    color_menu.addSeparator()

    app._SWATCHES = [
        ("Red", "#FF0000"), ("Green", "#00FF00"), ("Blue", "#0000FF"),
        ("Yellow", "#FFFF00"), ("Cyan", "#00FFFF"), ("Magenta", "#FF00FF"),
        ("Orange", "#FFA500"), ("Pink", "#FFC0CB"), ("Purple", "#800080"),
        ("Brown", "#A52A2A"), ("Maroon", "#800000"), ("Olive", "#808000"),
        ("Teal", "#008080"), ("Navy", "#000080"), ("Gray", "#808080"),
        ("Light Gray", "#D3D3D3"), ("Black", "#000000"), ("White", "#FFFFFF"),
    ]
    for name, hexcol in app._SWATCHES:
        act = QtWidgets.QAction(name, app)
        pix = QPixmap(12, 12)
        pix.fill(QColor(hexcol))
        act.setIcon(QIcon(pix))
        act.triggered.connect(lambda _, c=hexcol: app.select_swatch(c, None))
        color_menu.addAction(act)

    view_menu = menubar.addMenu("&View")

    view_actions = [
        ("Top View", 0, "Ctrl+T"),
        ("Bottom View", 1, "Ctrl+B"),
        ("Front View", 2, "Ctrl+F"),
        ("Back View", 3, "Ctrl+V"),
        ("Left View", 4, "Ctrl+L"),
        ("Right View", 5, "Ctrl+R"),
        ("SW Isometric", 6, "Ctrl+W"),
        ("SE Isometric", 7, "Ctrl+E"),
        ("NW Isometric", 8, "Ctrl+I"),
        ("NE Isometric", 9, "Ctrl+O"),
    ]
    for name, idx, shortcut in view_actions:
        act = QtWidgets.QAction(name, app)
        act.setShortcut(QKeySequence(shortcut))
        act.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        act.triggered.connect(lambda _, i=idx: app._set_view(i))
        view_menu.addAction(act)

    view_menu.addSeparator()

    app.act_zoom_in = QtWidgets.QAction("Zoom In", app)
    app.act_zoom_in.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
    app.act_zoom_in.triggered.connect(app.on_zoom_in)
    view_menu.addAction(app.act_zoom_in)

    app.act_zoom_out = QtWidgets.QAction("Zoom Out", app)
    app.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
    app.act_zoom_out.triggered.connect(app.on_zoom_out)
    view_menu.addAction(app.act_zoom_out)

    app.act_reset_view = QtWidgets.QAction("Reset View", app)
    app.act_reset_view.setShortcut("R")
    app.act_reset_view.triggered.connect(app.reset_view)
    view_menu.addAction(app.act_reset_view)

    view_menu.addSeparator()

    app.act_annotation_mode.setText("Annotation Mode")
    app.act_annotation_mode.setShortcut(QKeySequence("Ctrl+A"))
    app.act_annotation_mode.toggled.connect(app.toggle_annotation)
    view_menu.addAction(app.act_annotation_mode)

    app.act_toggle_annotations.setText("Show Annotations")
    app.act_toggle_annotations.setShortcut(QKeySequence("Shift+A"))
    app.act_toggle_annotations.toggled.connect(app.set_annotations_visible)
    view_menu.addAction(app.act_toggle_annotations)

    view_menu.addSeparator()

    act_toggle_nav = QtWidgets.QAction("Toggle Navigation Pane", app, checkable=True)
    act_toggle_nav.setChecked(True)
    act_toggle_nav.setShortcut(QKeySequence("N"))
    act_toggle_nav.toggled.connect(app.nav_dock.setVisible)
    app.act_toggle_nav = act_toggle_nav
    app.nav_dock.visibilityChanged.connect(app._on_nav_visibility_changed)
    view_menu.addAction(act_toggle_nav)

    playback_menu = menubar.addMenu("&Playback")

    app.act_loop.setText("Loop")
    app.act_loop.setShortcut(QKeySequence("L"))
    app.act_loop.toggled.connect(app._toggle_loop)
    playback_menu.addAction(app.act_loop)

    playback_menu.addSeparator()

    app.act_prev = QtWidgets.QAction("Previous", app)
    app.act_prev.setShortcut(QKeySequence(QtCore.Qt.Key_Left))
    app.act_prev.triggered.connect(app.on_prev)
    playback_menu.addAction(app.act_prev)

    app.act_next = QtWidgets.QAction("Next", app)
    app.act_next.setShortcut(QKeySequence(QtCore.Qt.Key_Right))
    app.act_next.triggered.connect(app.on_next)
    playback_menu.addAction(app.act_next)

    tools_menu = menubar.addMenu("&Tools")

    app.act_auto_contrast = QtWidgets.QAction("Auto Contrast", app)
    app.act_auto_contrast.triggered.connect(app.apply_auto_contrast)
    tools_menu.addAction(app.act_auto_contrast)

    app.act_reset_contrast = QtWidgets.QAction("Reset Contrast", app)
    app.act_reset_contrast.triggered.connect(app.reset_contrast)
    tools_menu.addAction(app.act_reset_contrast)

    tools_menu.addSeparator()

    app.act_hist = QtWidgets.QAction("Show RGB Histograms", app)
    app.act_hist.triggered.connect(app.show_histograms)
    tools_menu.addAction(app.act_hist)

    help_menu = menubar.addMenu("&Help")

    act_about = QtWidgets.QAction("About", app)
    act_about.triggered.connect(app.show_about_dialog)
    help_menu.addAction(act_about)
