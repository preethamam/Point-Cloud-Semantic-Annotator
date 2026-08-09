from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QPixmap

from ui import copy_color_panel


def build_menubar(app) -> None:
    menubar = app.menuBar()
    menubar.setContentsMargins(0, 0, 0, 0)

    file_menu = menubar.addMenu("&File")

    app.act_open_orig = QtWidgets.QAction("Open Original Folder", app)
    app.act_open_orig.triggered.connect(app.open_orig_folder)
    file_menu.addAction(app.act_open_orig)

    app.act_open_ann = QtWidgets.QAction("Open Annotation Folder", app)
    app.act_open_ann.triggered.connect(app.open_ann_folder)
    file_menu.addAction(app.act_open_ann)

    app.act_refresh_folders = QtWidgets.QAction("Refresh Folders", app)
    app.act_refresh_folders.triggered.connect(app.refresh_folders)
    file_menu.addAction(app.act_refresh_folders)

    app.act_select_move_folder = QtWidgets.QAction("Select Revise / Move To Folder", app)
    app.act_select_move_folder.setShortcut(QKeySequence("Ctrl+Shift+M"))
    app.act_select_move_folder.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    app.act_select_move_folder.triggered.connect(app.select_revise_move_folder)
    file_menu.addAction(app.act_select_move_folder)

    app.act_move_to_folder = QtWidgets.QAction("Revise / Move To Folder", app)
    app.act_move_to_folder.setShortcut(QKeySequence("Ctrl+M"))
    app.act_move_to_folder.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    # app.act_move_to_folder.setIcon(app._icon_revision())
    app.act_move_to_folder.triggered.connect(app.move_current_to_folder)
    file_menu.addAction(app.act_move_to_folder)

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
    app.act_eraser.setShortcut(QKeySequence("Ctrl+Shift+E"))
    app.act_eraser.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    app.act_eraser.toggled.connect(app._on_eraser_toggled)
    edit_menu.addAction(app.act_eraser)

    app.act_repair.setText("Repair Mode")
    app.act_repair.setShortcut(QKeySequence("Ctrl+Shift+R"))
    app.act_repair.toggled.connect(lambda on: on and app.act_clone.setChecked(False))
    app.act_repair.toggled.connect(app.toggle_repair_mode)
    edit_menu.addAction(app.act_repair)

    app.act_clone.setText("Clone Mode")
    app.act_clone.setShortcut(QKeySequence("Ctrl+Shift+C"))
    app.act_clone.toggled.connect(lambda on: on and app.act_repair.setChecked(False))
    app.act_clone.toggled.connect(app.toggle_clone_mode)
    edit_menu.addAction(app.act_clone)

    edit_menu.addSeparator()

    color_menu = edit_menu.addMenu("Color")

    app.act_pick_color = QtWidgets.QAction("Pick Color", app)
    pick_icon = QIcon(QPixmap("icons/color-pick.png").scaled(14, 14, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    app.act_pick_color.setIcon(pick_icon)
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
        pix = QPixmap(14, 14)
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
    app.act_reset_view.setShortcut(QKeySequence("Ctrl+Shift+V"))
    app.act_reset_view.triggered.connect(app.reset_view)
    view_menu.addAction(app.act_reset_view)

    view_menu.addSeparator()

    app.act_annotation_mode.setText("Annotation Mode")
    app.act_annotation_mode.setShortcut(QKeySequence("Ctrl+Alt+A"))
    app.act_annotation_mode.toggled.connect(app.toggle_annotation)
    view_menu.addAction(app.act_annotation_mode)

    app.act_toggle_annotations.setText("Show Annotations")
    app.act_toggle_annotations.setShortcut(QKeySequence("Ctrl+A"))
    app.act_toggle_annotations.toggled.connect(app.set_annotations_visible)
    view_menu.addAction(app.act_toggle_annotations)

    view_menu.addSeparator()

    app.act_points_spheres.setText("Render Points as Spheres")
    app.act_points_spheres.setShortcut(QKeySequence("Ctrl+Shift+P"))
    app.act_points_spheres.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    app.act_points_spheres.toggled.connect(app.set_points_render_mode)
    view_menu.addAction(app.act_points_spheres)

    view_menu.addSeparator()

    act_toggle_nav = QtWidgets.QAction("Toggle Navigation Pane", app, checkable=True)
    act_toggle_nav.setChecked(True)
    act_toggle_nav.setShortcut(QKeySequence("Ctrl+N"))
    act_toggle_nav.toggled.connect(app.nav_dock.setVisible)
    app.act_toggle_nav = act_toggle_nav
    app.nav_dock.visibilityChanged.connect(app._on_nav_visibility_changed)
    view_menu.addAction(act_toggle_nav)

    playback_menu = menubar.addMenu("&Playback")

    app.act_loop.setText("Loop")
    app.act_loop.setShortcut(QKeySequence("Ctrl+Shift+L"))
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

    enhancement_menu = menubar.addMenu("&Enhancement")

    app.act_auto_contrast = QtWidgets.QAction("Auto Contrast", app)
    app.act_auto_contrast.triggered.connect(app.apply_auto_contrast)
    enhancement_menu.addAction(app.act_auto_contrast)

    app.act_reset_contrast = QtWidgets.QAction("Reset Contrast", app)
    app.act_reset_contrast.triggered.connect(app.reset_contrast)
    enhancement_menu.addAction(app.act_reset_contrast)

    enhancement_menu.addSeparator()

    app.act_hist = QtWidgets.QAction("Show RGB Histograms", app)
    app.act_hist.triggered.connect(app.show_histograms)
    enhancement_menu.addAction(app.act_hist)

    enhancement_menu.addSeparator()

    copy_colors_menu = enhancement_menu.addMenu("Copy Original Point Cloud Colors")
    copy_colors_menu.setStyleSheet(copy_color_panel.MENU_STYLE)

    def _close_all_menus():
        # copy_colors_menu is a submenu of enhancement_menu; closing just
        # the submenu leaves its parent menu-bar dropdown open. Closing
        # every currently active popup collapses the whole chain.
        w = QtWidgets.QApplication.activePopupWidget()
        while w is not None:
            w.close()
            w = QtWidgets.QApplication.activePopupWidget()

    copy_colors_widget = copy_color_panel.build_copy_color_widget(
        app, on_applied=_close_all_menus
    )
    copy_colors_action = QtWidgets.QWidgetAction(copy_colors_menu)
    copy_colors_action.setDefaultWidget(copy_colors_widget)
    copy_colors_menu.addAction(copy_colors_action)

    review_menu = menubar.addMenu("&Review")

    app.act_show_review_textbox = QtWidgets.QAction("Show Review Textbox", app, checkable=True)
    app.act_show_review_textbox.setShortcut(QKeySequence("Ctrl+Shift+T"))
    app.act_show_review_textbox.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    app.act_show_review_textbox.toggled.connect(app.toggle_review_textbox)
    review_menu.addAction(app.act_show_review_textbox)

    review_menu.addSeparator()

    app.act_save_comment = QtWidgets.QAction("Save Comment", app)
    app.act_save_comment.triggered.connect(app.save_comment)
    review_menu.addAction(app.act_save_comment)

    app.act_export_excel = QtWidgets.QAction("Export to Excel", app)
    app.act_export_excel.triggered.connect(app.export_to_excel)
    review_menu.addAction(app.act_export_excel)

    review_menu.addSeparator()

    app.act_append_review = QtWidgets.QAction("Append Review Data (Cumulative)", app, checkable=True)
    app.act_append_review.setToolTip(
        "When on, review data (comments + stats) keeps accumulating across "
        "folders and app runs instead of resetting on each new folder."
    )
    # Reflect the persisted value before wiring the handler so restoring the
    # checkbox at startup doesn't trigger a redundant state write.
    app.act_append_review.setChecked(app._is_review_append_mode())
    app.act_append_review.toggled.connect(app._toggle_review_append_mode)
    review_menu.addAction(app.act_append_review)

    app.act_clear_review = QtWidgets.QAction("Clear Review Data", app)
    app.act_clear_review.triggered.connect(app.clear_review_data)
    review_menu.addAction(app.act_clear_review)

    app.act_restore_review = QtWidgets.QAction("Restore Review Data", app)
    app.act_restore_review.triggered.connect(app.restore_review_data)
    review_menu.addAction(app.act_restore_review)

    help_menu = menubar.addMenu("&Help")

    act_about = QtWidgets.QAction("About", app)
    act_about.triggered.connect(app.show_about_dialog)
    help_menu.addAction(act_about)

    _install_ribbon_options_button(app, menubar)

    esc_shortcut = QtWidgets.QShortcut(QKeySequence(QtCore.Qt.Key_Escape), app)
    esc_shortcut.activated.connect(
        lambda: app.isFullScreen() and app._set_ribbon_display_mode("shown")
    )


class _RibbonButtonResizer(QtCore.QObject):
    """Keeps the ribbon-options chevron pinned to the menu bar's right edge.
    setCornerWidget()'s exact edge alignment is style-dependent (per Qt
    docs) and left a visible gap under native Windows theming, so this
    positions the button with absolute coordinates instead, recomputed on
    every menu bar resize."""

    def __init__(self, menubar, btn, margin: int):
        super().__init__(menubar)
        self.menubar = menubar
        self.btn = btn
        self.margin = margin

    def reposition(self):
        x = max(0, self.menubar.width() - self.btn.width() - self.margin)
        y = max(0, (self.menubar.height() - self.btn.height()) // 2)
        self.btn.move(x, y)
        self.btn.raise_()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            self.reposition()
        return False


def _install_ribbon_options_button(app, menubar) -> None:
    """Small chevron button (top-right of the menu bar) offering Word-style
    ribbon display options: full-screen, hide ribbon, or always show it."""
    btn = QtWidgets.QToolButton(menubar)
    btn.setText("▾")
    btn.setToolTip("Ribbon Display Options")
    btn.setAutoRaise(True)
    btn.setCursor(QtCore.Qt.PointingHandCursor)
    btn.setFixedSize(30, menubar.sizeHint().height())
    btn.setStyleSheet("""
        QToolButton {
            border: none;
            background: transparent;
            color: #333;
            font-size: 16px;
            font-weight: bold;
            padding: 0px;
            margin: 0px;
        }
        QToolButton:hover {
            color: #000000;
        }
    """)

    menu = QtWidgets.QMenu(btn)

    def _show_menu():
        pos = btn.mapToGlobal(QtCore.QPoint(0, btn.height()))
        menu.exec_(pos)

    btn.clicked.connect(_show_menu)

    act_fullscreen = QtWidgets.QAction("Full-screen Mode", app, checkable=True)
    act_hide_ribbon = QtWidgets.QAction("Hide Ribbon", app, checkable=True)
    act_show_ribbon = QtWidgets.QAction("Always Show Ribbon", app, checkable=True)
    act_show_ribbon.setChecked(True)

    group = QtWidgets.QActionGroup(app)
    group.setExclusive(True)
    for act in (act_fullscreen, act_hide_ribbon, act_show_ribbon):
        group.addAction(act)
        menu.addAction(act)

    act_fullscreen.triggered.connect(lambda: app._set_ribbon_display_mode("fullscreen"))
    act_hide_ribbon.triggered.connect(lambda: app._set_ribbon_display_mode("hidden"))
    act_show_ribbon.triggered.connect(lambda: app._set_ribbon_display_mode("shown"))

    app.act_ribbon_fullscreen = act_fullscreen
    app.act_ribbon_hidden = act_hide_ribbon
    app.act_ribbon_shown = act_show_ribbon

    app._ribbon_btn_resizer = _RibbonButtonResizer(menubar, btn, margin=-4)
    menubar.installEventFilter(app._ribbon_btn_resizer)

    btn.show()
    app._ribbon_btn_resizer.reposition()
