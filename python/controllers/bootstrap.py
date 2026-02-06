from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from configs.constants import NAV_DOCK_WIDTH, NAV_NAME_MAX, NAV_THUMB_SIZE, NAV_FAST_THRESHOLD, NAV_FAST_ICON_BATCH
from services.storage import load_state, log_gui, save_state
from controllers import app_helpers
from services.thumbnail import ThumbnailService


def init_actions(app) -> None:
    app.act_annotation_mode = QtWidgets.QAction(app)
    app.act_annotation_mode.setCheckable(True)
    app.act_annotation_mode.setChecked(True)

    app.act_toggle_annotations = QtWidgets.QAction(app)
    app.act_toggle_annotations.setCheckable(True)
    app.act_toggle_annotations.setChecked(True)

    app.act_points_spheres = QtWidgets.QAction(app)
    app.act_points_spheres.setCheckable(True)
    app.act_points_spheres.setChecked(True)

    app.act_clone = QtWidgets.QAction(app)
    app.act_clone.setCheckable(True)

    app.act_repair = QtWidgets.QAction(app)
    app.act_repair.setCheckable(True)

    app.act_loop = QtWidgets.QAction(app)
    app.act_loop.setCheckable(True)
    app.act_loop.setChecked(False)

    app.act_eraser = QtWidgets.QAction(app)
    app.act_eraser.setCheckable(True)

    app.act_autosave = QtWidgets.QAction(app)


def init_state(app) -> None:
    app.NAV_THUMB_SIZE = NAV_THUMB_SIZE
    app.NAV_NAME_MAX = NAV_NAME_MAX
    app.NAV_FAST_THRESHOLD = NAV_FAST_THRESHOLD
    app.NAV_FAST_ICON_BATCH = NAV_FAST_ICON_BATCH
    app.brush_size = 8
    app.initial_loop_timer = 1.0
    app.point_size = 6
    app.current_color = [255, 0, 0]
    app.history, app.redo_stack = [], []
    app._waiting = None
    app.directory, app.files, app.index = None, [], 0
    app._last_gamma_value = 100
    app.ann_dir, app.orig_dir = None, None
    app.annotations_visible = True
    app._session_edited = None
    app.annotation_alpha = 1.0
    app.repair_mode = False
    app.clone_mode = False
    app._is_closing = False

    app._shared_camera = None
    app._cam_observer_id = None
    app._cam_syncing = False

    app._batch = False
    app._cam_pause = False
    app._in_zoom = False
    app._view_change_active = False
    app._need_split_fit = False

    app._fit_pad = 1.08

    app.current_view = 0

    app.current_color = [255, 0, 0]
    app._last_paint_color = app.current_color.copy()

    app.loop_delay_sec = float(app.initial_loop_timer)

    app._visited = set()
    app._annotated = set()
    app._dirty = set()

    app._fit_delay_ms = 33

    app._paint_step_frac = 0.33
    app._brush_coverage = 1.25
    app._last_paint_xy = None
    app._in_stroke = False

    app._constrain_line = False
    app._anchor_xy = None
    app._line_len_px = 0.0

    app._stroke_active = False
    app._stroke_idxs = set()
    app._colors_before_stroke = None
    app._expecting_ann = False
    app._pending_orig_dir = None
    app._nav_last_width = NAV_DOCK_WIDTH
    app._nav_was_visible = True
    app._nav_fast_mode = False
    app._nav_fast_icon_timer = None
    app._nav_fast_icon_idx = 0


def init_timers(app) -> None:
    app._fit_timer = QtCore.QTimer(app)
    app._fit_timer.setSingleShot(True)
    app._fit_timer.timeout.connect(app._fit_to_canvas)

    app.thumbs = ThumbnailService(app, NAV_THUMB_SIZE)

    app._stroke_render_timer = QtCore.QTimer(app)
    app._stroke_render_timer.setSingleShot(True)
    app._stroke_render_timer.timeout.connect(app._render_views_once)

    app._loop_timer = QtCore.QTimer(app)
    app._loop_timer.setSingleShot(False)
    app._loop_timer.timeout.connect(app._on_loop_tick)

    app._paint_timer = QtCore.QElapsedTimer()
    app._paint_timer.start()
    app._min_paint_ms = 8

    app._thumb_ui_timer = QtCore.QTimer(app)
    app._thumb_ui_timer.setInterval(300)
    app._thumb_ui_timer.timeout.connect(app.thumbs.poll_thumbnails)
    app._thumb_ui_timer.start()


def build_ui(app) -> None:
    app._build_ui()


def init_status_bar(app) -> None:
    sb = QtWidgets.QStatusBar(app)
    sb.setObjectName("MainStatusBar")
    app.setStatusBar(sb)
    sb.show()

    sb = app.statusBar()
    sb.setSizeGripEnabled(False)

    app.sb_viewing = QtWidgets.QLabel("")
    app.sb_viewing.setStyleSheet("""
        QLabel {
            font-size: 11px;
            color: #222;
            padding-left: 6px;
        }
    """)

    app.sb_gl = QtWidgets.QLabel("")
    app.sb_gl.setStyleSheet("padding: 0 6px;")

    app.sb_index = QtWidgets.QLabel("")
    app.sb_anno = QtWidgets.QLabel("")
    app.sb_loop = QtWidgets.QLabel("")
    app.sb_thumb = QtWidgets.QLabel("")

    for w in (app.sb_index, app.sb_anno, app.sb_loop, app.sb_thumb):
        w.setStyleSheet("padding: 0 6px;")

    sb.addPermanentWidget(app.sb_viewing)
    sb.addPermanentWidget(app.sb_gl)
    sb.addPermanentWidget(QtWidgets.QWidget(), 1)
    sb.addPermanentWidget(app.sb_index)
    sb.addPermanentWidget(app.sb_anno)
    sb.addPermanentWidget(app.sb_loop)
    sb.addPermanentWidget(app.sb_thumb)


def init_shortcuts(app) -> None:
    QShortcut(QKeySequence("+"),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=app._on_plus)

    QShortcut(QKeySequence("="),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=app._on_plus)

    QShortcut(QKeySequence("-"),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=app._on_minus)

    QShortcut(QKeySequence("Home"),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=app.on_first)

    QShortcut(QKeySequence("End"),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=app.on_last)

    QShortcut(QKeySequence("PgUp"),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=lambda: app.on_page(-10))

    QShortcut(QKeySequence("PgDown"),
              app,
              context=QtCore.Qt.ApplicationShortcut,
              activated=lambda: app.on_page(+10))


def init_nav_menu_ribbon(app) -> None:
    app._build_nav_dock()
    app._build_menubar()
    app._install_ribbon_toolbar()


def init_interaction(app) -> None:
    app.act_annotation_mode.setChecked(True)
    app.update_cursor()
    app._nav_release_pending = False

    # Capture global key presses even when focus isn't on a plotter.
    app.installEventFilter(app)

    app.plotter.interactor.setMouseTracking(True)
    app.plotter.interactor.installEventFilter(app)

    app.plotter_ref.interactor.setMouseTracking(True)
    app.plotter_ref.interactor.installEventFilter(app)


def restore_state(app) -> None:
    try:
        st = load_state()
        ann = st.get("annotation_dir", "")
        org = st.get("original_dir", "")
        pairs = st.get("project_pairs", {})
        app.ann_dir = Path(ann) if ann else None
        app.orig_dir = None
        app.index = max(0, st.get("index", 0))
        if app.ann_dir:
            ann_key = str(app.ann_dir)
            if not isinstance(pairs, dict):
                pairs = {}
            if ann_key not in pairs and org:
                pairs[ann_key] = org
                save_state({"project_pairs": pairs})
            cand = pairs.get(ann_key, "")
            app.orig_dir = Path(cand) if cand else None
            app.directory = app.ann_dir
            app.files = app._get_sorted_files()
            if app.files and (app.index < 0 or app.index >= len(app.files)):
                app.index = 0
            if app.orig_dir is not None:
                has_full_match = all((app.orig_dir / p.name).exists() for p in app.files)
                if not has_full_match:
                    log_gui(f"restore_state: orig_dir cleared (mismatch) orig_dir={app.orig_dir}")
                    app.orig_dir = None
            app._populate_nav_list()
        else:
            app.directory, app.files = None, []
        if app.orig_dir is not None and app.ann_dir is None:
            app._expecting_ann = True
        else:
            app._expecting_ann = False
        if hasattr(app, "act_open_orig"):
            app.act_open_orig.setEnabled(not app._expecting_ann)
        if hasattr(app, "act_open_ann"):
            app.act_open_ann.setEnabled(app._expecting_ann)
        log_gui(f"restore_state: ann_dir={app.ann_dir} orig_dir={app.orig_dir} files={len(app.files)}")
    except Exception:
        pass


def finalize_startup(app) -> None:
    app.showMaximized()
    QtCore.QTimer.singleShot(0, app._restore_nav_width)

    if app.files:
        app.load_cloud()


def bootstrap(app) -> None:
    init_actions(app)
    init_state(app)
    init_timers(app)
    build_ui(app)
    init_status_bar(app)
    init_shortcuts(app)
    init_nav_menu_ribbon(app)
    init_interaction(app)
    restore_state(app)
    finalize_startup(app)
