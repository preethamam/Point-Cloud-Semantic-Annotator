#!/usr/bin/env python3
"""
Point Cloud Semantic Annotation Tool

This application allows semantic annotation of point clouds (PCD or PLY),
and remembers last folder/index between sessions.

Features:
  - Brush tool with adjustable size (press B then +/-)
  - Toggleable annotation mode (checkbox or 'A') to switch between navigation and painting
  - Undo/Redo (buttons + Ctrl/Cmd+Z, Ctrl/Cmd+Y)
  - Reset view (button + 'R')
  - Previous/Next navigation (buttons + Left/Right arrows)
  - Save annotations (Save button + Ctrl/Cmd+S)
  - Autosave annotations
  - Open folder via button
  - Maximized window with Top view or isometric initial view
  - Magenta circular cursor matching brush size in annotation mode
  - File counter and filename overlays fixed at bottom
  - State persistence (last folder & index) via JSON in script directory

Dependencies:
  pip install pyvista pyvistaqt scipy numpy PyQt5

Usage:
  python app.py
"""
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Qt5Agg")
import hashlib
import threading

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from appdirs import user_data_dir
from joblib import Parallel, delayed
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QCursor, QIcon, QKeySequence, QPainter, QPixmap, QPen
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from vtkmodules.vtkIOPLY import vtkPLYWriter
from vtkmodules.vtkRenderingCore import vtkCamera, vtkPropPicker
from PyQt5.QtWidgets import QShortcut, QToolButton

APP_NAME = "Point Cloud Annotator"
state_dir = Path(user_data_dir(APP_NAME, appauthor=False))
state_dir.mkdir(parents=True, exist_ok=True)
STATE_FILE = state_dir / "state.json"

THUMB_DIR = state_dir / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)
THUMB_SIZE = 96   # pixels (safe, fast, clean)
PERCENTAGE_CORE_FACTOR = 0.80
THUMB_N_JOBS = max(1, int((os.cpu_count() or 1) * PERCENTAGE_CORE_FACTOR))
THUMB_BACKEND = "loky"

# ─── Nav dock tuning ───────────────────────────
NAV_THUMB_SIZE   = THUMB_SIZE      # thumbnail image size
NAV_NAME_MAX     = 30      # max filename chars under thumbnail
NAV_DOCK_WIDTH   = 155     # px (adjust to taste)
RIBBON_ENH_VIEW_HEIGHT = 88


def _is_annotated_pair(ann_path: Path, orig_path: Path) -> bool:
    try:
        import numpy as np
        import pyvista as pv

        pc_a = pv.read(str(ann_path))
        pc_o = pv.read(str(orig_path))

        if pc_a.n_points != pc_o.n_points:
            return True

        if "RGB" not in pc_a.array_names or "RGB" not in pc_o.array_names:
            return False

        return not np.array_equal(pc_a["RGB"], pc_o["RGB"])
    except Exception:
        return False

def _generate_thumbnail_job(path: Path, out_png: Path, size=96):
    """
    Generate a thumbnail PNG for a point cloud.
    Runs in background worker. No Qt calls allowed here.
    """
    try:
        pc = pv.read(str(path))
        if pc.n_points == 0:
            return

        plotter = pv.Plotter(off_screen=True, window_size=(size, size))
        plotter.set_background("white")

        if "RGB" in pc.array_names:
            plotter.add_points(pc, scalars="RGB", rgb=True, point_size=4)
        else:
            plotter.add_points(pc, color="gray", point_size=4)

        # --- Top-down orthographic view (no zoom, no border) ---
        cam = plotter.camera
        cam.ParallelProjectionOn()
        cam.SetViewUp(0, 1, 0)

        xmin, xmax, ymin, ymax, zmin, zmax = pc.bounds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx, cy, zmax + (zmax - zmin + 1e-3))

        xr = xmax - xmin
        yr = ymax - ymin
        cam.SetParallelScale(0.5 * max(xr, yr))

        plotter.reset_camera_clipping_range()

        img = plotter.screenshot(transparent_background=False)
        plotter.close()

        img = img[:, :, :3].astype(np.uint8)
        Image.fromarray(img).save(out_png)

    except Exception:
        pass

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

        # Grid for compact rows
        self.controls = QtWidgets.QWidget(self)
        self.grid = QtWidgets.QGridLayout(self.controls)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(3)
        self.grid.setColumnStretch(1, 1)  # control column stretches
        # Plain bottom title (no box)
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
        """Label on left, control on right, optional trailing widget (e.g., value label)."""
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("font-size: 11px; color: #222;")
        lbl.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)

        self.grid.addWidget(lbl, self._row, 0)

        # Make common controls not waste space
        try:
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        except Exception:
            pass

        # If no trailing widget, let control span both columns (1–2)
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
                    
class Annotator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Icon and window
        icon = Path(__file__).parent / 'icons' / 'app.png'
        if not icon.exists(): icon = Path(__file__).parent / 'icons' / 'app.ico'
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.setWindowTitle("Point Cloud Annotator")
        
        # ============================================================
        # Application state actions (MUST exist before eventFilter)
        # ============================================================
        self.act_annotation_mode = QtWidgets.QAction(self)
        self.act_annotation_mode.setCheckable(True)
        self.act_annotation_mode.setChecked(True)

        self.act_toggle_annotations = QtWidgets.QAction(self)
        self.act_toggle_annotations.setCheckable(True)
        self.act_toggle_annotations.setChecked(True)

        self.act_clone = QtWidgets.QAction(self)
        self.act_clone.setCheckable(True)

        self.act_repair = QtWidgets.QAction(self)
        self.act_repair.setCheckable(True)
        
        # Loop playback state
        self.act_loop = QtWidgets.QAction(self)
        self.act_loop.setCheckable(True)
        self.act_loop.setChecked(False)        
        
        # Eraser action
        self.act_eraser = QtWidgets.QAction(self)
        self.act_eraser.setCheckable(True)
        
        # Autosave action
        self.act_autosave = QtWidgets.QAction(self)
            
        # State
        self.brush_size = 8
        self.initial_loop_timer = 1.0 # seconds
        self.point_size = 6
        self.current_color = [255,0,0]
        self.history, self.redo_stack = [], []
        self._waiting = None
        self.directory, self.files, self.index = None, [], 0
        self._last_gamma_value = 100  # default gamma = 1.0
        self.ann_dir, self.orig_dir = None, None
        self.annotations_visible = True
        self._session_edited = None  # set per cloud: np.bool_ mask
        self.annotation_alpha = 1.0  # 0..1
        self.repair_mode = False  # NEW
        self.clone_mode = False
        self._is_closing = False
        
        self._shared_camera = None           # NEW: camera shared by left/right in repair
        self._cam_observer_id = None         # NEW: observer handle for unlinking
        self._cam_syncing = False            # NEW: re-entrancy guard
        
        self._batch = False
        self._cam_pause = False      # NEW: pause camera-sync renders during loads
        self._in_zoom = False          # reentrancy guard for atomic zoom
        self._view_change_active = False
        self._need_split_fit = False   # NEW: fit both views after split

        self._fit_pad = 1.08  # camera fit padding factor
        
        # Current camera view index (matches view menu actions)
        self.current_view = 0   # 0 = Top view by default
        
        self.current_color = [255,0,0]
        self._last_paint_color = self.current_color.copy()   # NEW: restore when eraser off

        # Loop delay (seconds)
        self.loop_delay_sec = float(self.initial_loop_timer)
        
        # PATCH 7: navigation state
        self._visited = set()          # idx
        self._annotated = set()        # idx (disk-based)
        self._dirty = set()            # idx (session-based)
                
        # ——— Debounced camera fitting ———
        self._fit_delay_ms = 33  # ~2 frames at 60Hz; tweak 16–50ms if you like
        self._fit_timer = QtCore.QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_to_canvas)
        
        # ——— Fast painting path ———
        self._paint_step_frac = 0.33      # denser than 0.8 to avoid gaps while staying fast
        self._brush_coverage  = 1.25    # 15% inflation to guarantee full coverage
        self._last_paint_xy   = None
        self._in_stroke       = False
        
        # PATCH 5: thumbnail backend (NO UI)
        self._thumb_lock = threading.Lock()
        self._thumb_job_set = set()        # (src_path, out_png)
        self._thumb_out_by_idx = {}        # idx -> out_png
        self._thumb_worker_running = False
        self._thumb_worker_start_pending = False

        self._stroke_render_timer = QtCore.QTimer(self)
        self._stroke_render_timer.setSingleShot(True)
        self._stroke_render_timer.timeout.connect(self._render_views_once)
        
        # ——— Loop playback (Feature 2) ———
        self._loop_timer = QtCore.QTimer(self)
        self._loop_timer.setSingleShot(False)
        self._loop_timer.timeout.connect(self._on_loop_tick)
        
        # ——— Paint throttling (limit to ~120 FPS) ———
        self._paint_timer = QtCore.QElapsedTimer()
        self._paint_timer.start()
        self._min_paint_ms = 8   # ~8 ms between paint batches (~120 Hz)    
        
        # PATCH 6: thumbnail UI refresher
        self._thumb_ui_timer = QtCore.QTimer(self)
        self._thumb_ui_timer.setInterval(300)  # ms
        self._thumb_ui_timer.timeout.connect(self._poll_thumbnails)
        self._thumb_ui_timer.start()
        
        self._constrain_line = False   # Shift-drag = straight line
        self._anchor_xy = None         # (x,y) where the line started
        self._line_len_px = 0.0        # distance already painted along the line (pixels)

        # Build UI
        self._build_ui()        
        
        sb = QtWidgets.QStatusBar(self)
        sb.setObjectName("MainStatusBar")
        self.setStatusBar(sb)
        sb.show()
        
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)

        # LEFT: Viewing filename
        self.sb_viewing = QtWidgets.QLabel("")
        self.sb_viewing.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #222;
                padding-left: 6px;
            }
        """)

        # RIGHT: existing indicators
        self.sb_index = QtWidgets.QLabel("")
        self.sb_anno  = QtWidgets.QLabel("")
        self.sb_loop  = QtWidgets.QLabel("")
        self.sb_thumb = QtWidgets.QLabel("")

        # subtle spacing
        for w in (self.sb_index, self.sb_anno, self.sb_loop, self.sb_thumb):
            w.setStyleSheet("padding: 0 6px;")
            
        sb.addPermanentWidget(self.sb_viewing)      # ⬅ LEFT
        sb.addPermanentWidget(QtWidgets.QWidget(), 1)  # stretch spacer
        sb.addPermanentWidget(self.sb_index)
        sb.addPermanentWidget(self.sb_anno)
        sb.addPermanentWidget(self.sb_loop)
        sb.addPermanentWidget(self.sb_thumb)
        
        QShortcut(QKeySequence("+"),
          self,
          context=QtCore.Qt.ApplicationShortcut,
          activated=self._on_plus)
        
        QShortcut(QKeySequence("="),
          self,
          context=QtCore.Qt.ApplicationShortcut,
          activated=self._on_plus)

        QShortcut(QKeySequence("-"),
                self,
                context=QtCore.Qt.ApplicationShortcut,
                activated=self._on_minus)
        
        QShortcut(QKeySequence("Home"),
                  self,
                  context=QtCore.Qt.ApplicationShortcut,
                  activated=self.on_first)

        QShortcut(QKeySequence("End"),
                  self,
                  context=QtCore.Qt.ApplicationShortcut,
                  activated=self.on_last)

        QShortcut(QKeySequence("PgUp"),
                  self,
                  context=QtCore.Qt.ApplicationShortcut,
                  activated=lambda: self.on_page(-10))

        QShortcut(QKeySequence("PgDown"),
                  self,
                  context=QtCore.Qt.ApplicationShortcut,
                  activated=lambda: self.on_page(+10))


        # PATCH 2B: Left navigation dock
        self._build_nav_dock()
        
        # NEW: Menu bar (Patch 1)
        self._build_menubar()
        
        self._install_ribbon_toolbar()
        
        # --- ensure Annotation Mode ON at startup ---
        self.act_annotation_mode.setChecked(True)
        self.update_cursor()
        self._nav_release_pending = False
            
        # allow us to catch mouse‐moves on the VTK widget
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.installEventFilter(self)
        
        # ⬇️ Make zoom-at-cursor work on the LEFT (original) view too
        self.plotter_ref.interactor.setMouseTracking(True)
        self.plotter_ref.interactor.installEventFilter(self)

        self._stroke_active           = False
        self._stroke_idxs             = set()
        self._colors_before_stroke    = None        

        # Restore state
        if STATE_FILE.exists():
            try:
                st = json.loads(STATE_FILE.read_text())
                ann = st.get('annotation_dir', '')
                org = st.get('original_dir', '')
                self.ann_dir = Path(ann) if ann else None
                self.orig_dir = Path(org) if org else None
                self.index = max(0, st.get('index', 0))
                # list files from the annotation folder only
                if self.ann_dir:
                    self.directory = self.ann_dir  # keep old name if you use it elsewhere
                    self.files = self._get_sorted_files()
                    self._populate_nav_list()
                else:
                    self.directory, self.files = None, []
            except:
                pass
        # Show and load
        self.showMaximized()

        # restore nav dock width AFTER show
        QtCore.QTimer.singleShot(0, self._restore_nav_width)

        if self.files:
            self.load_cloud()

    def _install_ribbon_toolbar(self):
        """Put the ribbon in QMainWindow's toolbar area so docks sit below it."""
        self.ribbon = self._build_ribbon()

        self.ribbon_tb = QtWidgets.QToolBar("Ribbon", self)
        self.ribbon_tb.setMovable(False)
        self.ribbon_tb.setFloatable(False)
        self.ribbon_tb.setAllowedAreas(QtCore.Qt.TopToolBarArea)
        self.ribbon_tb.setContentsMargins(0, 0, 0, 0)

        # Prevent the toolbar from forcing a tiny height
        self.ribbon_tb.setIconSize(QtCore.QSize(16, 16))
        self.ribbon_tb.addWidget(self.ribbon)

        self.addToolBar(QtCore.Qt.TopToolBarArea, self.ribbon_tb)

    @staticmethod
    def _natural_key(path):
        """Split filename into text/number chunks for natural sorting."""
        return [int(tok) if tok.isdigit() else tok.lower()
                for tok in re.split(r'(\d+)', path.name)]
    
    def _get_sorted_files(self):
        all_files = list(self.directory.glob('*.ply')) \
                + list(self.directory.glob('*.pcd'))
        return sorted(all_files, key=self._natural_key)

    def _compute_brush_idx(self, x, y):
        """
        Exact WYSIWYG: points are rendered as round sprites (radius s_px).
        Paint a point only if its sprite fits fully inside the brush circle:
            ||center - cursor|| <= r_px - s_px
        Fallback to circle–circle intersection when r_px <= s_px (tiny brushes).
        """
        if not hasattr(self, 'kdtree') or self.kdtree is None or not hasattr(self, 'actor'):
            return []

        ren   = self.plotter.renderer
        inter = self.plotter.interactor
        H     = inter.height()

        from vtkmodules.vtkRenderingCore import vtkPropPicker
        picker = vtkPropPicker()
        if not picker.Pick(x, H - y, 0, ren):
            return []
        wc = np.array(picker.GetPickPosition(), dtype=float)
        if not np.isfinite(wc).all():
            return []

        # Stable display depth at brush center
        ren.SetWorldPoint(wc[0], wc[1], wc[2], 1.0); ren.WorldToDisplay()
        xd, yd, zd = ren.GetDisplayPoint()

        # World size of one display pixel at this depth
        ren.SetDisplayPoint(xd + 1.0, yd, zd); ren.DisplayToWorld()
        wx1, wy1, wz1, _ = ren.GetWorldPoint()
        ren.SetDisplayPoint(xd, yd + 1.0, zd); ren.DisplayToWorld()
        wx2, wy2, wz2, _ = ren.GetWorldPoint()
        px_world = max(
            float(np.linalg.norm(np.array([wx1, wy1, wz1]) - wc)),
            float(np.linalg.norm(np.array([wx2, wy2, wz2]) - wc)),
        )

        # Brush & sprite geometry (display pixels)
        r_px = float(max(1, self.brush_size))          # brush radius (px)
        s_px = 0.5 * float(max(1, self.point_size))    # round sprite radius (px)

        # Loose world preselect (brush + sprite), slight inflation for safety
        inflate = float(getattr(self, "_brush_coverage", 1.15))
        world_r = max(1e-9, (r_px + s_px) * px_world * inflate)
        cand    = self.kdtree.query_ball_point(wc, world_r)
        if not cand:
            return []

        # Screen-space refine
        cx, cy = float(x), float(H - y)
        keep   = []
        SetWorldPoint   = ren.SetWorldPoint
        WorldToDisplay  = ren.WorldToDisplay
        GetDisplayPoint = ren.GetDisplayPoint
        pts = self.cloud.points

        r_in = r_px - s_px
        if r_in > 0.5:
            r2_in = r_in * r_in
            for i in cand:
                wx, wy, wz = pts[i]
                SetWorldPoint(wx, wy, wz, 1.0); WorldToDisplay()
                dx, dy, _ = GetDisplayPoint()
                if (dx - cx)*(dx - cx) + (dy - cy)*(dy - cy) <= r2_in:
                    keep.append(i)
        else:
            # Tiny brushes: allow circle–circle INTERSECTION so you can still paint
            r2_sum = (r_px + s_px) * (r_px + s_px)
            for i in cand:
                wx, wy, wz = pts[i]
                SetWorldPoint(wx, wy, wz, 1.0); WorldToDisplay()
                dx, dy, _ = GetDisplayPoint()
                if (dx - cx)*(dx - cx) + (dy - cy)*(dy - cy) <= r2_sum:
                    keep.append(i)

        return keep

    def _snap_camera(self, plotter):
        cam = plotter.camera
        try:
            return dict(
                pos=np.array(cam.GetPosition(), dtype=float),
                fp=np.array(cam.GetFocalPoint(), dtype=float),
                vu=np.array(cam.GetViewUp(), dtype=float),
                pp=bool(cam.GetParallelProjection()),
                ps=float(cam.GetParallelScale()),
                va=float(cam.GetViewAngle()),
            )
        except Exception:
            return None

    def _restore_camera(self, plotter, snap):
        if not snap:
            return
        cam = plotter.camera
        cam.SetPosition(*snap['pos'])
        cam.SetFocalPoint(*snap['fp'])
        cam.SetViewUp(*snap['vu'])
        if snap['pp']:
            cam.ParallelProjectionOn()
            cam.SetParallelScale(snap['ps'])
        else:
            cam.ParallelProjectionOff()
            cam.SetViewAngle(snap['va'])

    def _build_menubar(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        self.act_open_orig = QtWidgets.QAction("Open Original Folder…", self)
        self.act_open_orig.triggered.connect(self.open_orig_folder)
        file_menu.addAction(self.act_open_orig)

        self.act_open_ann = QtWidgets.QAction("Open Annotation Folder…", self)
        self.act_open_ann.triggered.connect(self.open_ann_folder)
        file_menu.addAction(self.act_open_ann)

        file_menu.addSeparator()

        self.act_save = QtWidgets.QAction("Save", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.on_save)
        file_menu.addAction(self.act_save)

        self.act_autosave.setText("Autosave")
        self.act_autosave.setCheckable(True)
        self.act_autosave.setChecked(True)
        file_menu.addAction(self.act_autosave)

        file_menu.addSeparator()

        self.act_clear_thumbs = QtWidgets.QAction("Clear Thumbnail Cache", self)
        self.act_clear_thumbs.triggered.connect(self._clear_thumbnail_cache)
        file_menu.addAction(self.act_clear_thumbs)

        file_menu.addSeparator()

        self.act_exit = QtWidgets.QAction("Exit", self)
        self.act_exit.triggered.connect(self.close)
        file_menu.addAction(self.act_exit)

        # Edit
        edit_menu = menubar.addMenu("&Edit")

        self.act_undo = QtWidgets.QAction("Undo", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(self.on_undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QtWidgets.QAction("Redo", self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.triggered.connect(self.on_redo)
        edit_menu.addAction(self.act_redo)

        edit_menu.addSeparator()

        # Eraser (already self.act_eraser)
        self.act_eraser.setText("Eraser")
        self.act_eraser.setShortcut(QKeySequence("E"))
        self.act_eraser.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        self.act_eraser.toggled.connect(self._on_eraser_toggled)
        edit_menu.addAction(self.act_eraser)

        # Repair/Clone (already self.act_repair / self.act_clone)
        self.act_repair.setText("Repair Mode")
        self.act_repair.setShortcut(QKeySequence("Shift+R"))
        self.act_repair.toggled.connect(lambda on: on and self.act_clone.setChecked(False))
        self.act_repair.toggled.connect(self.toggle_repair_mode)
        edit_menu.addAction(self.act_repair)

        self.act_clone.setText("Clone Mode")
        self.act_clone.setShortcut(QKeySequence("C"))
        self.act_clone.toggled.connect(lambda on: on and self.act_repair.setChecked(False))
        self.act_clone.toggled.connect(self.toggle_clone_mode)
        edit_menu.addAction(self.act_clone)

        edit_menu.addSeparator()

        # Color submenu
        color_menu = edit_menu.addMenu("Color")

        self.act_pick_color = QtWidgets.QAction("Pick Color…", self)
        self.act_pick_color.triggered.connect(self.pick_color)
        color_menu.addAction(self.act_pick_color)

        color_menu.addSeparator()

        self._SWATCHES = [
            ("Red", "#FF0000"), ("Green", "#00FF00"), ("Blue", "#0000FF"),
            ("Yellow", "#FFFF00"), ("Cyan", "#00FFFF"), ("Magenta", "#FF00FF"),
            ("Orange", "#FFA500"), ("Pink", "#FFC0CB"), ("Purple", "#800080"),
            ("Brown", "#A52A2A"), ("Maroon", "#800000"), ("Olive", "#808000"),
            ("Teal", "#008080"), ("Navy", "#000080"), ("Gray", "#808080"),
            ("Light Gray", "#D3D3D3"), ("Black", "#000000"), ("White", "#FFFFFF"),
        ]
        for name, hexcol in self._SWATCHES:
            act = QtWidgets.QAction(name, self)
            pix = QPixmap(12, 12)
            pix.fill(QColor(hexcol))
            act.setIcon(QIcon(pix))
            act.triggered.connect(lambda _, c=hexcol: self.select_swatch(c, None))
            color_menu.addAction(act)

        # View
        view_menu = menubar.addMenu("&View")

        # View presets (unchanged)
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
            act = QtWidgets.QAction(name, self)
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(QtCore.Qt.ApplicationShortcut)
            act.triggered.connect(lambda _, i=idx: self._set_view(i))
            view_menu.addAction(act)

        view_menu.addSeparator()

        self.act_zoom_in = QtWidgets.QAction("Zoom In", self)
        self.act_zoom_in.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        self.act_zoom_in.triggered.connect(self.on_zoom_in)
        view_menu.addAction(self.act_zoom_in)

        self.act_zoom_out = QtWidgets.QAction("Zoom Out", self)
        self.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.act_zoom_out.triggered.connect(self.on_zoom_out)
        view_menu.addAction(self.act_zoom_out)

        self.act_reset_view = QtWidgets.QAction("Reset View", self)
        self.act_reset_view.setShortcut("R")
        self.act_reset_view.triggered.connect(self.reset_view)
        view_menu.addAction(self.act_reset_view)

        view_menu.addSeparator()

        self.act_annotation_mode.setText("Annotation Mode")
        self.act_annotation_mode.setShortcut(QKeySequence("Ctrl+A"))
        self.act_annotation_mode.toggled.connect(self.toggle_annotation)
        view_menu.addAction(self.act_annotation_mode)

        self.act_toggle_annotations.setText("Show Annotations")
        self.act_toggle_annotations.setShortcut(QKeySequence("Shift+A"))
        self.act_toggle_annotations.toggled.connect(self.set_annotations_visible)
        view_menu.addAction(self.act_toggle_annotations)

        view_menu.addSeparator()

        act_toggle_nav = QtWidgets.QAction("Toggle Navigation Pane", self, checkable=True)
        act_toggle_nav.setChecked(True)
        act_toggle_nav.setShortcut(QKeySequence("N"))
        act_toggle_nav.toggled.connect(self.nav_dock.setVisible)
        self.nav_dock.visibilityChanged.connect(act_toggle_nav.setChecked)
        view_menu.addAction(act_toggle_nav)

        # Playback
        playback_menu = menubar.addMenu("&Playback")

        self.act_loop.setText("Loop")
        self.act_loop.setShortcut(QKeySequence("L"))
        self.act_loop.toggled.connect(self._toggle_loop)
        playback_menu.addAction(self.act_loop)

        playback_menu.addSeparator()

        self.act_prev = QtWidgets.QAction("Previous", self)
        self.act_prev.setShortcut(QKeySequence(QtCore.Qt.Key_Left))
        self.act_prev.triggered.connect(self.on_prev)
        playback_menu.addAction(self.act_prev)

        self.act_next = QtWidgets.QAction("Next", self)
        self.act_next.setShortcut(QKeySequence(QtCore.Qt.Key_Right))
        self.act_next.triggered.connect(self.on_next)
        playback_menu.addAction(self.act_next)

        # Tools
        tools_menu = menubar.addMenu("&Tools")

        self.act_auto_contrast = QtWidgets.QAction("Auto Contrast", self)
        self.act_auto_contrast.triggered.connect(self.apply_auto_contrast)
        tools_menu.addAction(self.act_auto_contrast)

        self.act_reset_contrast = QtWidgets.QAction("Reset Contrast", self)
        self.act_reset_contrast.triggered.connect(self.reset_contrast)
        tools_menu.addAction(self.act_reset_contrast)

        tools_menu.addSeparator()

        self.act_hist = QtWidgets.QAction("Show RGB Histograms", self)
        self.act_hist.triggered.connect(self.show_histograms)
        tools_menu.addAction(self.act_hist)

        # Help
        help_menu = menubar.addMenu("&Help")

        act_about = QtWidgets.QAction("About", self)
        act_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act_about)
        
    
    def _make_icon(self, size, draw_fn):
        pix = QPixmap(size, size)
        pix.fill(QtCore.Qt.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        draw_fn(painter, size)
        painter.end()
        return QIcon(pix)

    def _icon_from_file(self, filename):
        icon_path = Path(__file__).parent / "icons" / filename
        if not icon_path.exists():
            return None
        icon = QIcon(str(icon_path))
        if icon.isNull():
            return None
        return icon

    def _icon_pencil(self):
        icon = self._icon_from_file("annotate.png")
        if icon is not None:
            return icon

        def draw(p, s):
            p.setPen(QPen(QColor("#2b2b2b"), 2, QtCore.Qt.SolidLine,
                          QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            p.drawLine(3, s - 3, s - 3, 3)
            p.setPen(QPen(QColor("#d28b36"), 2, QtCore.Qt.SolidLine,
                          QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            p.drawLine(5, s - 5, s - 5, 5)
        return self._make_icon(16, draw)

    def _icon_eraser(self):
        icon = self._icon_from_file("eraser.png")
        if icon is not None:
            return icon

        def draw(p, s):
            p.setPen(QPen(QColor("#2b2b2b"), 1.5))
            p.setBrush(QColor("#f2b07b"))
            p.drawRoundedRect(3, 6, 10, 6, 2, 2)
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.drawLine(4, 7, 12, 11)
        return self._make_icon(16, draw)

    def _icon_repair(self):
        icon = self._icon_from_file("repair.png")
        if icon is not None:
            return icon

        def draw(p, s):
            p.setPen(QPen(QColor("#2b2b2b"), 1.8, QtCore.Qt.SolidLine,
                          QtCore.Qt.RoundCap))
            p.drawEllipse(2, 2, 6, 6)
            p.drawLine(7, 7, 13, 13)
            p.drawLine(10, 12, 13, 9)
        return self._make_icon(16, draw)

    def _icon_clone(self):
        icon = self._icon_from_file("clone.png")
        if icon is not None:
            return icon

        def draw(p, s):
            p.setPen(QPen(QColor("#2b2b2b"), 1.5))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(3, 5, 8, 8)
            p.drawRect(6, 2, 8, 8)
        return self._make_icon(16, draw)

    def _icon_palette(self):
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
        return self._make_icon(16, draw)

    def _icon_contrast(self):
        icon = self._icon_from_file("contrast.png")
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
        return self._make_icon(16, draw)

    def _icon_reset_view(self):
        icon = self._icon_from_file("reset.png")
        if icon is not None:
            return icon
        return self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)

    def _icon_hist(self):
        icon = self._icon_from_file("histogram.png")
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
        return self._make_icon(16, draw)

    def _icon_prev(self):
        icon = self._icon_from_file("previous.png")
        if icon is not None:
            return icon
        return self.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack)

    def _icon_next(self):
        icon = self._icon_from_file("next.png")
        if icon is not None:
            return icon
        return self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward)

    def _icon_loop(self):
        icon = self._icon_from_file("loop.png")
        if icon is not None:
            return icon
        return self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)

    def _icon_reset_contrast(self):
        icon = self._icon_from_file("reset-contrast.png")
        if icon is not None:
            return icon
        return self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)

    def _icon_eye(self):
        icon = self._icon_from_file("view.png")
        if icon is not None:
            return icon

        def draw(p, s):
            p.setPen(QPen(QColor("#2b2b2b"), 1.2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(2, 5, 12, 6)
            p.setBrush(QColor("#2b2b2b"))
            p.drawEllipse(7, 7, 2, 2)
        return self._make_icon(16, draw)

    def _icon_zoom(self, plus=True):
        if plus:
            icon = self._icon_from_file("zoom-in.png")
        else:
            icon = self._icon_from_file("zoom-out.png")
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
        return self._make_icon(16, draw)

    def _ribbon_button(self, icon, tooltip, checkable=False, icon_size=14, button_size=22):
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

    def _build_ribbon(self):
        ribbon = QtWidgets.QWidget(self)
        ribbon.setFixedHeight(130)
        ribbon.setStyleSheet("background:#efefef;")

        h = QtWidgets.QHBoxLayout(ribbon)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)
        h.setAlignment(QtCore.Qt.AlignTop)

        # ───────────────── Navigation ─────────────────
        nav = RibbonGroup("Navigation", 130, title_position="top")

        btn_prev = self._ribbon_button(
            self._icon_prev(),
            "Previous (Left Arrow)"
        )
        btn_prev.clicked.connect(self.on_prev)

        btn_next = self._ribbon_button(
            self._icon_next(),
            "Next (Right Arrow)"
        )
        btn_next.clicked.connect(self.on_next)

        chk_loop = self._ribbon_button(
            self._icon_loop(),
            "Loop playback",
            checkable=True
        )
        chk_loop.setChecked(self.act_loop.isChecked())
        chk_loop.toggled.connect(self.act_loop.setChecked)
        self.act_loop.toggled.connect(chk_loop.setChecked)

        # --- Delay spinbox (existing behavior, unchanged) ---
        delay = QtWidgets.QDoubleSpinBox()
        delay.setRange(0.1, 60.0)
        delay.setSingleStep(0.1)
        delay.setValue(self.loop_delay_sec)
        delay.setDecimals(2)
        delay.setFixedWidth(64)
        delay.valueChanged.connect(self._set_loop_delay)

        # --- Delay presets dropdown (NEW, minimal) ---
        btn_delay_menu = self._ribbon_button(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView),
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
            delay.setValue(float(val)) 

        for v in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
            act = QtWidgets.QAction(f"{v:.1f} s", self)
            act.triggered.connect(lambda _, x=v: _set_delay(x))
            delay_menu.addAction(act)

        delay_menu.addSeparator()

        def _custom_delay():
            val, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Loop Delay",
                "Seconds:",
                self.loop_delay_sec,
                0.1,
                60.0,
                1,
            )
            if ok:
                _set_delay(val)

        act_custom = QtWidgets.QAction("Custom…", self)
        act_custom.triggered.connect(_custom_delay)
        delay_menu.addAction(act_custom)

        btn_delay_menu.setMenu(delay_menu)

        # --- Combine spinbox + dropdown into one row widget ---
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

        nav.add(nav_row)
        nav.add_row("Delay", delay_row)

        # ───────────────── Annotation ─────────────────
        ann = RibbonGroup("Annotation", 200)

        # --- Alpha ---
        s_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s_alpha.setRange(0, 100)
        s_alpha.setValue(int(self.annotation_alpha * 100))
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

        lbl_alpha = QtWidgets.QLabel(f"{int(self.annotation_alpha * 100)}%")
        lbl_alpha.setFixedWidth(36)
        lbl_alpha.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        s_alpha.valueChanged.connect(self._on_ribbon_alpha)
        ann.add_row("Alpha (A +/-)", s_alpha, lbl_alpha)

        # --- Brush ---
        s_brush = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s_brush.setRange(1, 200)
        s_brush.setValue(int(self.brush_size))
        s_brush.setStyleSheet(s_alpha.styleSheet())

        lbl_brush = QtWidgets.QLabel(f"{int(self.brush_size)} px")
        lbl_brush.setFixedWidth(36)
        lbl_brush.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        s_brush.valueChanged.connect(self._on_ribbon_brush)
        ann.add_row("Brush (B +/-)", s_brush, lbl_brush)

        # --- Point ---
        s_point = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s_point.setRange(1, 20)
        s_point.setValue(self.point_size)
        s_point.setStyleSheet(s_alpha.styleSheet())

        lbl_point = QtWidgets.QLabel(f"{self.point_size} px")
        lbl_point.setFixedWidth(36)
        lbl_point.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        s_point.valueChanged.connect(self._on_ribbon_point)
        ann.add_row("Point (D +/-)", s_point, lbl_point)
        
        self.ribbon_sliders = {
            "alpha": (s_alpha, lbl_alpha),
            "brush": (s_brush, lbl_brush),
            "point": (s_point, lbl_point),
        }

        # ───────────────── Color ─────────────────
        col = RibbonGroup("Color", 170)

        btn_pick = QtWidgets.QPushButton("Pick Color…")
        btn_pick.clicked.connect(self.pick_color)
        col.add(btn_pick)

        swatches = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(swatches)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(3)
        g.setVerticalSpacing(3)

        # More swatches (fills space, no waste)      
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
            "#00FFFF", "#FF00FF", "#FFA500", "#800080",
            "#A52A2A", "#808080", "#000000", "#FFFFFF",
            "#008080", "#000080", "#808000", "#FFC0CB",

            "#C0C0C0",  # Silver
            "#FFD700",  # Gold
            "#4B0082",  # Indigo
            "#2E8B57",  # SeaGreen
            "#DC143C",  # Crimson
            "#4682B4",  # SteelBlue
            "#9ACD32",  # YellowGreen
            "#8B4513",  # SaddleBrown
        ]
        
        # 6 columns looks dense & tidy in this width
        cols = 6
        swatch_group = QtWidgets.QButtonGroup(col)
        swatch_group.setExclusive(True)
        for i, c in enumerate(colors):
            b = QtWidgets.QPushButton()
            b.setCheckable(True)                         # ← KEY
            b.setAutoExclusive(False)                    # handled by group
            b.setFixedSize(15, 15)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: {c};
                    border: 1px solid #777;
                    border-radius: 2px;
                }}
                QPushButton:checked {{
                    border: 3px solid #00E5FF;           /* #39FF14: true neon green */
                    padding: -1px;                      /* keeps size stable */
                    background: {c};
                }}
            """)

            swatch_group.addButton(b)                    # ← KEY
            b.clicked.connect(lambda _, x=c: self.select_swatch(x))
            g.addWidget(b, i // cols, i % cols)

        col.add(swatches)
        color_height = col.sizeHint().height()
        if color_height > 0:
            ann.setFixedHeight(color_height)

        # ───────────────── Edit ─────────────────
        edit = RibbonGroup("Edit", 130)

        chk_ann = self._ribbon_button(
            self._icon_pencil(),
            "Annotation mode (A)",
            checkable=True
        )
        chk_ann.setChecked(self.act_annotation_mode.isChecked())
        chk_ann.toggled.connect(self.act_annotation_mode.setChecked)
        self.act_annotation_mode.toggled.connect(chk_ann.setChecked)

        chk_eraser = self._ribbon_button(
            self._icon_eraser(),
            "Eraser",
            checkable=True
        )
        chk_eraser.setChecked(self.act_eraser.isChecked())
        chk_eraser.toggled.connect(self.act_eraser.setChecked)
        self.act_eraser.toggled.connect(chk_eraser.setChecked)

        chk_repair = self._ribbon_button(
            self._icon_repair(),
            "Repair",
            checkable=True
        )
        chk_repair.setChecked(self.act_repair.isChecked())
        chk_repair.toggled.connect(self.act_repair.setChecked)
        self.act_repair.toggled.connect(chk_repair.setChecked)

        chk_clone = self._ribbon_button(
            self._icon_clone(),
            "Clone",
            checkable=True
        )
        chk_clone.setChecked(self.act_clone.isChecked())
        chk_clone.toggled.connect(self.act_clone.setChecked)
        self.act_clone.toggled.connect(chk_clone.setChecked)

        edit_row = QtWidgets.QWidget()
        edit_row_layout = QtWidgets.QHBoxLayout(edit_row)
        edit_row_layout.setContentsMargins(0, 0, 0, 0)
        edit_row_layout.setSpacing(2)
        edit_row_layout.addWidget(chk_ann)
        edit_row_layout.addWidget(chk_eraser)
        edit_row_layout.addWidget(chk_repair)
        edit_row_layout.addWidget(chk_clone)
        edit.add(edit_row)

        # ───────────────── Enhancement ─────────────────
        enh = RibbonGroup("Enhancement", 195, title_position="bottom")
        enh.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        enh.controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        enh.grid.setAlignment(QtCore.Qt.AlignTop)
        if color_height > 0:
            enh.setFixedHeight(color_height)

        s_gamma = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s_gamma.setRange(10, 300)
        s_gamma.setValue(100)
        s_gamma.valueChanged.connect(self.on_gamma_change)
        s_gamma.setStyleSheet(s_alpha.styleSheet())

        self.ribbon_gamma_slider = s_gamma
        self.ribbon_gamma_label = QtWidgets.QLabel("1.00")
        self.ribbon_gamma_label.setStyleSheet("font-size: 11px; color: #222;")
        self.ribbon_gamma_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        btn_auto = self._ribbon_button(
            self._icon_contrast(),
            "Auto contrast",
            icon_size=16,
            button_size=24
        )
        btn_auto.clicked.connect(self.apply_auto_contrast)

        btn_reset = self._ribbon_button(
            self._icon_reset_contrast(),
            "Reset contrast",
            icon_size=16,
            button_size=24
        )
        btn_reset.clicked.connect(self.reset_contrast)

        btn_hist = self._ribbon_button(
            self._icon_hist(),
            "Show histograms",
            icon_size=16,
            button_size=24
        )
        btn_hist.clicked.connect(self.show_histograms)

        enh.add_row("Gamma (G +/-)", s_gamma, self.ribbon_gamma_label)
        self.ribbon_sliders["gamma"] = (s_gamma, self.ribbon_gamma_label)
        enh_row = QtWidgets.QWidget()
        enh_row_layout = QtWidgets.QHBoxLayout(enh_row)
        enh_row_layout.setContentsMargins(0, 0, 0, 0)
        enh_row_layout.setSpacing(4)
        enh_row_layout.addWidget(btn_auto)
        enh_row_layout.addWidget(btn_reset)
        enh_row_layout.addWidget(btn_hist)
        enh.add(enh_row)
        
        # ───────────────── View ─────────────────
        view = RibbonGroup("View", 195, title_position="bottom")
        view.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        view.controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        view.grid.setAlignment(QtCore.Qt.AlignTop)
        if color_height > 0:
            view.setFixedHeight(color_height)

        btn_reset = self._ribbon_button(
            self._icon_reset_view(),
            "Reset view (R)"
        )
        btn_reset.clicked.connect(self.reset_view)

        btn_zoom_in = self._ribbon_button(
            self._icon_zoom(True),
            "Zoom in"
        )
        btn_zoom_in.clicked.connect(self.on_zoom_in)

        btn_zoom_out = self._ribbon_button(
            self._icon_zoom(False),
            "Zoom out"
        )
        btn_zoom_out.clicked.connect(self.on_zoom_out)

        # --- View preset dropdown ---
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

        cmb_view.setCurrentIndex(self.current_view)
        self.ribbon_view_combo = cmb_view
        def _on_view_changed(i):
            self._set_view(cmb_view.itemData(i))
            self._release_view_combo_focus()

        cmb_view.currentIndexChanged.connect(_on_view_changed)

        # --- Show Annotations ---
        chk_toggle = self._ribbon_button(
            self._icon_eye(),
            "Show annotations",
            checkable=True
        )
        chk_toggle.setChecked(self.act_toggle_annotations.isChecked())
        chk_toggle.toggled.connect(self.act_toggle_annotations.setChecked)

        # action → ribbon sync
        self.act_toggle_annotations.toggled.connect(chk_toggle.setChecked)

        # --- Assemble ---
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

        # ───────────────── Assemble ─────────────────
        nav_edit = QtWidgets.QWidget()
        nav_edit_layout = QtWidgets.QVBoxLayout(nav_edit)
        nav_edit_layout.setContentsMargins(0, 0, 0, 0)
        nav_edit_layout.setSpacing(4)
        nav_edit_layout.addWidget(nav)
        nav_edit_layout.addWidget(edit)

        h.addWidget(nav_edit, 0, QtCore.Qt.AlignTop)
        for grp in (ann, col, enh, view):
            h.addWidget(grp, 0, QtCore.Qt.AlignTop)

        h.addStretch(1)
        return ribbon

    def _release_view_combo_focus(self):
        """Drop combo focus so the blue highlight doesn't linger."""
        def _focus():
            try:
                if hasattr(self, "plotter"):
                    self.plotter.interactor.setFocus()
                else:
                    self.setFocus()
            except Exception:
                self.setFocus()
        QtCore.QTimer.singleShot(0, _focus)

    def _build_nav_dock(self):
        """Patch 2B: Left navigation dock (empty shell)."""
        # Dock widget
        self.nav_dock = QtWidgets.QDockWidget("Navigation", self)
        # PATCH 6.1: fixed nav dock width to start width
        self.nav_dock.setMinimumWidth(110)
        self.nav_dock.setMaximumWidth(400)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.nav_dock)
        
        self.nav_dock.setObjectName("NavigationDock")
        self.nav_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        # PATCH 2B.2: remove close / float buttons from nav dock
        self.nav_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.nav_dock.setTitleBarWidget(QtWidgets.QWidget())

        # ---- Navigation dock content ----
        container = QtWidgets.QWidget(self.nav_dock)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        # Search / Go-to box
        self.nav_search = QtWidgets.QLineEdit()
        self.nav_search.setPlaceholderText("Go to index or search filename…")
        self.nav_search.returnPressed.connect(self._on_nav_search_entered)
        self.nav_search.installEventFilter(self)

        layout.addWidget(self.nav_search)

        # Feedback / status label
        self.nav_status = QtWidgets.QLabel("")
        self.nav_status.setStyleSheet("color: gray; font-size: 10px; padding: 0px;")
        self.nav_status.setMaximumHeight(4)
        layout.addWidget(self.nav_status)

        # PATCH 4: Navigation list (filenames only)
        self.nav_list = QtWidgets.QListWidget()
        self.nav_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.nav_list.setUniformItemSizes(True)  # faster scrolling
        self.nav_list.setAlternatingRowColors(True)
        self.nav_list.currentRowChanged.connect(self._on_nav_row_changed)

        # (optional) Enter/Return activates too
        self.nav_list.setSpacing(4)
        self.nav_list.setStyleSheet("""
        QListWidget::item {
            padding: 4px;
        }
        """)

        layout.addWidget(self.nav_list, 1)

        layout.addStretch(0)


        self.nav_dock.setWidget(container)

        # Add dock to main window (left side)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.nav_dock)

        # Start visible (can change later if you want default hidden)
        self.nav_dock.setVisible(True)
    
    
    def _build_ui(self):
        """
        Build ONLY the visual layout:
        - Two 3D viewports
        - Text overlays

        ALL interaction logic is handled via:
        - QAction (menu + shortcuts)
        - Navigation dock
        """

        w = QtWidgets.QWidget(self)
        self.setCentralWidget(w)

        # Central area is ONLY the content row; ribbon is installed as a toolbar now.
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ─────────────────────────────────────────────
        # 3D Viewports
        # ─────────────────────────────────────────────
        
        # LEFT: Original / Reference
        self.plotter_ref = QtInteractor(self)
        self.plotter_ref.set_background("white")
        self.plotter_ref.setVisible(False)   # 🔑 start hidden
        lay.addWidget(self.plotter_ref.interactor, stretch=4)
        
        # 🔹 Vertical divider
        self.vline = QtWidgets.QFrame()
        self.vline.setFrameShape(QtWidgets.QFrame.VLine)
        self.vline.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.vline.setFixedWidth(8)
        self.vline.setStyleSheet("color: #cfcfcf;")        
        self.vline.setVisible(False)   # shown only in Repair/Clone
        lay.addWidget(self.vline)

        # RIGHT: Annotated
        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")
        lay.addWidget(self.plotter.interactor, stretch=4)

        # Anti-aliasing (safe-guarded)
        try:
            self.plotter.ren_win.SetMultiSamples(8)
            self.plotter_ref.ren_win.SetMultiSamples(8)
        except Exception:
            pass

        # Left pane (Original PC) title overlay
        self.left_title = QtWidgets.QLabel(self.plotter_ref.interactor)
        self.left_title.setAutoFillBackground(True)
        self.left_title.setStyleSheet(
            "color:black; font-weight:bold; "
            "background-color:white; font-size:14px;"
        )
        self.left_title.setText("Original Point Cloud")
        self.left_title.hide()

        # ─────────────────────────────────────────────
        # Overlays (ANNOTATED viewport only)
        # ─────────────────────────────────────────────
        self.right_title = QtWidgets.QLabel(self.plotter.interactor)
        self.right_title.setAutoFillBackground(True)
        self.right_title.setStyleSheet(
            "color:black; font-weight:bold; "
            "background-color:white; font-size:14px;"
        )
        self.right_title.setAlignment(QtCore.Qt.AlignCenter)
        self.right_title.setText("Annotated Point Cloud")
        self.right_title.show()
    
    def _set_view(self, idx: int):
        self.current_view = idx
        if hasattr(self, "ribbon_view_combo"):
            combo = self.ribbon_view_combo
            try:
                combo.blockSignals(True)
                pos = combo.findData(idx)
                if pos >= 0:
                    combo.setCurrentIndex(pos)
            finally:
                combo.blockSignals(False)
        self.apply_view(idx)        

    def open_ann_folder(self):
        fol = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Annotation PC Folder'
        )
        if not fol:
            return

        self.ann_dir = Path(fol)
        self.directory = self.ann_dir
        self.files = self._get_sorted_files()

        if not self.files:
            QtWidgets.QMessageBox.critical(
                self, 'Error', 'No PLY/PCD in Annotation folder'
            )
            return

        # Reset navigation + history FIRST
        self.index = 0
        self.history.clear()
        self.redo_stack.clear()

        # Thumbnail hygiene
        self._prune_thumbnail_cache()

        # Rebuild navigation UI
        self._populate_nav_list()

        self.load_cloud()

        # Scan annotation state only when safe
        if self.orig_dir is not None:
            QtCore.QTimer.singleShot(0, self._scan_annotated_files)


    def open_orig_folder(self):
        fol = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Original PC Folder')
        if not fol: return
        self.orig_dir = Path(fol)
        # If an annotated file is already showing, reload so toggle works now
        if self.files:
            self.load_cloud()

    def load_cloud(self):
        # PATCH 7: mark visited
        self._visited.add(self.index)
        self._decorate_nav_item(self.index)

        pc = pv.read(str(self.files[self.index]))
        self._begin_batch()
        if 'RGB' not in pc.array_names:
            pc['RGB'] = np.zeros((pc.n_points, 3), dtype=np.uint8)

        self.cloud = pc

        # Keep annotated layer as a SEPARATE numpy array (don’t point at mesh array)
        self.colors = pc['RGB'].copy()  # annotated colors currently in the file

        # Load pristine original if present; otherwise fall back to annotated
        self.original_colors = self.colors.copy()
        if self.orig_dir:
            cand = self.orig_dir / Path(self.files[self.index]).name
            if cand.exists():
                pc0 = pv.read(str(cand))
                if 'RGB' in pc0.array_names and pc0.n_points == pc.n_points:
                    self.original_colors = pc0['RGB'].copy()

        # Build KD-tree once
        self.kdtree = cKDTree(pc.points)

        # Enhanced/base starts from ORIGINAL (not annotated)
        self.enhanced_colors = self.original_colors.copy()

        # Initialize the mesh scalars to the current base so the first frame is correct
        self.cloud['RGB'] = self.enhanced_colors.astype(np.uint8)

        # Pre-fit the left view so the first frame is already centered and near-final
        self._pre_fit_camera(self.cloud, self.plotter)
        self.plotter.clear()
        self.actor = self.plotter.add_points(
            self.cloud, scalars='RGB', rgb=True, point_size=self.point_size, reset_camera=False,   # ensure False
            render_points_as_spheres=True
        )
        # RIGHT reference view (always RAW ORIGINAL, no contrast) — REPLACE
        self.plotter_ref.clear()

        ref_colors = self.original_colors.astype(np.uint8)          # ← raw original
        self.cloud_ref = pv.PolyData(self.cloud.points.copy())      # separate mesh
        self.cloud_ref['RGB'] = ref_colors

        self.actor_ref = self.plotter_ref.add_points(
            self.cloud_ref, scalars='RGB', rgb=True,
            point_size=self.point_size, reset_camera=False,          # ← no auto reset
            render_points_as_spheres=True
        )

        # Pre-fit cameras (IMPORTANT: avoid double-fitting when camera is shared in split mode)
        self._pre_fit_camera(self.cloud, self.plotter)

        if self._shared_cam_active():
            # both panes share ONE vtkCamera; do not pre-fit again using the other pane's aspect
            self.plotter_ref.renderer.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
            self._shared_camera = self.plotter.renderer.GetActiveCamera()
            self._need_split_fit = True   # ensure we fit ONCE after layout settles
        else:
            self._pre_fit_camera(self.cloud_ref, self.plotter_ref)


        # send explicit (x,y) to on_click
        self.plotter.track_click_position(lambda pos: self.on_click(pos[0], pos[1]))

        # update overlays
        self._position_overlays()
        
        self._last_gamma_value = 100
        self.on_gamma_change(100)
        self.enhanced_colors = self.original_colors.copy()

        self._session_edited = np.zeros(self.cloud.n_points, dtype=bool)
        has_any_edit_now = np.any(self.colors != self.original_colors)
        self.act_toggle_annotations.setEnabled(bool(has_any_edit_now))
        
        self.annotations_visible = getattr(self, 'act_toggle_annotations', None) is None or self.act_toggle_annotations.isChecked()
        self.update_annotation_visibility()
        self._end_batch()
        self._update_status_bar()

    def _position_overlays(self):
        # RIGHT (annotated) overlays
        if hasattr(self, 'right_title') and hasattr(self, 'plotter'):
            h1 = self.plotter.interactor.height()
            w1 = self.plotter.interactor.width()
            self.right_title.adjustSize()
            self.right_title.move((w1 - self.right_title.width()) // 2,
                                    h1 - self.right_title.height() - 2)
            self.right_title.raise_()

        # LEFT (original) label
        if hasattr(self, 'left_title') and hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
            h2 = self.plotter_ref.interactor.height()
            w2 = self.plotter_ref.interactor.width()
            self.left_title.adjustSize()
            self.left_title.move((w2 - self.left_title.width()) // 2,
                                h2 - self.left_title.height() - 2)
            self.left_title.raise_()

    def _view_direction(self):
        if self.current_view == 0:
            return np.array([0.0, 0.0, -1.0])
        if self.current_view == 1:
            return np.array([0.0, 0.0, 1.0])
        if self.current_view == 2:
            return np.array([0.0, 1.0, 0.0])
        if self.current_view == 3:
            return np.array([0.0, -1.0, 0.0])
        if self.current_view == 4:
            return np.array([1.0, 0.0, 0.0])
        if self.current_view == 5:
            return np.array([-1.0, 0.0, 0.0])
        if self.current_view == 6:
            v = np.array([1.0, 1.0, -1.0])
        elif self.current_view == 7:
            v = np.array([-1.0, 1.0, -1.0])
        elif self.current_view == 8:
            v = np.array([1.0, -1.0, -1.0])
        else:
            v = np.array([-1.0, -1.0, -1.0])
        return v / max(np.linalg.norm(v), 1e-6)

    def _fit_view(self, plotter):
        """Fit camera of a given plotter to its mesh bounds without changing orientation.
        Supports both perspective and parallel (orthographic) projections cleanly.
        """
        if getattr(self, '_is_closing', False) or getattr(self, '_batch', False):
            return
        if plotter is None or plotter.renderer is None:
            return

        # detect which mesh this plotter shows
        mesh = None
        if plotter is self.plotter and hasattr(self, 'cloud'):
            mesh = self.cloud
        elif plotter is self.plotter_ref and hasattr(self, 'cloud_ref'):
            mesh = self.cloud_ref
        if mesh is None or mesh.n_points == 0:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        xr, yr, zr = (xmax - xmin), (ymax - ymin), (zmax - zmin)
        r = 0.5 * np.linalg.norm([xr, yr, zr])
        if r <= 0:
            return

        cam = plotter.camera
        w = max(1, plotter.interactor.width())
        h = max(1, plotter.interactor.height())
        aspect = w / float(h)

        # Keep current orientation (no re-view); compute distance/scale only
        dirp = np.array(cam.GetDirectionOfProjection(), dtype=float)
        if not np.isfinite(dirp).all() or np.linalg.norm(dirp) < 1e-6:
            dirp = self._view_direction()
        dirp /= np.linalg.norm(dirp)

        if cam.GetParallelProjection():
            # In orthographic, ParallelScale is half the visible height at the focal plane.
            # Fit both width and height considering aspect.
            half_h = 0.5 * yr
            half_w = 0.5 * xr
            # To fit width, height must be >= half_w / aspect
            scale_needed = max(half_h, half_w / max(aspect, 1e-6))
            cam.SetFocalPoint(cx, cy, cz)
            # Keep current camera distance (irrelevant for parallel), but ensure position isn’t NaN
            pos = np.array(cam.GetPosition(), dtype=float)
            if not np.isfinite(pos).all():
                pos = np.array([cx, cy, cz]) - dirp * (r * 2.0 + 1.0)
            cam.SetPosition(*pos)
            cam.SetParallelScale(scale_needed * self._fit_pad)
        else:
            # Perspective: use view angle + widget aspect to compute distance
            vfov = np.deg2rad(cam.GetViewAngle())
            hfov = 2 * np.arctan(np.tan(vfov / 2) * aspect)
            eff = min(vfov, hfov)
            dist = r / np.tan(eff / 2) * self._fit_pad
            pos = np.array([cx, cy, cz]) - dirp * dist
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(*pos)

        plotter.reset_camera_clipping_range()
        if (not getattr(self, '_is_closing', False)
                and not getattr(self, '_batch', False)
                and not getattr(self, '_view_change_active', False)):
            plotter.render()

    def _mesh_bounds_in_camera_xy(self, cam, mesh):
        """
        Return (half_w, half_h) of mesh bounds measured in CAMERA/view coordinates (x,y).
        Robust for any view direction (top, iso, etc.).
        """
        # 8 corners of world AABB
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        corners = np.array([
            [xmin, ymin, zmin, 1.0],
            [xmin, ymin, zmax, 1.0],
            [xmin, ymax, zmin, 1.0],
            [xmin, ymax, zmax, 1.0],
            [xmax, ymin, zmin, 1.0],
            [xmax, ymin, zmax, 1.0],
            [xmax, ymax, zmin, 1.0],
            [xmax, ymax, zmax, 1.0],
        ], dtype=float)

        # VTK view transform: world -> camera coords
        M = cam.GetViewTransformMatrix()  # vtkMatrix4x4
        mat = np.array([[M.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)

        cam_pts = (mat @ corners.T).T  # Nx4
        x = cam_pts[:, 0]
        y = cam_pts[:, 1]

        half_w = 0.5 * float(x.max() - x.min())
        half_h = 0.5 * float(y.max() - y.min())
        return half_w, half_h


    def _fit_shared_camera_once(self, mesh):
        """
        Fit the SHARED vtkCamera so that the object fits in BOTH panes.
        This avoids the 'fit-left then fit-right' ping-pong that causes random scaling.
        """
        if mesh is None or mesh.n_points == 0:
            return
        if self._shared_camera is None:
            return

        cam = self._shared_camera

        # center of bounds
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
        center = np.array([cx, cy, cz], dtype=float)

        # camera orientation
        dop = np.array(cam.GetDirectionOfProjection(), dtype=float)
        n = float(np.linalg.norm(dop))
        if not np.isfinite(dop).all() or n < 1e-9:
            dop = np.array([0.0, 0.0, -1.0], dtype=float)
            n = 1.0
        dop /= n

        # measure object extents in camera X/Y (view coords)
        half_w, half_h = self._mesh_bounds_in_camera_xy(cam, mesh)

        # two viewports aspects (may differ slightly)
        w1 = max(1, int(self.plotter.interactor.width()))
        h1 = max(1, int(self.plotter.interactor.height()))
        a1 = w1 / float(h1)

        w2 = max(1, int(self.plotter_ref.interactor.width()))
        h2 = max(1, int(self.plotter_ref.interactor.height()))
        a2 = w2 / float(h2)

        pad = float(getattr(self, "_fit_pad", 1.10))

        cam.SetFocalPoint(cx, cy, cz)

        if cam.GetParallelProjection():
            # ParallelScale is half the visible height.
            # Need scale big enough to fit width AND height for each aspect.
            s1 = max(half_h, half_w / max(a1, 1e-6))
            s2 = max(half_h, half_w / max(a2, 1e-6))
            scale = max(s1, s2) * pad

            pos = np.array(cam.GetPosition(), dtype=float)
            if not np.isfinite(pos).all():
                # pick a safe distance (doesn't affect size in parallel)
                r = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])) or 1.0
                pos = center - dop * (r * 2.0 + 1.0)

            cam.SetPosition(*pos)
            cam.SetParallelScale(scale)

        else:
            vfov = np.deg2rad(float(cam.GetViewAngle()))
            vfov = max(vfov, 1e-3)

            def dist_needed(aspect):
                hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * max(aspect, 1e-6))
                hfov = max(hfov, 1e-3)
                d_h = half_h / np.tan(vfov / 2.0)
                d_w = half_w / np.tan(hfov / 2.0)
                return max(d_h, d_w)

            dist = max(dist_needed(a1), dist_needed(a2)) * pad
            cam.SetPosition(*(center - dop * dist))

    def _fit_to_canvas(self):
        if getattr(self, '_is_closing', False) or getattr(self, '_batch', False):
            return

        split = self._is_split_mode()
        shared = bool(split and self._shared_camera is not None)

        if shared:
            # IMPORTANT: fit ONCE (camera is shared)
            mesh = getattr(self, "cloud", None)
            if mesh is None or mesh.n_points == 0:
                mesh = getattr(self, "cloud_ref", None)
            if mesh is None:
                return

            self._fit_shared_camera_once(mesh)

            try:
                self.plotter.reset_camera_clipping_range()
                self.plotter_ref.reset_camera_clipping_range()
            except Exception:
                pass

            if getattr(self, '_view_change_active', False):
                self._view_change_active = False
                self._sync_renders()
            else:
                self._sync_renders()
            return

        # --- non-shared / single-pane behavior ---
        if hasattr(self, 'plotter'):
            self._fit_view(self.plotter)
        if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
            self._fit_view(self.plotter_ref)
        if getattr(self, '_view_change_active', False):
            self._view_change_active = False
            self._render_views_once()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'right_title'):
            self._position_overlays()
        if not getattr(self, '_batch', False):
            self._schedule_fit()  # coalesce multiple resize fits

    def apply_view(self, idx: int = None):
        """
        Top view  -> orthographic, look straight down +Z with +Y up.
        Isometric -> perspective, SOUTH-WEST isometric (from -X,-Y, +Z) with Z up.
        We set orientation explicitly for both panes, then defer the fit.
        """
        topdown = (self.current_view == 0)  # 0 = Top view
        self._view_change_active = True
        self._cam_pause = True
        views = [getattr(self, 'plotter', None), getattr(self, 'plotter_ref', None)]
        for view in views:
            if view:
                try:
                    view.interactor.setUpdatesEnabled(False)
                except Exception:
                    pass

        def _apply(plotter, mesh, topdown: bool):
            if plotter is None or mesh is None or mesh.n_points == 0:
                return
            cam = plotter.camera
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0

            if topdown:
                # Orthographic straight-down
                cam.ParallelProjectionOn()
                cam.SetViewUp(0, 1, 0)             # keep text/UI upright
                dop = np.array([0.0, 0.0, -1.0])   # look down +Z
            elif self.current_view == 1:
                cam.ParallelProjectionOn()
                cam.SetViewUp(0, 1, 0)             # keep text/UI upright
                dop = np.array([0.0, 0.0, 1.0])   # look up +Z
            elif self.current_view == 2:
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([0.0, 1.0, 0.0])   # look along -Y
            elif self.current_view == 3:
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([0.0, -1.0, 0.0])    # look along +Y
            elif self.current_view == 4:
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([1.0, 0.0, 0.0])   # look along -X
            elif self.current_view == 5:
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([-1.0, 0.0, 0.0])    # look along +X
            elif self.current_view == 6:
                # SOUTH-WEST isometric: from (-X, -Y, +Z) toward center
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([1.0, 1.0, -1.0])   # direction-of-projection (to center)
                dop /= np.linalg.norm(dop)
            elif self.current_view == 7:
                # SOUTH-EAST isometric: from (+X, -Y, +Z) toward center
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([-1.0, 1.0, -1.0])  # direction-of-projection (to center)
                dop /= np.linalg.norm(dop)
            elif self.current_view == 8:
                # NORTH-WEST isometric: from (-X, +Y, +Z) toward center
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([1.0, -1.0, -1.0])  # direction-of-projection (to center)
                dop /= np.linalg.norm(dop)
            elif self.current_view == 9:
                # NORTH-EAST isometric: from (+X, +Y, +Z) toward center
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([-1.0, -1.0, -1.0]) # direction-of-projection (to center)
                dop /= np.linalg.norm(dop)

            cam.SetFocalPoint(cx, cy, cz)

            # Rough position; final distance is set by _fit_view later (we just set orientation)
            r = 0.5 * np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]) or 1.0
            w = max(1, plotter.interactor.width()); h = max(1, plotter.interactor.height())
            vfov = np.deg2rad(cam.GetViewAngle())
            hfov = 2*np.arctan(np.tan(vfov/2) * (w/float(h)))
            eff = min(vfov, hfov)
            dist = r / np.tan(max(eff, 1e-3)/2) * float(getattr(self, '_fit_pad', 1.12))
            pos = np.array([cx, cy, cz]) - dop * dist
            cam.SetPosition(*pos)

            plotter.reset_camera_clipping_range()

        try:
            # Left (annotated)
            _apply(self.plotter, getattr(self, 'cloud', None), topdown)

            # Right (original)
            if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
                _apply(self.plotter_ref, getattr(self, 'cloud_ref', None), topdown)
        finally:
            for view in views:
                if view:
                    try:
                        view.interactor.setUpdatesEnabled(True)
                    except Exception:
                        pass
            self._cam_pause = False

        # Defer the actual fit so it uses final widget sizes
        self._schedule_fit()

    def reset_view(self):
        self.plotter.reset_camera()
        self.apply_view()

    def toggle_annotation(self):
        if self.act_annotation_mode.isChecked():
            # just change cursor; leave interactor active for everything but left-drag
            self.update_cursor()
        else:
            self.plotter.interactor.unsetCursor()

    def update_cursor(self):
        """
        Cursor ring shows the exact paint footprint when using the strict brush:
        effective radius = brush_radius_px - 0.5 * point_size_px.
        """
        # Brush & sprite sizes in display pixels
        r_px  = max(1, int(self.brush_size))    # brush radius (px)
        ps_px = max(1, int(self.point_size))    # rendered point size (px)

        # Strict footprint (never paints outside the ring)
        r_eff = int(round(max(1.0, r_px - 0.5 * ps_px)))
        d     = 2 * r_eff

        # Make a transparent pixmap a bit larger to avoid stroke clipping
        pix = QPixmap(d + 4, d + 4)
        pix.fill(QtCore.Qt.transparent)

        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = p.pen()
        pen.setColor(QColor(255, 0, 255))  # magenta
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(2, 2, d, d)
        p.end()

        # Hotspot at the center of the ring
        if self.clone_mode:
            # Clone → cursor ONLY on Original (left)
            self.plotter_ref.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))
            self.plotter.interactor.unsetCursor()

        elif self.repair_mode:
            # Repair → cursor ONLY on Annotated (right)
            self.plotter.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))
            self.plotter_ref.interactor.unsetCursor()

        else:
            # Normal annotation
            self.plotter.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))
            self.plotter_ref.interactor.unsetCursor()


    def change_brush(self, val):
        v = int(max(1, min(val, 200)))
        self.brush_size = float(v)
        # update ribbon label
        if hasattr(self, "ribbon_sliders") and "brush" in self.ribbon_sliders:
            _, lbl = self.ribbon_sliders["brush"]
            lbl.setText(f"{v} px")
        if self.act_annotation_mode.isChecked():
            self.update_cursor()


    def change_point(self, val):
        """Update rendered point size and keep 'round points' sticky."""
        v = max(1, min(int(val), 20))
        self.point_size = v

        def _apply_point_size(actor, render_fn):
            try:
                prop = actor.GetProperty()
                prop.SetPointSize(v)
                # keep round sprites ON even after size changes (sticky)
                try:
                    prop.SetRenderPointsAsSpheres(True)
                except Exception:
                    # older VTK may not have this setter; ignore gracefully
                    pass
                if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                    render_fn()
            except Exception:
                pass

        if hasattr(self, 'actor') and self.actor is not None:
            _apply_point_size(self.actor, self.plotter.render)

        if hasattr(self, 'actor_ref') and self.actor_ref is not None:
            _apply_point_size(self.actor_ref, self.plotter_ref.render)

        # Cursor ring depends on point size in strict brush mode — keep it in sync
        if self.act_annotation_mode.isChecked():
            self.update_cursor()


    def pick_color(self):
        if self.clone_mode:
            return
        
        c = QtWidgets.QColorDialog.getColor()
        if c.isValid():
            self.current_color = [c.red(), c.green(), c.blue()]
            self._last_paint_color = self.current_color.copy()      # NEW
            self.act_eraser.setChecked(False)                       # NEW


    def select_swatch(self, col, btn=None):
        """
        Select paint color from menu or picker.
        UI-agnostic: no widgets, no swatches.
        """
        # 🚫 Ignore color changes in Clone mode
        if self.clone_mode:
            return

        qc = QColor(col)
        self.current_color = [qc.red(), qc.green(), qc.blue()]
        self._last_paint_color = self.current_color.copy()

        # Exit eraser mode if active
        if hasattr(self, "act_eraser"):
            self.act_eraser.setChecked(False)


    def on_click(self, x, y):
        if not self.act_annotation_mode.isChecked():
            return
        # 1) pick center
        picker = vtkPropPicker()
        h = self.plotter.interactor.height()
        picker.Pick(x, h - y, 0, self.plotter.renderer)
        pt = np.array(picker.GetPickPosition())
        if np.allclose(pt, (0,0,0)):
            return

        # 2) pick an edge pixel at (x  slider_px, y) to get world distance
        r_px = self.brush_size
        picker.ErasePickList()                  # clear previous
        picker.Pick(x + r_px, h - y, 0, self.plotter.renderer)
        pt_edge = np.array(picker.GetPickPosition())

        # 3) compute world‐space radius and query
        world_r = np.linalg.norm(pt_edge - pt)
        idx = self.kdtree.query_ball_point(pt, world_r)
        
        if not idx:
            return
        # record for undo/redo
        old = self.colors[idx].copy()
        self.history.append((idx, old))
        self.redo_stack.clear()
        # apply new color (eraser only when eraser is active)
        if self.clone_mode:
            # Clone: copy from original
            self.colors[idx] = self.original_colors[idx]

        elif self.act_eraser.isChecked() or self.current_color is None:
            # Eraser: revert to original
            self.colors[idx] = self.original_colors[idx]

        else:
            # Normal paint
            self.colors[idx] = self.current_color
        
        self._session_edited[idx] = True
        self._mark_dirty_once()
        self.toggle_ann_chk.setEnabled(bool(np.any(self._session_edited)))
        
        # push update back into the plot
        self.update_annotation_visibility()

    def _maybe_autosave_before_nav(self):
        if self.act_autosave.isChecked():
            edited = (
                getattr(self, '_session_edited', None) is not None
                and np.any(self._session_edited)
            )
            if edited:
                self.on_save(_autosave=True)

    def on_prev(self):
        if not self.files:
            return

        self._maybe_autosave_before_nav()

        if self.index > 0:
            self.index -= 1
        else:
            # wrap around to last
            self.index = len(self.files) - 1

        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._sync_nav_selection()
        self._update_status_bar()
        
    def on_next(self):
        if not self.files:
            return

        self._maybe_autosave_before_nav()

        if self.index < len(self.files) - 1:
            self.index += 1
        else:
            # wrap around to first
            self.index = 0

        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._sync_nav_selection()
        self._update_status_bar()

    def on_first(self):
        if not self.files:
            return
        self._maybe_autosave_before_nav()
        self.index = 0
        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._sync_nav_selection()
        self._update_status_bar()

    def on_last(self):
        if not self.files:
            return
        self._maybe_autosave_before_nav()
        self.index = len(self.files) - 1
        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._sync_nav_selection()
        self._update_status_bar()

    def on_page(self, delta: int):
        """Jump by +/-N (default N=10 via PgUp/PgDown). Wraps like Next/Prev."""
        if not self.files:
            return
        self._maybe_autosave_before_nav()

        n = len(self.files)
        if n <= 0:
            return

        self.index = (self.index + int(delta)) % n  # wrap-around behavior (consistent with on_next/on_prev)

        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._sync_nav_selection()
        self._update_status_bar()

    def _toggle_loop(self, on: bool):
        if on:
            delay_ms = int(self.loop_delay_sec * 1000)
            self._loop_timer.start(delay_ms)
        else:
            self._loop_timer.stop()
            
        self._update_loop_status()    

    def _on_loop_tick(self):
        # behave exactly like pressing Next
        self.on_next()

    def on_undo(self):
        if not self.history:
            return
        idx, old = self.history.pop()
        self.redo_stack.append((idx, self.colors[idx].copy()))
        self.colors[idx] = old
        self._session_edited[idx] = False
        self.act_toggle_annotations.setEnabled(bool(np.any(self._session_edited)))
        self.update_annotation_visibility()

    def on_redo(self):
        if not self.redo_stack:
            return
        idx, cols = self.redo_stack.pop()
        self.history.append((idx, self.colors[idx].copy()))
        self.colors[idx] = cols
        self._session_edited[idx] = True
        self.act_toggle_annotations.setEnabled(bool(np.any(self._session_edited)))
        self.update_annotation_visibility()

    def on_save(self, _autosave: bool = False):
        out = Path(self.files[self.index])
        ext = out.suffix.lower()

        if _autosave:
            # Silent choice for autosave: keep it conservative (no gamma baking)
            choice = QtWidgets.QMessageBox.Yes   # choice = QtWidgets.QMessageBox.Yes (Gamma baking) | QtWidgets.QMessageBox.No (keep as is)
        else:
            # ——— Ask user whether to save with gamma-enhanced colors ———
            # ——— Ask user whether to save with gamma-enhanced colors ———
            choice = QtWidgets.QMessageBox.question(
                self, 'Save Options',
                'Save with contrast-enhanced colors (Gamma adjusted)?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )

        # ——— Decide which colors to save ———
        save_colors = self.colors.copy()

        if choice == QtWidgets.QMessageBox.Yes:
            # only update untouched points to enhanced colors
            untouched_mask = ~np.any(save_colors != self.original_colors, axis=1)
            save_colors[untouched_mask] = self.enhanced_colors[untouched_mask]

        # assign new RGB array before saving
        self.cloud['RGB'] = save_colors

        # ——— Write file depending on format ———
        if ext == '.ply':
            writer = vtkPLYWriter()
            writer.SetFileName(str(out))
            writer.SetInputData(self.cloud)
            writer.SetArrayName('RGB')
            writer.SetFileTypeToBinary()
            writer.Write()

        elif ext == '.pcd':
            self.cloud.save(str(out), binary=True)

        else:
            self.cloud.save(str(out))

        # Mark current cloud as saved for this session
        try:
            self._session_edited = np.zeros(self.cloud.n_points, dtype=bool)
            self.toggle_ann_chk.setEnabled(bool(np.any(self._session_edited)))
        except Exception:
            pass

        if not _autosave:
            QtWidgets.QMessageBox.information(
                self, 'Saved',
                f'Successfully saved {ext[1:]} file with colors to and reloaded:\n{out}'
            )
            
        self._dirty.discard(self.index)
        self._annotated.add(self.index)
        self._decorate_nav_item(self.index)
        self._update_status_bar()

    def _on_plus(self):
        if self._waiting == "brush":
            self._nudge_slider(self.ribbon_sliders["brush"][0], +2)
        elif self._waiting == "point":
            self._nudge_slider(self.ribbon_sliders["point"][0], +1)
        elif self._waiting == "alpha":
            self._nudge_slider(self.ribbon_sliders["alpha"][0], +5)
        elif self._waiting == "gamma":
            self._nudge_slider(self.ribbon_sliders["gamma"][0], +5)
        elif self._waiting == "zoom":
            self.plotter.camera.Zoom(1.1)
            self.plotter.render()

    def _on_minus(self):
        if self._waiting == "brush":
            self._nudge_slider(self.ribbon_sliders["brush"][0], -2)
        elif self._waiting == "point":
            self._nudge_slider(self.ribbon_sliders["point"][0], -1)
        elif self._waiting == "alpha":
            self._nudge_slider(self.ribbon_sliders["alpha"][0], -5)
        elif self._waiting == "gamma":
            self._nudge_slider(self.ribbon_sliders["gamma"][0], -5)
        elif self._waiting == "zoom":
            self.plotter.camera.Zoom(0.9)
            self.plotter.render()
       
    def on_zoom_in(self):
        # behave exactly like “Z +”
        self._waiting = 'zoom'
        self._on_plus()

    def on_zoom_out(self):
        # behave exactly like “Z –”
        self._waiting = 'zoom'
        self._on_minus()

        
    def eventFilter(self, obj, event):
        if getattr(self, '_is_closing', False):
            return False
        
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_B:
                self._waiting = "brush"
                return True

            if event.key() == QtCore.Qt.Key_D:
                self._waiting = "point"
                return True

            if event.key() == QtCore.Qt.Key_A:
                self._waiting = "alpha"
                return True

            if event.key() == QtCore.Qt.Key_Z:
                self._waiting = "zoom"
                return True
            
            if event.key() == QtCore.Qt.Key_G:          # ✅ ADD
                self._waiting = "gamma"
                return True
            
        if event.type() == QtCore.QEvent.KeyRelease:
            if event.key() in (QtCore.Qt.Key_B,
                            QtCore.Qt.Key_D,
                            QtCore.Qt.Key_A,
                            QtCore.Qt.Key_Z,
                            QtCore.Qt.Key_G):
                self._waiting = None

        
        # PATCH 3.7: if Qt tries to re-focus the nav box after Enter, bounce it back once
        if obj is getattr(self, "nav_search", None):
            if event.type() == QtCore.QEvent.FocusIn and getattr(self, "_nav_release_pending", False):
                self._nav_release_pending = False
                QtCore.QTimer.singleShot(0, lambda: self.plotter.interactor.setFocus())
                return True  # swallow this focus-in so caret doesn't appear

        # navigation search box
        if obj is getattr(self, "nav_search", None):
            if event.type() == QtCore.QEvent.ShortcutOverride:
                k = event.key()
                if k in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                         QtCore.Qt.Key_Home, QtCore.Qt.Key_End,
                         QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
                    event.accept()
                    return True

            if event.type() == QtCore.QEvent.KeyPress:
                k = event.key()
                if k == QtCore.Qt.Key_Left:
                    self.on_prev()
                    return True
                if k == QtCore.Qt.Key_Right:
                    self.on_next()
                    return True
                if k == QtCore.Qt.Key_Home:
                    self.on_first()
                    return True
                if k == QtCore.Qt.Key_End:
                    self.on_last()
                    return True
                if k == QtCore.Qt.Key_PageUp:
                    self.on_page(-10)
                    return True
                if k == QtCore.Qt.Key_PageDown:
                    self.on_page(+10)
                    return True
                if k == QtCore.Qt.Key_Escape:
                    self.nav_search.clear()
                    return True


        # RIGHT pane wheel
        if obj is self.plotter.interactor and event.type() == QtCore.QEvent.Wheel:
            if getattr(self, '_stroke_active', False) or getattr(self, '_in_zoom', False):
                return False
            self._zoom_at_cursor_for(self.plotter, event.x(), event.y(), event.angleDelta().y())
            return True


        # LEFT pane wheel (original view)
        if obj is getattr(self, 'plotter_ref', None).interactor and event.type() == QtCore.QEvent.Wheel:
            # If the left pane isn’t currently shown, don’t swallow the wheel event.
            if not self.plotter_ref.isVisible():
                return False
            # When cameras are linked in Repair mode, drive the shared camera via the right plotter.
            target = self.plotter if (self.repair_mode or self.clone_mode) and self._shared_camera is not None else self.plotter_ref
            self._zoom_at_cursor_for(target, event.x(), event.y(), event.angleDelta().y())
            return True

        if self.act_annotation_mode.isChecked():
            if self.clone_mode:
                # Clone mode → ONLY allow painting on ORIGINAL (left) window
                if obj is not self.plotter_ref.interactor:
                    return False
            else:
                # Normal / Repair mode → ONLY paint on ANNOTATED (right) window
                if obj is not self.plotter.interactor:
                    return False
        
            # 1) start stroke on LeftButtonPress
            if event.type() == QtCore.QEvent.MouseButtonPress \
            and event.button() == QtCore.Qt.LeftButton:
                self._stroke_active        = True
                self._in_stroke            = True
                self._stroke_idxs.clear()
                self._colors_before_stroke = self.colors.copy()

                x0, y0 = event.x(), event.y()
                self._last_paint_xy        = (x0, y0)

                # Shift = constrain to a straight line from the anchor
                self._constrain_line = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
                if self._constrain_line:
                    self._anchor_xy   = (x0, y0)
                    self._line_len_px = 0.0
                else:
                    self._anchor_xy   = None
                    self._line_len_px = 0.0
                return True

            # 2) paint on LeftButton + Move
            if self._stroke_active \
            and event.type() == QtCore.QEvent.MouseMove \
            and (event.buttons() & QtCore.Qt.LeftButton):

                # Throttle heavy work (≈120 Hz)
                if self._paint_timer.elapsed() < getattr(self, '_min_paint_ms', 8):
                    return True
                self._paint_timer.restart()

                r_px = max(1, self.brush_size)
                step_px = max(1.0, r_px * float(getattr(self, "_paint_step_frac", 0.8)))

                touched_idx = []
                changed_any = False

                if self._constrain_line and self._anchor_xy:
                    # —— Straight line from anchor to current —— 
                    ax, ay = self._anchor_xy
                    x2, y2 = event.x(), event.y()
                    vx, vy = (x2 - ax), (y2 - ay)
                    dist   = math.hypot(vx, vy)

                    if dist >= 1.0:
                        start_len = float(self._line_len_px)
                        end_len   = dist
                        delta     = end_len - start_len
                        n_steps   = int(delta / step_px)

                        # Only stamp the NEW segment beyond what we've already painted
                        for k in range(1, n_steps + 1):
                            t_len = start_len + k * step_px
                            t     = min(1.0, t_len / dist)
                            xx    = ax + vx * t
                            yy    = ay + vy * t

                            idx = self._compute_brush_idx(xx, yy)
                            if not idx:
                                continue
                            self._stroke_idxs.update(idx)
                            if self.clone_mode:
                                # Clone: copy from original
                                self.colors[idx] = self.original_colors[idx]

                            elif self.act_eraser.isChecked() or self.current_color is None:
                                # Eraser: revert to original
                                self.colors[idx] = self.original_colors[idx]

                            else:
                                # Normal paint
                                self.colors[idx] = self.current_color
                            self._session_edited[idx] = True
                            self._mark_dirty_once() 
                            touched_idx.append(idx)
                            changed_any = True

                        self._line_len_px = end_len
                else:
                    # —— Freehand with interpolated stamping —— 
                    x2, y2 = event.x(), event.y()
                    x1, y1 = self._last_paint_xy if self._last_paint_xy else (x2, y2)
                    dist   = math.hypot(x2 - x1, y2 - y1)
                    n_steps = max(1, int(dist / step_px))

                    for k in range(1, n_steps + 1):
                        t  = k / float(n_steps)
                        xx = x1 + (x2 - x1) * t
                        yy = y1 + (y2 - y1) * t

                        idx = self._compute_brush_idx(xx, yy)
                        if not idx:
                            continue
                        self._stroke_idxs.update(idx)
                        if self.clone_mode:
                            # Clone: copy from original
                            self.colors[idx] = self.original_colors[idx]

                        elif self.act_eraser.isChecked() or self.current_color is None:
                            # Eraser: revert to original
                            self.colors[idx] = self.original_colors[idx]

                        else:
                            # Normal paint
                            self.colors[idx] = self.current_color

                        self._session_edited[idx] = True
                        self._mark_dirty_once() 
                        touched_idx.append(idx)
                        changed_any = True

                    self._last_paint_xy = (x2, y2)

                if changed_any:
                    self.act_toggle_annotations.setEnabled(bool(np.any(self._session_edited)))
                    flat = np.concatenate(touched_idx) if touched_idx else np.array([], dtype=int)
                    if flat.size:
                        self._blend_into_mesh_subset(flat)
                    self._stroke_render_timer.start(0)
                return True

            # 3) finish on LeftButtonRelease
            if self._stroke_active \
            and event.type() == QtCore.QEvent.MouseButtonRelease \
            and event.button() == QtCore.Qt.LeftButton:
                self._stroke_active  = False
                self._in_stroke      = False
                self._last_paint_xy  = None
                self._anchor_xy      = None
                self._line_len_px    = 0.0
                self._constrain_line = False

                if self._stroke_idxs:
                    idxs = list(self._stroke_idxs)
                    old  = self._colors_before_stroke[idxs]
                    self.history.append((idxs, old))
                    self.redo_stack.clear()
                self._colors_before_stroke = None

                # One full refresh to ensure global consistency
                self.update_annotation_visibility()
                return True

        # all other events (including two-finger pan, right-drag, etc.) go to QtInteractor
        return super().eventFilter(obj, event)
        
    def _on_eraser_toggled(self, on: bool):
        # Always ensure annotation mode is ON when toggling eraser
        if not self.act_annotation_mode.isChecked():
            self.act_annotation_mode.setChecked(True)
            self.update_cursor()

        if on:
            # Enter eraser mode
            self.current_color = None
        else:
            # Leave eraser mode → restore last paint color
            self.current_color = self._last_paint_color.copy()
           
    def reset_contrast(self):
        if "gamma" in self.ribbon_sliders:
            gamma_slider, gamma_lbl = self.ribbon_sliders["gamma"]
            gamma_slider.blockSignals(True)
            gamma_slider.setValue(100)
            gamma_slider.blockSignals(False)
            gamma_lbl.setText("1.00")

        current = self.colors.copy()
        untouched_mask = np.all(current == self.original_colors, axis=1)
        current[untouched_mask] = self.original_colors[untouched_mask]
        self.enhanced_colors = self.original_colors.copy()

        self.cloud['RGB'] = current
        self.update_annotation_visibility()

        if self.repair_mode and hasattr(self, 'cloud_ref'):
            self.cloud_ref['RGB'] = self.original_colors.astype(np.uint8)
            if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                self.plotter_ref.render()
                
    def on_gamma_change(self, val):
        gamma = 2 ** ((val - 100) / 50.0)
        
        # Optional UI updates (never required)
        if hasattr(self, "ribbon_gamma_label"):
            self.ribbon_gamma_label.setText(f"{gamma:.2f}")
        elif hasattr(self, "tool_sliders"):
            try:
                _, lbl = self.tool_sliders.gamma
                lbl.setText(f"{gamma:.2f}")
            except Exception:
                pass

        original = self.original_colors.astype(np.float32) / 255.0
        min_vals = original.min(axis=0, keepdims=True)
        max_vals = original.max(axis=0, keepdims=True)
        stretched = (original - min_vals) / (max_vals - min_vals + 1e-5)

        corrected = np.power(stretched, gamma)
        self.enhanced_colors = (corrected * 255).astype(np.uint8)

        current = self.colors.copy()
        mask = np.all(current == self.original_colors, axis=1)
        current[mask] = self.enhanced_colors[mask]

        self.cloud['RGB'] = current
        self.update_annotation_visibility()

        if self.repair_mode and hasattr(self, 'cloud_ref'):
            self.cloud_ref['RGB'] = self.original_colors.astype(np.uint8)
            if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                self.plotter_ref.render()

    def apply_auto_contrast(self):
        # Normalize RGB to [0,1]
        rgb = self.original_colors.astype(np.float32) / 255.0

        # Stretch per channel based on percentiles (e.g., 2% to 98%)
        p_low, p_high = 2, 98
        lo = np.percentile(rgb, p_low, axis=0)
        hi = np.percentile(rgb, p_high, axis=0)

        stretched = (rgb - lo) / (hi - lo + 1e-5)
        stretched = np.clip(stretched, 0, 1)
        self.enhanced_colors = (stretched * 255).astype(np.uint8)

        # Apply only to unpainted points
        current = self.colors.copy()
        mask = np.all(current == self.original_colors, axis=1)
        current[mask] = self.enhanced_colors[mask]

        self.cloud['RGB'] = current
        self.update_annotation_visibility()

        # In repair mode, ORIGINAL view must stay raw (no contrast) — REPLACE/ENSURE
        if self.repair_mode and hasattr(self, 'cloud_ref'):
            self.cloud_ref['RGB'] = self.original_colors.astype(np.uint8)
            if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                self.plotter_ref.render()


        # Also update gamma label to reflect "Auto"
        if "gamma" in self.ribbon_sliders:
            gamma_slider, gamma_lbl = self.ribbon_sliders["gamma"]
            gamma_slider.blockSignals(True)
            gamma_slider.setValue(100)   # logical reset point
            gamma_slider.blockSignals(False)
            gamma_lbl.setText("Auto")

        
    def show_histograms(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Smoothed RGB Distributions — Original vs Enhanced")

        # Labels and colors
        channels = ['Red', 'Green', 'Blue']
        colors = ['r', 'g', 'b']
        linestyles = ['-', '--']  # solid = original, dashed = enhanced

        for i, (label, color) in enumerate(zip(channels, colors)):
            # Original
            orig_vals = self.original_colors[:, i].astype(np.float32)
            kde_orig = gaussian_kde(orig_vals)
            x = np.linspace(0, 255, 256)
            ax.plot(x, kde_orig(x), color=color, linestyle=linestyles[0], label=f"{label} (Original)")

            # Enhanced
            enh_vals = self.enhanced_colors[:, i].astype(np.float32)
            kde_enh = gaussian_kde(enh_vals)
            ax.plot(x, kde_enh(x), color=color, linestyle=linestyles[1], label=f"{label} (Enhanced)")

        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.show(block=False)
        
    def set_annotations_visible(self, vis: bool):
        self.annotations_visible = bool(vis)
        # keep UI in sync without retriggering signals
        if hasattr(self, 'toggle_ann_chk'):
            self.toggle_ann_chk.blockSignals(True)
            self.toggle_ann_chk.setChecked(self.annotations_visible)
            self.toggle_ann_chk.blockSignals(False)
        self.update_annotation_visibility()

    def _current_base(self):
        """Whatever contrast is active now: enhanced if available, else original."""
        base = getattr(self, 'enhanced_colors', None)
        if base is None or len(base) != len(self.original_colors):
            base = self.original_colors
        return base
    
    def _clone_source(self):
        """Color source for Clone mode."""
        return self.original_colors

    def _on_toggle_ann_changed(self, state):
        self.annotations_visible = (state == QtCore.Qt.Checked)
        self.update_annotation_visibility()
        
    def on_alpha_change(self, val):                 # NEW
        self.annotation_alpha = max(0.0, min(1.0, val / 100.0))
        self.update_annotation_visibility()

    def update_annotation_visibility(self):
        if getattr(self, '_is_closing', False):
            return
        
        if not hasattr(self, 'cloud') or self.cloud is None:
            return

        # Base: current contrast on ORIGINAL baseline (your existing logic)
        base = getattr(self, 'enhanced_colors', None)
        if base is None or len(base) != len(self.original_colors):
            base = self.original_colors
        base = base.astype(np.uint8)

        # Start from base
        display = base.copy()

        # If annotations are hidden, just show base
        if not getattr(self, 'annotations_visible', True):
            self.cloud['RGB'] = display.astype(np.uint8)
            if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                self.plotter.render()
            return

        # Edited points = where annotated layer differs from original baseline
        edited_mask = np.any(self.colors != self.original_colors, axis=1)
        if not np.any(edited_mask):
            self.cloud['RGB'] = display.astype(np.uint8)
            if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                self.plotter.render()
            return

        # Alpha blend only on edited points
        a = float(getattr(self, 'annotation_alpha', 1.0))
        if a >= 0.999:
            display[edited_mask] = self.colors[edited_mask]
        elif a <= 0.001:
            # effectively invisible; keep base
            pass
        else:
            fg = self.colors[edited_mask].astype(np.float32)
            bg = base[edited_mask].astype(np.float32)
            out = (a * fg + (1.0 - a) * bg).round().astype(np.uint8)
            display[edited_mask] = out

        self.cloud['RGB'] = display.astype(np.uint8)
        if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
            self.plotter.render()


    def toggle_repair_mode(self, on: bool):
        
        if on:
            self._need_split_fit = True   # ← NEW

        if on and self.clone_mode:
            self.act_clone.setChecked(False)    # ← force exit Clone first
        
        self.repair_mode = bool(on)
        self.plotter_ref.setVisible(self.repair_mode)
        self.vline.setVisible(self.repair_mode)

        # Force Annotation Mode ON while repairing
        if self.repair_mode and not self.act_annotation_mode.isChecked():
            self.act_annotation_mode.setChecked(True)
            
        # Always enable Annotation Mode in Repair and refresh cursor
        if self.repair_mode:
            # Make sure the checkbox state is ON (don’t rely on prior state)
            self.act_annotation_mode.blockSignals(True)
            self.act_annotation_mode.setChecked(True)
            self.act_annotation_mode.blockSignals(False)
            self.update_cursor()                  # ensure magenta ring cursor is active

            # Auto-turn Eraser ON (you can toggle it OFF to paint)
            self.act_eraser.setChecked(True)
        else:
            # leaving repair mode: no change to eraser or annotation mode
            self.act_eraser.setChecked(False)
            # (Leaving repair mode: we leave eraser state as-is — no change)

        # Show/Hide left label
        if hasattr(self, 'left_title'):
            self.left_title.setVisible(self.repair_mode)

        # Refresh left scalars (original base)
        if self.repair_mode and hasattr(self, 'cloud_ref'):
            self.cloud_ref['RGB'] = self.original_colors.astype(np.uint8)  # ← raw original

        # ← NEW: link/unlink cameras
        if self.repair_mode:
            self._link_cameras()
        else:
            self._unlink_cameras()

        # Reposition overlays after layout change
        QtCore.QTimer.singleShot(0, self._position_overlays)
        self._schedule_fit()
        self.update_annotation_visibility()
    
    def toggle_clone_mode(self, on: bool):        
        if on:
            self._need_split_fit = True
        self.clone_mode = bool(on)     
        
        if on and self.repair_mode:
            self.act_repair.setChecked(False)   # ← force exit Repair first
        
        if self.clone_mode:
            # Ensure annotation tools are active (as you wanted)
            self.act_annotation_mode.setChecked(True)
            self.act_toggle_annotations.setChecked(True)    

            # Show original (left) panel + title
            self.plotter_ref.setVisible(True)
            self.vline.setVisible(self.clone_mode)
            if hasattr(self, 'left_title'):
                self.left_title.setVisible(True)

            # ✅ KEY: use the same shared-camera behavior as Repair
            self._link_cameras()

        else:
            # Stop sharing camera first (keeps current view)
            self._unlink_cameras()

            # Hide original panel + title
            self.plotter_ref.setVisible(False)
            self.vline.setVisible(False)
            if hasattr(self, 'left_title'):
                self.left_title.setVisible(False)
                
            # restore last paint color
            self.current_color = self._last_paint_color.copy()

        # Layout + fit after the splitter settles
        QtCore.QTimer.singleShot(0, self._finalize_layout)        
        self.update_cursor()

    def closeEvent(self, e):
        self._is_closing = True  # block any future renders

        # stop watching events on the VTK interactor
        try:
            self.plotter.interactor.removeEventFilter(self)
            self.plotter_ref.interactor.removeEventFilter(self)
        except Exception:
            pass

        # close the two QtInteractor windows safely
        for view in [getattr(self, 'plotter_ref', None), getattr(self, 'plotter', None)]:
            try:
                if view is not None:
                    view.close()
            except Exception:
                pass
            
        try:
            self._loop_timer.stop()
        except Exception:
            pass

        try:
            self._loop_timer.stop()
            self.statusBar().clearMessage()   # PATCH 10
        except Exception:
            pass

        try:
            state = {}
            if STATE_FILE.exists():
                state = json.loads(STATE_FILE.read_text())

            state.update({
                'annotation_dir': str(self.ann_dir or ''),
                'original_dir':   str(self.orig_dir or ''),
                'index':          int(self.index),
                'nav_dock_width': int(self.nav_dock.width()),
            })
            STATE_FILE.write_text(json.dumps(state))

        except Exception:
            pass
        super().closeEvent(e)
        
    def _sync_renders(self):
        """Render both views safely (avoid recursion / closing / batching)."""
        if (self._cam_syncing or getattr(self, '_is_closing', False)
                or getattr(self, '_batch', False) or getattr(self, '_cam_pause', False)):  # NEW
            return
        self._cam_syncing = True
        try:
            if hasattr(self, 'plotter'):
                self.plotter.render()
            if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
                self.plotter_ref.render()
        finally:
            self._cam_syncing = False


    def _link_cameras(self):
        """Make both panels share the same vtkCamera and keep renders in sync."""
        if not hasattr(self, 'plotter') or not hasattr(self, 'plotter_ref'):
            return
        cam = self.plotter.renderer.GetActiveCamera()
        self.plotter_ref.renderer.SetActiveCamera(cam)      # share the SAME camera instance
        # Observe camera changes to render both views
        if self._cam_observer_id is None:
            self._shared_camera = cam
            self._cam_observer_id = cam.AddObserver("ModifiedEvent", lambda *_: self._sync_renders())
        # one initial sync render
        self._sync_renders()

    def _unlink_cameras(self):
        """Detach the right panel from the shared camera (but keep current view)."""
        if self._shared_camera is not None and self._cam_observer_id is not None:
            try:
                self._shared_camera.RemoveObserver(self._cam_observer_id)
            except Exception:
                pass
        self._cam_observer_id = None
        # Give right panel its own camera, duplicating current pose so it doesn’t jump
        try:
            if hasattr(self, 'plotter_ref'):
                new_cam = vtkCamera()
                new_cam.DeepCopy(self.plotter.renderer.GetActiveCamera())
                self.plotter_ref.renderer.SetActiveCamera(new_cam)
        except Exception:
            pass
        self._shared_camera = None        
        
    def _begin_batch(self):
        self._batch = True
        self._cam_pause = True

        # DO NOT unlink cameras here (causes camera instance churn + pumping)
        self._cam_snap_l = self._snap_camera(self.plotter)
        # In split/shared camera mode, snapping the "right" is redundant and can reintroduce mismatch
        self._cam_snap_r = None if self._shared_cam_active() else (
            self._snap_camera(self.plotter_ref) if self.plotter_ref.isVisible() else None
        )

        for view in [getattr(self, 'plotter', None), getattr(self, 'plotter_ref', None)]:
            if view:
                view.interactor.setUpdatesEnabled(False)

    def _finalize_layout(self):
        if getattr(self, '_is_closing', False):
            return

        # Block shared-camera observer renders while we settle
        self._cam_pause = True

        split = self._is_split_mode()

        try:
            # Restore previous camera pose (prevents visible zoom pumping)
            if hasattr(self, "_cam_snap_l") and self._cam_snap_l:
                self._restore_camera(self.plotter, self._cam_snap_l)

            if split:
                cam = self.plotter.renderer.GetActiveCamera()
                self.plotter_ref.renderer.SetActiveCamera(cam)
                self._shared_camera = cam

                # Restore camera pose (IMPORTANT: in split mode camera is shared → restore ONCE)
                if hasattr(self, "_cam_snap_l") and self._cam_snap_l:
                    self._restore_camera(self.plotter, self._cam_snap_l)
                    # plotter_ref shares the same camera; do NOT restore a second snapshot


                # ✅ ONE-TIME fit when entering split mode
                if getattr(self, "_need_split_fit", False):
                    self._fit_to_canvas()
                    self._need_split_fit = False

                self.plotter.reset_camera_clipping_range()
                self.plotter_ref.reset_camera_clipping_range()

            else:
                # Normal single-pane behavior: keep your existing view + fit behavior
                self.apply_view()
                try:
                    self._fit_to_canvas()
                except Exception:
                    pass

            # overlays after layout settles
            self._position_overlays()

        finally:
            self._cam_pause = False

        # One final refresh only
        self._sync_renders()

        # Clear snapshots
        self._cam_snap_l = None
        self._cam_snap_r = None

    def _pre_fit_camera(self, mesh, plotter):
        """
        Set camera orientation + an initial distance/scale using mesh bounds,
        before any actors are added. No render here.
        """
        if mesh is None or mesh.n_points == 0 or plotter is None:
            return

        cam = plotter.camera
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
        xr, yr, zr = (xmax - xmin), (ymax - ymin), (zmax - zmin)
        r = 0.5 * float(np.linalg.norm([xr, yr, zr])) or 1.0

        w = max(1, plotter.interactor.width())
        h = max(1, plotter.interactor.height())
        aspect = w / float(h)
        pad = float(getattr(self, "_fit_pad", 1.12))

        is_parallel = self.current_view in (0, 1)
        dop = self._view_direction()
        dop /= max(np.linalg.norm(dop), 1e-6)

        if is_parallel:
            # Orthographic top/bottom
            cam.ParallelProjectionOn()
            cam.SetViewUp(0, 1, 0)
            # ParallelScale ≈ half visible height
            scale_h = 0.5 * yr
            scale_w = 0.5 * xr / max(aspect, 1e-6)
            cam.SetParallelScale(max(scale_h, scale_w) * pad)
            pos = np.array([cx, cy, cz]) - dop * (r * 2.0 + 1.0)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(*pos)
        else:
            # Perspective views
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            vfov = np.deg2rad(cam.GetViewAngle())
            hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect)
            eff = max(1e-3, min(vfov, hfov))
            dist = r / np.tan(eff / 2.0) * pad
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(*(np.array([cx, cy, cz]) - dop * dist))

    def _end_batch(self):
        for view in [getattr(self, 'plotter', None), getattr(self, 'plotter_ref', None)]:
            if view:
                view.interactor.setUpdatesEnabled(True)

        self._batch = False
        self._cam_pause = False
        QtCore.QTimer.singleShot(0, self._finalize_layout)


    def _schedule_fit(self, delay=None):
        """Coalesce multiple fit requests into one."""
        if getattr(self, '_is_closing', False):
            return
        if getattr(self, '_batch', False):
            return  # during batch, we defer fits until _end_batch()
        if delay is None:
            delay = getattr(self, '_fit_delay_ms', 33)
        self._fit_timer.stop()
        self._fit_timer.start(int(delay))

    def _render_views_once(self):
        if getattr(self, '_is_closing', False) or getattr(self, '_batch', False):
            return
        try:
            if hasattr(self, 'plotter'):
                self.plotter.render()
            if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
                self.plotter_ref.render()
        except Exception:
            pass

    def _blend_into_mesh_subset(self, idx):
        """
        Update self.cloud['RGB'][idx] only, reflecting current annotation visibility/alpha.
        """
        if idx is None or len(idx) == 0:
            return

        # current baseline (enhanced if available, else original)
        base = getattr(self, 'enhanced_colors', None)
        if base is None or len(base) != len(self.original_colors):
            base = self.original_colors

        if not getattr(self, 'annotations_visible', True):
            # annotations hidden → show base colors
            self.cloud['RGB'][idx] = base[idx].astype(np.uint8)
            return

        a = float(getattr(self, 'annotation_alpha', 1.0))
        if a >= 0.999:
            self.cloud['RGB'][idx] = self.colors[idx].astype(np.uint8)
        elif a <= 0.001:
            self.cloud['RGB'][idx] = base[idx].astype(np.uint8)
        else:
            fg = self.colors[idx].astype(np.float32)
            bg = base[idx].astype(np.float32)
            out = (a * fg + (1.0 - a) * bg).round().astype(np.uint8)
            self.cloud['RGB'][idx] = out
            
    def _zoom_at_cursor_for(self, plotter, x: int, y: int, delta_y: int):
        """
        Fluid, infinite zoom anchored at the cursor for a given plotter (left or right).
        Atomic when cameras are linked to avoid shake.
        """
        if plotter is None:
            return
        ren   = plotter.renderer
        inter = plotter.interactor
        cam   = plotter.camera
        H     = inter.height()

        # ====== ATOMIC SECTION when shared camera is active ======
        atomic = bool((self.repair_mode or self.clone_mode) and self._shared_camera is not None)

        if atomic:
            if self._in_zoom:   # prevent re-entrancy from nested events
                return
            self._in_zoom = True
            self._cam_pause = True                 # pause _sync_renders()
            # Temporarily stop UI updates to avoid mid-step paints
            try:
                self.plotter.interactor.setUpdatesEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'plotter_ref'):
                    self.plotter_ref.interactor.setUpdatesEnabled(False)
            except Exception:
                pass

        try:
            # -------- helpers --------
            def ray_through_xy(renderer, xx, yy):
                renderer.SetDisplayPoint(float(xx), float(H - yy), 0.0); renderer.DisplayToWorld()
                x0, y0, z0, w0 = renderer.GetWorldPoint()
                if abs(w0) > 1e-12: x0, y0, z0 = x0 / w0, y0 / w0, z0 / w0

                renderer.SetDisplayPoint(float(xx), float(H - yy), 1.0); renderer.DisplayToWorld()
                x1, y1, z1, w1 = renderer.GetWorldPoint()
                if abs(w1) > 1e-12: x1, y1, z1 = x1 / w1, y1 / w1, z1 / w1

                o = np.array([x0, y0, z0], dtype=float)
                d = np.array([x1, y1, z1], dtype=float) - o
                n = float(np.linalg.norm(d))
                if n < 1e-12:
                    o = np.array(cam.GetPosition(), dtype=float)
                    d = np.array(cam.GetFocalPoint(), dtype=float) - o
                    n = float(np.linalg.norm(d))
                return o, (d / max(n, 1e-12))

            # --- pre-zoom camera state + focal-plane anchor under cursor ---
            pos0 = np.array(cam.GetPosition(),   dtype=float)
            fp0  = np.array(cam.GetFocalPoint(), dtype=float)
            vu0  = np.array(cam.GetViewUp(),     dtype=float)

            n0 = fp0 - pos0
            n0n = float(np.linalg.norm(n0))
            if n0n < 1e-12:
                return
            n0 /= n0n

            o0, d0 = ray_through_xy(ren, x, y)
            denom0 = float(np.dot(d0, n0))
            anchor = fp0.copy() if abs(denom0) < 1e-12 else (o0 + d0 * float(np.dot(fp0 - o0, n0) / denom0))

            # --- smooth zoom (same “feel”) ---
            factor = 1.0 if delta_y == 0 else (1.2 ** (delta_y / 120.0))
            if factor <= 0.0:
                return

            if cam.GetParallelProjection():
                cam.SetParallelScale(cam.GetParallelScale() / max(1e-6, factor))
                shift = (anchor - fp0) * (1.0 - 1.0 / factor)
                cam.SetFocalPoint(*(fp0 + shift))
                cam.SetPosition(*(pos0 + shift))
                cam.SetViewUp(*vu0)
            else:
                cam.SetPosition(* (anchor + (pos0 - anchor) / factor))
                cam.SetFocalPoint(* (anchor + (fp0  - anchor) / factor))
                cam.SetViewUp(*vu0)

            # --- force post-zoom cursor ray to pass through the SAME anchor ---
            pos1 = np.array(cam.GetPosition(),   dtype=float)
            fp1  = np.array(cam.GetFocalPoint(), dtype=float)
            n1   = fp1 - pos1; n1n = float(np.linalg.norm(n1))
            if n1n >= 1e-12:
                n1 /= n1n
                o1, d1 = ray_through_xy(ren, x, y)
                denom1 = float(np.dot(d1, n1))
                if abs(denom1) > 1e-12:
                    t1  = float(np.dot(anchor - o1, n1) / denom1)
                    q   = o1 + d1 * t1
                    pan = anchor - q
                    if np.isfinite(pan).all():
                        cam.SetPosition(*(pos1 + pan))
                        cam.SetFocalPoint(*(fp1 + pan))
                        cam.SetViewUp(*vu0)

            # Defer clipping until the very end to avoid extra matrix churn
        finally:
            if atomic:
                # Resume UI updates
                try:
                    self.plotter.interactor.setUpdatesEnabled(True)
                except Exception:
                    pass
                try:
                    if hasattr(self, 'plotter_ref'):
                        self.plotter_ref.interactor.setUpdatesEnabled(True)
                except Exception:
                    pass
                self._cam_pause = False
                self._in_zoom = False

        # Single, synchronized refresh (prevents “shake”)
        try:
            plotter.reset_camera_clipping_range()
            if self.repair_mode and getattr(self, 'plotter_ref', None) is not None and self.plotter_ref.isVisible():
                # Render both once
                self._sync_renders()
            else:
                plotter.render()
        except Exception:
            pass
        
    def _on_nav_search_entered(self):
        """Patch 3: Go to index or filename from nav dock."""
        if not self.files:
            return

        text = self.nav_search.text().strip()
        
        # PATCH 3.2: empty input behaves like old goto_edit
        if not text:
            self.nav_search.clear()
            self.nav_search.clearFocus()
            try:
                self.plotter.interactor.setFocus()
            except Exception:
                self.setFocus()
            return

        # --- Case 1: numeric → index (1-based) ---
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(self.files):
                self._maybe_autosave_before_nav()
                self.index = idx
                self.history.clear()
                self.redo_stack.clear()
                self.load_cloud()
                self._position_overlays()
                self._sync_nav_selection()
                self._update_status_bar()
                self._nav_release_pending = True
                QtCore.QTimer.singleShot(0, self._reset_nav_search)
            else:
                self.nav_status.setText("Index out of range")
            return

        # --- Case 2: filename substring search ---
        text_low = text.lower()
        matches = [
            i for i, p in enumerate(self.files)
            if text_low in p.name.lower()
        ]

        if not matches:
            self.nav_status.setText("No matching filenames")
            return

        # Jump to the FIRST match (deterministic & fast)
        idx = matches[0]
        self._maybe_autosave_before_nav()
        self.index = idx
        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._sync_nav_selection()
        self._update_status_bar()

            
        # PATCH 3.6: visually exit the command field (kill blinking caret)
        self._nav_release_pending = True
        QtCore.QTimer.singleShot(0, self._reset_nav_search)
        
    # PATCH 3.6 (final): force command field to exit editing mode
    def _reset_nav_search(self):
        # PATCH 3.7: clear text and push focus away (Qt may try to steal it back once)
        try:
            self.nav_search.blockSignals(True)
            self.nav_search.clear()
            self.nav_search.deselect()
            self.nav_search.blockSignals(False)
        except Exception:
            pass

        # Put focus back to the 3D view
        try:
            self.plotter.interactor.setFocus()
        except Exception:
            self.setFocus()

    def _nav_row_text(self, i: int) -> str:
        """Row label: '0001 | filename.ply' (1-based index)."""
        if not self.files:
            return ""
        idx_w = max(4, len(str(len(self.files))))
        return f"{i+1:0{idx_w}d} | {self.files[i].name}"

    def _populate_nav_list(self):
        if not hasattr(self, "nav_list"):
            return

        self._nav_item_widgets = {}
        with self._thumb_lock:
            self._thumb_out_by_idx = {}
            self._thumb_job_set.clear()

        self.nav_list.blockSignals(True)
        self.nav_list.clear()

        if not self.files:
            self.nav_list.blockSignals(False)
            return

        for i in range(len(self.files)):
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(NAV_THUMB_SIZE + 16, NAV_THUMB_SIZE + 48))
            item.setData(QtCore.Qt.UserRole, i)

            w = self._make_nav_item_widget(i)

            self.nav_list.addItem(item)
            self.nav_list.setItemWidget(item, w)

            # backend thumbnail request (Patch 5)
            self._request_thumbnail(i)

        self.nav_list.blockSignals(False)

        # Sync selection AFTER items exist
        self._sync_nav_selection()

        # Replay state AFTER widgets exist (CRITICAL)
        max_idx = len(self.files)
        for idx in (self._dirty | self._annotated | self._visited):
            if 0 <= idx < max_idx:
                self._decorate_nav_item(idx)

    def _sync_nav_selection(self):
        """Keep nav list selection in sync with self.index."""
        if not hasattr(self, "nav_list") or not self.files:
            return
        i = int(getattr(self, "index", 0))
        if i < 0 or i >= self.nav_list.count():
            return

        self.nav_list.blockSignals(True)
        self.nav_list.setCurrentRow(i)
        self.nav_list.scrollToItem(self.nav_list.currentItem(), QtWidgets.QAbstractItemView.PositionAtCenter)
        self.nav_list.blockSignals(False)

    def _on_nav_row_changed(self, row: int):
        """Single-click navigation from nav list."""
        if not self.files:
            return
        if row < 0 or row >= len(self.files):
            return
        if row == self.index:
            return  # avoid reload loop

        self._maybe_autosave_before_nav()

        self.index = row
        self.history.clear()
        self.redo_stack.clear()
        self.load_cloud()
        self._position_overlays()
        self._update_status_bar()


    def _thumb_key(self, ann_path: Path) -> str:
        """
        Stable thumbnail key.
        Hashes ONLY the ORIGINAL file if available.
        Annotation edits must NOT affect thumbnails.
        """
        # Choose source of truth
        if self.orig_dir is not None:
            orig = self.orig_dir / ann_path.name
            src = orig if orig.exists() else ann_path
        else:
            src = ann_path

        st = src.stat()

        h = hashlib.sha1()
        h.update(str(src.resolve()).encode("utf-8"))   # full path (folder identity)
        h.update(str(st.st_size).encode("utf-8"))      # content proxy
        h.update(str(int(st.st_mtime)).encode("utf-8"))# original timestamp
        return h.hexdigest()


    def _thumb_path(self, path: Path) -> Path:
        return THUMB_DIR / f"{self._thumb_key(path)}.png"

    def _thumb_exists(self, path: Path) -> bool:
        return self._thumb_path(path).exists()

    def _generate_thumbnail(self, path: Path, out_png: Path, size=96):
        """
        Generate a thumbnail PNG for a point cloud.
        Runs in background thread. NO Qt calls allowed here.
        """
        _generate_thumbnail_job(path, out_png, size)

    def _request_thumbnail(self, idx: int):
        """
        Schedule thumbnail generation.
        ALWAYS use original folder as source of truth if available.
        """
        if not self.files or idx < 0 or idx >= len(self.files):
            return

        ann_path = self.files[idx]

        # ✅ SOURCE OF TRUTH DECISION HERE
        if self.orig_dir is not None:
            orig_path = self.orig_dir / ann_path.name
            src_path = orig_path if orig_path.exists() else ann_path
        else:
            src_path = ann_path

        out_png = self._thumb_path(ann_path)

        if out_png.exists():
            return

        with self._thumb_lock:
            self._thumb_out_by_idx[idx] = out_png
            self._thumb_job_set.add((src_path, out_png))

        if not self._thumb_worker_running and not self._thumb_worker_start_pending:
            self._thumb_worker_start_pending = True
            QtCore.QTimer.singleShot(0, self._start_thumb_worker)

    def _start_thumb_worker(self):
        if getattr(self, "_thumb_worker_running", False):
            return

        self._thumb_worker_start_pending = False
        self._thumb_worker_running = True

        def worker():
            try:
                while True:
                    with self._thumb_lock:
                        jobs = list(self._thumb_job_set)
                        self._thumb_job_set.clear()

                    if not jobs:
                        break

                    Parallel(
                        n_jobs=THUMB_N_JOBS,
                        backend=THUMB_BACKEND,
                        verbose=0
                    )(
                        delayed(_generate_thumbnail_job)(src, out, THUMB_SIZE)
                        for src, out in jobs
                    )
            finally:
                self._thumb_worker_running = False

        threading.Thread(target=worker, daemon=True).start()

    def _thumb_icon_for_index(self, idx: int):
        """
        Return QIcon for thumbnail if available, else None.
        UI-safe (Qt only, no disk generation).
        """
        try:
            path = self.files[idx]
            png = self._thumb_path(path)
            if not png.exists():
                return None
            pix = QPixmap(str(png))
            if pix.isNull():
                return None
            pix = pix.scaled(
                THUMB_SIZE, THUMB_SIZE,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            return QIcon(pix)
        except Exception:
            return None

    def _refresh_nav_thumbnail(self, idx: int):
        entry = self._nav_item_widgets.get(idx)
        if entry is None:
            return
        lbl = entry["img"]
        
        if lbl is None:
            return

        icon = self._thumb_icon_for_index(idx)
        if icon is None:
            return

        pix = icon.pixmap(NAV_THUMB_SIZE, NAV_THUMB_SIZE)
        lbl.setPixmap(pix)

    def _poll_thumbnails(self):
        """Refresh UI icons when thumbnail files appear on disk."""
        with self._thumb_lock:
            pending = list(self._thumb_out_by_idx.items())

        updated = []
        for idx, out_png in pending[:60]:
            if out_png.exists():
                updated.append(idx)

        if updated:
            with self._thumb_lock:
                for idx in updated:
                    self._thumb_out_by_idx.pop(idx, None)

            for idx in updated:
                self._refresh_nav_thumbnail(idx)

            if hasattr(self, "nav_list"):
                self.nav_list.viewport().update()
            
        self._update_status_bar()


    def _nav_display_name(self, name: str) -> str:
        if len(name) <= NAV_NAME_MAX:
            return name
        return name[:NAV_NAME_MAX - 1] + "…"
    
    def _make_nav_item_widget(self, idx: int):
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

        # --- Thumbnail container (for overlays) ---
        thumb_container = QtWidgets.QFrame()
        thumb_container.setFixedSize(NAV_THUMB_SIZE, NAV_THUMB_SIZE)
        thumb_container.setStyleSheet("background: transparent;")
        thumb_container.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        thumb_layout = QtWidgets.QStackedLayout(thumb_container)
        thumb_layout.setContentsMargins(0, 0, 0, 0)

        # Thumbnail image
        lbl_img = QtWidgets.QLabel()
        lbl_img.setAlignment(QtCore.Qt.AlignCenter)

        icon = self._thumb_icon_for_index(idx)
        if icon is not None:
            lbl_img.setPixmap(icon.pixmap(NAV_THUMB_SIZE, NAV_THUMB_SIZE))

        thumb_layout.addWidget(lbl_img)

        # 🔴 Dirty indicator (top-right)
        dot_dirty = QtWidgets.QLabel(thumb_container)
        dot_dirty.setFixedSize(10, 10)
        dot_dirty.setStyleSheet("background:red; border-radius:5px;")
        dot_dirty.move(NAV_THUMB_SIZE - 10, 2)
        dot_dirty.hide()

        # 🟢 Annotated indicator (bottom-right)
        dot_annot = QtWidgets.QLabel(thumb_container)
        dot_annot.setFixedSize(10, 10)
        dot_annot.setStyleSheet("background:green; border-radius:5px;")
        dot_annot.move(NAV_THUMB_SIZE - 10, NAV_THUMB_SIZE - 10)
        dot_annot.hide()

        # --- Text below ---
        name = self.files[idx].name
        txt = f"{idx+1:04d}\n{self._nav_display_name(name)}"

        lbl_txt = QtWidgets.QLabel(txt)
        lbl_txt.setAlignment(QtCore.Qt.AlignCenter)
        lbl_txt.setWordWrap(True)
        lbl_txt.setStyleSheet("font-size:11px;")

        lay.addWidget(thumb_container)
        lay.addWidget(lbl_txt)

        # Store references for updates
        self._nav_item_widgets[idx] = {
            "root": w,
            "img": lbl_img,
            "dirty": dot_dirty,
            "annotated": dot_annot,
        }

        # Apply current state
        self._decorate_nav_item(idx)

        return w
        
    def _scan_annotated_files(self):
        """Detect which files are annotated on disk (joblib)."""
        if not self.files or not self.orig_dir:
            return

        pairs = []
        for p in self.files:
            o = self.orig_dir / p.name
            if o.exists():
                pairs.append((p, o))

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(_is_annotated_pair)(a, o) for a, o in pairs
        )

        self._annotated.clear()
        for i, is_ann in enumerate(results):
            if is_ann:
                self._annotated.add(i)
                
        for idx in self._annotated:
            self._decorate_nav_item(idx)

    def _decorate_nav_item(self, idx: int):
        if not hasattr(self, "_nav_item_widgets"):
            return

        entry = self._nav_item_widgets.get(idx)
        if entry is None:
            return

        root = entry["root"]
        dot_dirty = entry["dirty"]
        dot_annot = entry["annotated"]

        # Visited → subtle background
        if idx in self._visited:
            root.setStyleSheet("background:#d0e7ff;")
        else:
            root.setStyleSheet("")

        # Dirty / Annotated dots
        dot_dirty.setVisible(idx in self._dirty)
        dot_annot.setVisible(idx in self._annotated)

    def _mark_dirty_once(self):
        """Mark current cloud dirty once per session."""
        if self.index not in self._dirty:
            self._dirty.add(self.index)
            self._decorate_nav_item(self.index)

    def _clear_thumbnail_cache(self):
        if not THUMB_DIR.exists():    
            QtWidgets.QMessageBox.warning(
                self, "Thumbnail Cache",
                "Thumbnail cache directory is not initialized yet."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Thumbnail Cache",
            "Delete the entire thumbnail cache folder?\n"
            "Thumbnails will be regenerated automatically.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        import shutil

        try:
            if THUMB_DIR.exists():
                shutil.rmtree(THUMB_DIR)

            # Recreate empty cache dir
            THUMB_DIR.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Failed to clear thumbnail cache:\n{e}"
            )
            return

        # Clear UI thumbnails immediately
        if hasattr(self, "_nav_item_widgets"):
            for entry in self._nav_item_widgets.values():
                entry["img"].clear()

        QtWidgets.QMessageBox.information(
            self, "Thumbnail Cache", "Thumbnail cache cleared."
        )


    def _prune_thumbnail_cache(self):
        if self.orig_dir is None or not THUMB_DIR.exists():
            return

        valid_keys = set()
        for p in self.orig_dir.glob("*.ply"):
            valid_keys.add(self._thumb_key(p))

        for thumb in THUMB_DIR.glob("*.png"):
            if thumb.stem not in valid_keys:
                try:
                    thumb.unlink()
                except Exception:
                    pass
    
    def _nudge_slider(self, slider, delta):
        if not slider.isEnabled():
            return
        v = slider.value() + delta
        v = max(slider.minimum(), min(slider.maximum(), v))
        slider.setValue(v)

    def _update_loop_status(self):
        self._update_status_bar()

    def _update_status_bar(self):
        # Viewing filename (LEFT)
        if self.files:
            self.sb_viewing.setText(f"Viewing: {self.files[self.index].name}")
        else:
            self.sb_viewing.setText("")

        # Index: "12 / 248"
        if self.files:
            self.sb_index.setText(f"File Index: {self.index + 1} / {len(self.files)}")
        else:
            self.sb_index.setText("")

        # Annotation state
        is_dirty = self.index in getattr(self, "_dirty", set())
        is_annot = self.index in getattr(self, "_annotated", set())
        if is_dirty:
            self.sb_anno.setText("Modified")
        elif is_annot:
            self.sb_anno.setText("Annotated")
        else:
            if len(getattr(self, "_thumb_out_by_idx", {})) > 0:
                self.sb_anno.setText("Processing Thumbnails…")
            else:
                self.sb_anno.setText("Clean")

        # Loop state (Patch 10)
        if self.act_loop.isChecked():
            self.sb_loop.setText(f"Looping ({self.loop_delay_sec:.1f} s)")
        else:
            self.sb_loop.setText("")

        # Thumbnail progress (optional but useful)
        total = len(self.files) if self.files else 0
        pending = len(getattr(self, "_thumb_out_by_idx", {}))
        done = max(0, total - pending)
        if total > 0:
            self.sb_thumb.setText(f"Thumbs: {done}/{total}")
        else:
            self.sb_thumb.setText("")
            
    def _restore_nav_width(self):
        try:
            if not STATE_FILE.exists():
                return

            st = json.loads(STATE_FILE.read_text())
            w = int(st.get('nav_dock_width', NAV_DOCK_WIDTH))

            self.resizeDocks(
                [self.nav_dock],
                [w],
                QtCore.Qt.Horizontal
            )
        except Exception:
            pass
        
    def _on_ribbon_delay_changed(self, val: float):
        self.loop_delay_sec = float(val)
        # If looping, restart timer with new interval
        if self.act_loop.isChecked():
            self._toggle_loop(False)
            self._toggle_loop(True)
        self._update_status_bar()

    def _on_ribbon_alpha(self, v: int):
        # update value label
        _, lbl = self.ribbon_sliders["alpha"]
        lbl.setText(f"{int(v)}%")
        self.on_alpha_change(v)  # uses your existing slot

    def _on_ribbon_brush(self, v: int):
        # update value label, then call your existing method
        _, lbl = self.ribbon_sliders["brush"]
        lbl.setText(f"{int(v)} px")
        self.change_brush(v)

    def _on_ribbon_point(self, v: int):
        _, lbl = self.ribbon_sliders["point"]
        lbl.setText(f"{int(v)} px")
        self.change_point(v)

    def _on_ribbon_gamma(self, v: int):
        # store last gamma for persistence-like behavior
        self._last_gamma_value = int(v)
        # let your existing gamma logic compute the numeric label
        self.on_gamma_change(v)
        
    def _set_loop_delay(self, val: float):
        """
        Set loop delay in seconds.
        Safe to call from ribbon, menu, or dialogs.
        """
        self.loop_delay_sec = float(val)

        # If looping is active, restart timer with new delay
        if self.act_loop.isChecked():
            self._toggle_loop(False)
            self._toggle_loop(True)

        self._update_status_bar()

    def _is_split_mode(self) -> bool:
        return bool(self.repair_mode or self.clone_mode) and hasattr(self, "plotter_ref") and self.plotter_ref.isVisible()

    def _shared_cam_active(self) -> bool:
        return bool(self._is_split_mode() and self._shared_camera is not None)

    def show_about_dialog(self):
        QtWidgets.QMessageBox.about(
            self,
            "About Point Cloud Annotator",
            """
    <b>Point Cloud Annotator</b><br>
    Version 2.0.0<br><br>

    <b>Description</b><br>
    Point Cloud Annotator is a professional tool for semantic annotation,
    repair, and review of large-scale 3D point clouds (PLY / PCD).
    It is designed for high-precision research, industrial inspection,
    and dataset generation workflows.<br><br>

    <b>Key Features</b>
    <ul>
    <li>Brush-based semantic painting with undo/redo</li>
    <li>Repair and clone modes with synchronized dual views</li>
    <li>Gamma-based contrast enhancement and auto-contrast</li>
    <li>Fast navigation with thumbnails and loop playback</li>
    <li>Session persistence and autosave support</li>
    </ul>

    <b>Technology Stack</b><br>
    Python · PyQt5 · PyVista · VTK · NumPy · SciPy<br><br>

    <b>License</b><br>
    MIT License<br><br>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, subject to the following conditions:<br><br>

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.<br><br>

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.<br><br>

    © 2026 Preetham Manjunatha. All rights reserved.
    """
        )


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Annotator()
    win.show()
    sys.exit(app.exec_())