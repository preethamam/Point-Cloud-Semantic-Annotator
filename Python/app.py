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
  - Open folder via button
  - Maximized window with top-down or isometric initial view
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
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from appdirs import user_data_dir
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QCursor, QIcon, QKeySequence, QPainter, QPixmap
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from vtkmodules.vtkIOPLY import vtkPLYWriter
from vtkmodules.vtkRenderingCore import vtkPropPicker, vtkCamera

APP_NAME = "Point Cloud Annotator"
state_dir = Path(user_data_dir(APP_NAME, appauthor=False))
state_dir.mkdir(parents=True, exist_ok=True)
STATE_FILE = state_dir / "state.json"

class Annotator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Icon and window
        icon = Path(__file__).parent / 'icon.png'
        if not icon.exists(): icon = Path(__file__).parent / 'app.ico'
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.setWindowTitle("Point Cloud Annotator")
        # State
        self.brush_size = 0.08
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
        self._is_closing = False
        
        self._shared_camera = None           # NEW: camera shared by left/right in repair
        self._cam_observer_id = None         # NEW: observer handle for unlinking
        self._cam_syncing = False            # NEW: re-entrancy guard
        
        self._batch = False
        self._cam_pause = False      # NEW: pause camera-sync renders during loads

        self._fit_pad = 1.12
                
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

        self._stroke_render_timer = QtCore.QTimer(self)
        self._stroke_render_timer.setSingleShot(True)
        self._stroke_render_timer.timeout.connect(self._render_views_once)
        
        # ——— Paint throttling (limit to ~120 FPS) ———
        self._paint_timer = QtCore.QElapsedTimer()
        self._paint_timer.start()
        self._min_paint_ms = 8   # ~8 ms between paint batches (~120 Hz)
        
        self._constrain_line = False   # Shift-drag = straight line
        self._anchor_xy = None         # (x,y) where the line started
        self._line_len_px = 0.0        # distance already painted along the line (pixels)

        # Build UI
        self._build_ui()        
            
        # allow us to catch mouse‐moves on the VTK widget
        self.plotter.interactor.setMouseTracking(True)
        self.plotter.interactor.installEventFilter(self)
        self._stroke_active           = False
        self._stroke_idxs             = set()
        self._colors_before_stroke    = None
        # Global shortcuts
        s = QtWidgets.QShortcut
        def _on_key_A():
            self.annot_chk.toggle()                 # keep original behavior
            setattr(self, '_waiting', 'alpha')      # NEW: A now targets alpha for +/-
            
        s(QKeySequence(QtCore.Qt.Key_A), self, activated=_on_key_A)
        s(QKeySequence('R'), self, activated=self.reset_view)
        s(QKeySequence.Undo, self, activated=self.on_undo)
        s(QKeySequence.Redo, self, activated=self.on_redo)
        s(QKeySequence(QtCore.Qt.Key_Left), self, activated=self.on_prev)
        s(QKeySequence(QtCore.Qt.Key_Right), self, activated=self.on_next)
        s(QKeySequence.Save, self, activated=self.on_save)
        
        
        sc_toggle = QtWidgets.QShortcut(QKeySequence('Shift+A'), self)
        sc_toggle.setContext(QtCore.Qt.ApplicationShortcut)   # <— important
        sc_toggle.activated.connect(lambda: self.toggle_ann_chk.setChecked(
            not self.toggle_ann_chk.isChecked()))
        
        for key, mode in [('B','brush'), ('D','point')]:
            sc = QtWidgets.QShortcut(QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(lambda m=mode: setattr(self, '_waiting', m))
        for plus in ['+','=']:
            sc = QtWidgets.QShortcut(QKeySequence(plus), self)
            sc.setContext(QtCore.Qt.ApplicationShortcut)
            sc.activated.connect(self._on_plus)
        sc = QtWidgets.QShortcut(QKeySequence('-'), self)
        sc.setContext(QtCore.Qt.ApplicationShortcut)
        sc.activated.connect(self._on_minus)
        
        z_sc = QtWidgets.QShortcut(QKeySequence('Z'), self)
        z_sc.setContext(QtCore.Qt.ApplicationShortcut)
        z_sc.activated.connect(lambda: setattr(self, '_waiting', 'zoom'))
        
        e_sc = QtWidgets.QShortcut(QKeySequence('E'), self)
        e_sc.setContext(QtCore.Qt.ApplicationShortcut)
        e_sc.activated.connect(self.activate_eraser)
        
        sc_repair = QtWidgets.QShortcut(QKeySequence('Shift+R'), self)     # NEW
        sc_repair.setContext(QtCore.Qt.ApplicationShortcut)                # NEW
        sc_repair.activated.connect(lambda: self.repair_btn.toggle())

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
                else:
                    self.directory, self.files = None, []
            except:
                pass
        # Show and load
        self.showMaximized()
        if self.files:
            self._enable_controls()
            self.load_cloud()

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
        r_px = float(max(1, self.brush_slider.value()))          # brush radius (px)
        s_px = 0.5 * float(max(1, self.point_slider.value()))    # round sprite radius (px)

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


    def _build_ui(self):
        w = QtWidgets.QWidget(); self.setCentralWidget(w)
        lay = QtWidgets.QHBoxLayout(w)
        # 3D viewport
        self.plotter = QtInteractor(self)
        self.plotter_ref = QtInteractor(self)      
        
        # Anti-alias points/lines (8x MSAA)
        try:
            self.plotter.ren_win.SetMultiSamples(8)
        except Exception:
            pass
        try:
            self.plotter_ref.ren_win.SetMultiSamples(8)
        except Exception:
            pass

        # Title/overlay for RIGHT (Original) panel  — NEW
        self.right_title = QtWidgets.QLabel(self.plotter_ref.interactor)  # NEW
        self.right_title.setAutoFillBackground(True)                      # NEW
        self.right_title.setStyleSheet('color:black; font-weight:bold; background-color:white; font-size:14px;')  # NEW
        self.right_title.setText('Original Point Cloud')                           # NEW
        self.right_title.hide()    
        self.plotter_ref.set_background('white')               # NEW
        self.plotter_ref.setVisible(False)                     # NEW
        lay.addWidget(self.plotter_ref.interactor, stretch=4)
        
        self.plotter.set_background('white')  # light gray background [0.961, 0.961, 0.961, 1.0]
        lay.addWidget(self.plotter.interactor, stretch=4)
        
        # Overlays
        self.counter_label = QtWidgets.QLabel(self.plotter.interactor)
        # make background fully opaque to clear old text
        self.counter_label.setAutoFillBackground(True)
        self.counter_label.setStyleSheet('color:black; font-weight:bold; background-color:white; font-size:14px;')
        self.counter_label.show()
        self.filename_label = QtWidgets.QLabel(self.plotter.interactor)
        # make background fully opaque to clear old text
        self.filename_label.setAutoFillBackground(True)
        self.filename_label.setStyleSheet('color:black; font-weight:bold; background-color:white; font-size:14px;')
        self.filename_label.setAlignment(QtCore.Qt.AlignCenter)
        self.filename_label.show()
        # Controls panel
        ctrl = QtWidgets.QVBoxLayout(); lay.addLayout(ctrl,stretch=1)
        
        folder_row = QtWidgets.QHBoxLayout()
        self.open_ann_btn  = QtWidgets.QPushButton('Open Annotation PC Folder')
        self.open_orig_btn = QtWidgets.QPushButton('Open Original PC Folder')
        self.open_ann_btn.clicked.connect(self.open_ann_folder)
        self.open_orig_btn.clicked.connect(self.open_orig_folder)
        folder_row.addWidget(self.open_ann_btn)
        folder_row.addWidget(self.open_orig_btn)
        ctrl.addLayout(folder_row)
        
        line0 = QtWidgets.QFrame()
        line0.setFrameShape(QtWidgets.QFrame.HLine)
        line0.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl.addWidget(line0)
        
        
        annot_row = QtWidgets.QHBoxLayout()
        self.annot_chk = QtWidgets.QCheckBox('Annotation Mode (A)')
        self.annot_chk.stateChanged.connect(self.toggle_annotation)
        annot_row.addWidget(self.annot_chk)

        self.toggle_ann_chk = QtWidgets.QCheckBox('Toggle Annotations (Shift+A)')
        self.toggle_ann_chk.setChecked(True)
        self.toggle_ann_chk.stateChanged.connect(self._on_toggle_ann_changed)  # <— new slot
        annot_row.addWidget(self.toggle_ann_chk)

        annot_row.addStretch(1)
        ctrl.addLayout(annot_row)
        
        alpha_row = QtWidgets.QHBoxLayout()
        alpha_row.addWidget(QtWidgets.QLabel('Annotations Alpha (A +/-):'))
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)     # NEW
        self.alpha_slider.setRange(0, 100)                              # NEW
        self.alpha_slider.setValue(int(self.annotation_alpha * 100))    # NEW
        self.alpha_slider.valueChanged.connect(self.on_alpha_change)    # NEW
        alpha_row.addWidget(self.alpha_slider)
        ctrl.addLayout(alpha_row)

        line1 = QtWidgets.QFrame()
        line1.setFrameShape(QtWidgets.QFrame.HLine)
        line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl.addWidget(line1)

        ctrl.addWidget(QtWidgets.QLabel('Initial View:'))
        self.view_combo = QtWidgets.QComboBox(); self.view_combo.addItems(['Top-Down','Isometric']); self.view_combo.currentIndexChanged.connect(self.apply_view)
        ctrl.addWidget(self.view_combo)
        
        # ————— Reset + Zoom controls —————
        zoom_row = QtWidgets.QHBoxLayout()
    
        # Reset View
        rv = QtWidgets.QPushButton('Reset View (R)')
        rv.setToolTip('Reset View (Shortcut: R)')
        rv.clicked.connect(self.reset_view)
        zoom_row.addWidget(rv)
        
        # Zoom out
        zo = QtWidgets.QPushButton('−')
        zo.setToolTip('Zoom Out (Shortcut: Z −)')
        zo.setFixedSize(30, 30)
        zo.clicked.connect(self.on_zoom_out)
        zoom_row.addWidget(zo)
        
        # Label
        zl = QtWidgets.QLabel('Zoom (Z +/-)')
        zl.setAlignment(QtCore.Qt.AlignCenter)
        zoom_row.addWidget(zl)
        
        # Zoom in
        zi = QtWidgets.QPushButton('+')
        zi.setToolTip('Zoom In (Shortcut: Z +)')
        zi.setFixedSize(30, 30)
        zi.clicked.connect(self.on_zoom_in)
        zoom_row.addWidget(zi)
        
        # add a bit of spacing below if you like
        zoom_row.addStretch(1)

        ctrl.addLayout(zoom_row)
        
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl.addWidget(line2)
        
        ctrl.addWidget(QtWidgets.QLabel('Brush Size (B +/-):'))
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.brush_slider.setRange(1,200); self.brush_slider.setValue(int(self.brush_size*100)); self.brush_slider.valueChanged.connect(self.change_brush)
        ctrl.addWidget(self.brush_slider)
        ctrl.addWidget(QtWidgets.QLabel('Point Size (D +/-):'))
        self.point_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.point_slider.setRange(1,20); self.point_slider.setValue(self.point_size); self.point_slider.valueChanged.connect(self.change_point)
        ctrl.addWidget(self.point_slider)
        line3 = QtWidgets.QFrame()
        line3.setFrameShape(QtWidgets.QFrame.HLine)
        line3.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl.addWidget(line3)
        
        self.color_btn = QtWidgets.QPushButton('Pick Color'); self.color_btn.clicked.connect(self.pick_color)
        ctrl.addWidget(self.color_btn)
        swl = QtWidgets.QGridLayout(); ctrl.addLayout(swl)
        
        line4 = QtWidgets.QFrame()
        line4.setFrameShape(QtWidgets.QFrame.HLine)
        line4.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl.addWidget(line4)
        
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)           # gamma 0.1 to 3.0
        self.gamma_slider.setValue(100)               # gamma = 1.0 default
        self.gamma_slider.valueChanged.connect(self.on_gamma_change)
        
        gamma_row = QtWidgets.QHBoxLayout()

        gamma_label = QtWidgets.QLabel('Contrast (Gamma):')
        self.gamma_value_label = QtWidgets.QLabel('1.00')  # initial value

        gamma_row.addWidget(gamma_label)
        gamma_row.addWidget(self.gamma_slider)
        gamma_row.addWidget(self.gamma_value_label)

        ctrl.addLayout(gamma_row)

        
        contrast_buttons_row = QtWidgets.QHBoxLayout()

        self.reset_contrast_btn = QtWidgets.QPushButton('Reset Contrast')
        self.reset_contrast_btn.clicked.connect(self.reset_contrast)

        self.auto_contrast_btn = QtWidgets.QPushButton('Auto Contrast')
        self.auto_contrast_btn.setToolTip('Automatically stretch RGB range to fit contrast')
        self.auto_contrast_btn.clicked.connect(self.apply_auto_contrast)

        contrast_buttons_row.addWidget(self.reset_contrast_btn)
        contrast_buttons_row.addWidget(self.auto_contrast_btn)

        ctrl.addLayout(contrast_buttons_row)
        
        self.hist_btn = QtWidgets.QPushButton('Show RGB Histograms')
        self.hist_btn.clicked.connect(self.show_histograms)
        ctrl.addWidget(self.hist_btn)
        
        line5 = QtWidgets.QFrame()
        line5.setFrameShape(QtWidgets.QFrame.HLine)
        line5.setFrameShadow(QtWidgets.QFrame.Sunken)
        ctrl.addWidget(line5)


        cols = ['#FF0000','#00FF00','#0000FF','#FFFF00','#00FFFF','#FF00FF','#800000','#008000','#000080','#808000','#008080','#800080','#FFC0CB','#FFA500','#A52A2A','#5F9EA0','#D2691E','#9ACD32']
        self.swatches=[]
        for i,c in enumerate(cols): b=QtWidgets.QPushButton(); b.setFixedSize(20,20); b.setStyleSheet(f'background:{c};border:1px solid #333;'); b.clicked.connect(lambda _,col=c,btn=b: self.select_swatch(col,btn)); self.swatches.append(b); swl.addWidget(b,i//6,i%6)
        ctrl.addStretch()
        
        ur = QtWidgets.QHBoxLayout()
        bp = QtWidgets.QVBoxLayout(); ctrl.addLayout(bp)
        bp.addLayout(ur)

        self.eraser_btn = QtWidgets.QPushButton('Eraser (E)')
        self.eraser_btn.setToolTip('Eraser Tool — revert to original colors')
        self.eraser_btn.clicked.connect(self.activate_eraser)
        
        self.undo_btn = QtWidgets.QPushButton('Undo (Ctrl+Z)')
        self.redo_btn = QtWidgets.QPushButton('Redo (Ctrl+Y)')
        self.eraser_btn.setText('Eraser (E)')  # reuse existing button
        
        self.repair_btn = QtWidgets.QPushButton('Repair (Shift+R)')  # NEW
        self.repair_btn.setCheckable(True)                           # NEW
        self.repair_btn.toggled.connect(self.toggle_repair_mode)     # NEW

        self.undo_btn.clicked.connect(self.on_undo)
        self.redo_btn.clicked.connect(self.on_redo)

        ur.addWidget(self.undo_btn)
        ur.addWidget(self.redo_btn)
        ur.addWidget(self.eraser_btn)
        ur.addWidget(self.repair_btn)
        
        nav = QtWidgets.QHBoxLayout(); bp.addLayout(nav)
        self.prev_btn = QtWidgets.QPushButton('Previous'); self.next_btn = QtWidgets.QPushButton('Next'); self.prev_btn.clicked.connect(self.on_prev); self.next_btn.clicked.connect(self.on_next); nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn)
        self.save_btn = QtWidgets.QPushButton('Save (Ctrl+S)'); self.save_btn.clicked.connect(self.on_save); bp.addWidget(self.save_btn)
        for w in [self.annot_chk,self.view_combo,self.brush_slider,self.point_slider,self.color_btn]+self.swatches+[self.prev_btn,self.next_btn,self.undo_btn,self.redo_btn,self.save_btn,self.toggle_ann_chk,self.alpha_slider,self.repair_btn]: w.setEnabled(False)


    def _enable_controls(self):
        for w in [self.annot_chk,self.view_combo,self.brush_slider,self.point_slider,self.color_btn,self.eraser_btn,self.gamma_slider,self.reset_contrast_btn,self.auto_contrast_btn,self.hist_btn]+self.swatches+[self.prev_btn,self.next_btn,self.undo_btn,self.redo_btn,self.save_btn,self.toggle_ann_chk,self.open_ann_btn, self.open_orig_btn,self.alpha_slider,self.repair_btn]: w.setEnabled(True)

    def open_ann_folder(self):
        fol = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Annotation PC Folder')
        if not fol: return
        self.ann_dir = Path(fol)
        self.directory = self.ann_dir
        self.files = self._get_sorted_files()
        if not self.files:
            QtWidgets.QMessageBox.critical(self, 'Error', 'No PLY/PCD in Annotation folder'); return
        self.index, self.history, self.redo_stack = 0, [], []
        self._enable_controls()
        self.load_cloud()

    def open_orig_folder(self):
        fol = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Original PC Folder')
        if not fol: return
        self.orig_dir = Path(fol)
        # Persist immediately if you like
        self._save_state()
        # If an annotated file is already showing, reload so toggle works now
        if self.files:
            self.load_cloud()

    def load_cloud(self):
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

        # Set orientation ONCE; actual fit is deferred by _end_batch() → _finalize_layout()
        self.apply_view()
        if self.view_combo.currentText() == 'Top-Down':
            self.plotter_ref.view_xy()
        else:
            self.plotter_ref.view_isometric()

        # send explicit (x,y) to on_click
        self.plotter.track_click_position(lambda _, x, y: self.on_click(x, y))
        # update overlays
        total, curr = len(self.files), self.index + 1
        self.counter_label.setText(f'{curr}/{total}')
        self.counter_label.adjustSize()
        fn = Path(self.files[self.index]).name
        self.filename_label.setText(fn)
        self.filename_label.adjustSize()
        self._position_overlays()
        
        self.gamma_slider.setValue(100)
        self.enhanced_colors = self.original_colors.copy()

        self._session_edited = np.zeros(self.cloud.n_points, dtype=bool)
        has_any_edit_now = np.any(self.colors != self.original_colors)
        self.toggle_ann_chk.setEnabled(has_any_edit_now)
        
        self.annotations_visible = getattr(self, 'toggle_ann_chk', None) is None or self.toggle_ann_chk.isChecked()
        self.update_annotation_visibility()
        self._save_state()
        self._end_batch()

    def _save_state(self):
        json.dump({
            'annotation_dir': str(self.ann_dir or ''),
            'original_dir':   str(self.orig_dir or ''),
            'index':          self.index
        }, STATE_FILE.open('w'))

    def _position_overlays(self):
        # LEFT (annotated) overlays
        if hasattr(self, 'counter_label') and hasattr(self, 'plotter'):
            h1 = self.plotter.interactor.height()
            w1 = self.plotter.interactor.width()
            self.counter_label.move(10, h1 - self.counter_label.height() - 10)
            self.filename_label.move((w1 - self.filename_label.width()) // 2,
                                    h1 - self.filename_label.height() - 10)
            self.counter_label.raise_()
            self.filename_label.raise_()

        # RIGHT (original) label
        if hasattr(self, 'right_title') and hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
            h2 = self.plotter_ref.interactor.height()
            w2 = self.plotter_ref.interactor.width()
            self.right_title.adjustSize()
            self.right_title.move((w2 - self.right_title.width()) // 2,
                                h2 - self.right_title.height() - 10)
            self.right_title.raise_()

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

        QtWidgets.QApplication.processEvents()
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
            dirp = np.array([0, 0, -1]) if self.view_combo.currentText() == 'Top-Down' else np.array([1, 1, -1])
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
        if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
            plotter.render()

    def _fit_to_canvas(self):
        if getattr(self, '_is_closing', False) or getattr(self, '_batch', False):
            return
        # Fit left (annotated)
        if hasattr(self, 'plotter'):
            self._fit_view(self.plotter)
        # Fit right (original) when visible
        if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
            self._fit_view(self.plotter_ref)            

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'counter_label'):
            self._position_overlays()
        if not getattr(self, '_batch', False):
            self._schedule_fit()  # coalesce multiple resize fits

    def apply_view(self):
        """
        Top-Down  -> orthographic, look straight down +Z with +Y up.
        Isometric -> perspective, SOUTH-WEST isometric (from -X,-Y, +Z) with Z up.
        We set orientation explicitly for both panes, then defer the fit.
        """
        topdown = (self.view_combo.currentText() == 'Top-Down')

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
            else:
                # SOUTH-WEST isometric: from (-X, -Y, +Z) toward center
                cam.ParallelProjectionOff()
                cam.SetViewUp(0, 0, 1)             # Z up
                dop = np.array([1.0, 1.0, -1.0])   # direction-of-projection (to center)
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

        # Left (annotated)
        _apply(self.plotter, getattr(self, 'cloud', None), topdown)

        # Right (original)
        if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
            _apply(self.plotter_ref, getattr(self, 'cloud_ref', None), topdown)

        # Defer the actual fit so it uses final widget sizes
        self._schedule_fit()


    def reset_view(self):
        self.plotter.reset_camera()
        self.apply_view()

    def toggle_annotation(self):
        if self.annot_chk.isChecked():
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
        r_px  = max(1, int(self.brush_slider.value()))    # brush radius (px)
        ps_px = max(1, int(self.point_slider.value()))    # rendered point size (px)

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
        self.plotter.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))


    def change_brush(self, val):
        v = max(1, min(val, 200))
        self.brush_size = v / 100.0
        self.brush_slider.setValue(v)
        if self.annot_chk.isChecked():
            self.update_cursor()

    def change_point(self, val):
        """Update rendered point size and keep 'round points' sticky."""
        v = max(1, min(int(val), 20))
        self.point_size = v
        self.point_slider.setValue(v)

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
        if self.annot_chk.isChecked():
            self.update_cursor()


    def pick_color(self):
        c = QtWidgets.QColorDialog.getColor()
        if c.isValid():
            self.current_color = [c.red(), c.green(), c.blue()]

    def select_swatch(self, col, btn):
        for b in self.swatches:
            b.setStyleSheet(b.styleSheet().replace('2px solid yellow', '1px solid #333'))
        btn.setStyleSheet(btn.styleSheet().replace('1px solid #333', '2px solid yellow'))
        qc = QColor(col)
        self.current_color = [qc.red(), qc.green(), qc.blue()]

    def on_click(self, x, y):
        if not self.annot_chk.isChecked():
            return
        # 1) pick center
        picker = vtkPropPicker()
        h = self.plotter.interactor.height()
        picker.Pick(x, h - y, 0, self.plotter.renderer)
        pt = np.array(picker.GetPickPosition())
        if np.allclose(pt, (0,0,0)):
            return

        # 2) pick an edge pixel at (x  slider_px, y) to get world distance
        r_px = self.brush_slider.value()
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
        # apply new color
        if self.repair_mode:
            # Repair = copy ORIGINAL back into annotated layer
            self.colors[idx] = self.original_colors[idx]
        else:
            if self.current_color is None:
                self.colors[idx] = self.original_colors[idx]
            else:
                self.colors[idx] = self.current_color
        
        self._session_edited[idx] = True
        self.toggle_ann_chk.setEnabled(np.any(self._session_edited)) 
        
        # push update back into the plot
        self.update_annotation_visibility()


    def on_prev(self):
        if self.index > 0:
            self.index -= 1
            self.history.clear()
            self.redo_stack.clear()
            self.load_cloud()
            self._position_overlays()

    def on_next(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.history.clear()
            self.redo_stack.clear()
            self.load_cloud()
            self._position_overlays()

    def on_undo(self):
        if not self.history:
            return
        idx, old = self.history.pop()
        self.redo_stack.append((idx, self.colors[idx].copy()))
        self.colors[idx] = old
        self._session_edited[idx] = False
        self.toggle_ann_chk.setEnabled(np.any(self._session_edited))
        self.update_annotation_visibility()

    def on_redo(self):
        if not self.redo_stack:
            return
        idx, cols = self.redo_stack.pop()
        self.history.append((idx, self.colors[idx].copy()))
        self.colors[idx] = cols
        self._session_edited[idx] = True
        self.toggle_ann_chk.setEnabled(np.any(self._session_edited))
        self.update_annotation_visibility()

    def on_save(self):
        out = Path(self.files[self.index])
        ext = out.suffix.lower()

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

        QtWidgets.QMessageBox.information(
            self, 'Saved',
            f'Successfully saved {ext[1:]} file with colors to and reloaded:\n{out}'
        )

    def _on_plus(self):
        if self._waiting == 'brush':
            self.change_brush(self.brush_slider.value() + 2)
        elif self._waiting == 'point':
            self.change_point(self.point_slider.value() + 1)
        elif self._waiting == 'zoom':
            self.plotter.camera.Zoom(1.1)
            self.plotter.render()
        elif self._waiting == 'alpha':                                  # NEW
            self.alpha_slider.setValue(min(100, self.alpha_slider.value() + 5))  # NEW

    def _on_minus(self):
        if self._waiting == 'brush':
            self.change_brush(self.brush_slider.value() - 2)
        elif self._waiting == 'point':
            self.change_point(self.point_slider.value() - 1)
        elif self._waiting == 'zoom':
            self.plotter.camera.Zoom(0.9)
            self.plotter.render()
        elif self._waiting == 'alpha':                                  # NEW
            self.alpha_slider.setValue(max(0, self.alpha_slider.value() - 5))  # NEW

    def on_zoom_in(self):
        # behave exactly like “Z +”
        self._waiting = 'zoom'
        self._on_plus()

    def on_zoom_out(self):
        # behave exactly like “Z –”
        self._waiting = 'zoom'
        self._on_minus()
    
    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        
    def eventFilter(self, obj, event):
        if getattr(self, '_is_closing', False):
            return

        if obj is self.plotter.interactor and self.annot_chk.isChecked():
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

                r_px = max(1, self.brush_slider.value())
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
                            if self.repair_mode:
                                self.colors[idx] = self.original_colors[idx]
                            else:
                                if self.current_color is None:
                                    self.colors[idx] = self.original_colors[idx]
                                else:
                                    self.colors[idx] = self.current_color
                            self._session_edited[idx] = True
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
                        if self.repair_mode:
                            self.colors[idx] = self.original_colors[idx]
                        else:
                            if self.current_color is None:
                                self.colors[idx] = self.original_colors[idx]
                            else:
                                self.colors[idx] = self.current_color
                        self._session_edited[idx] = True
                        touched_idx.append(idx)
                        changed_any = True

                    self._last_paint_xy = (x2, y2)

                if changed_any:
                    self.toggle_ann_chk.setEnabled(np.any(self._session_edited))
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
    
    def activate_eraser(self):
        self.current_color = None  # Special flag for erasing        
        
    def reset_contrast(self):
        # Don't fire on_gamma_change while we move the slider
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(100)   # visual reset to gamma=1.0
        self.gamma_slider.blockSignals(False)
        self.gamma_value_label.setText("1.00")

        # "Untouched" points are ones that still equal the original colors
        # (self.colors tracks user edits only; we never wrote enhancements into it)
        current = self.colors.copy()
        untouched_mask = np.all(current == self.original_colors, axis=1)

        # Reset those untouched points back to original (i.e., remove any enhancement)
        current[untouched_mask] = self.original_colors[untouched_mask]

        # Keep book-keeping tidy
        self.enhanced_colors = self.original_colors.copy()

        # Push to mesh and re-render
        self.cloud['RGB'] = current
        self.update_annotation_visibility()
        
        # In repair mode, ORIGINAL view must stay raw (no contrast) — REPLACE/ENSURE
        if self.repair_mode and hasattr(self, 'cloud_ref'):
            self.cloud_ref['RGB'] = self.original_colors.astype(np.uint8)
            if not getattr(self, '_is_closing', False) and not getattr(self, '_batch', False):
                self.plotter_ref.render()


                
    def on_gamma_change(self, val):
        gamma = 2 ** ((val - 100) / 50.0)  # nonlinear mapping
        self.gamma_value_label.setText(f"{gamma:.2f}")

        original = self.original_colors.astype(np.float32) / 255.0

        min_vals = original.min(axis=0, keepdims=True)
        max_vals = original.max(axis=0, keepdims=True)
        stretched = (original - min_vals) / (max_vals - min_vals + 1e-5)

        corrected = np.power(stretched, gamma)
        self.enhanced_colors = (corrected * 255).astype(np.uint8)

        current = self.colors.copy()
        mask = np.all(current == self.original_colors, axis=1)
        current[mask] = self.enhanced_colors[mask]

        self.cloud['RGB'] = current  # ✅ ← important!
        self.update_annotation_visibility()
        
        # In repair mode, ORIGINAL view must stay raw (no contrast) — REPLACE/ENSURE
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
        self.gamma_value_label.setText("Auto")
        
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
        plt.show()
        
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

        self.plotter.update_scalars(display, mesh=self.cloud, render=True)

    def toggle_repair_mode(self, on: bool):
        self.repair_mode = bool(on)
        self.plotter_ref.setVisible(self.repair_mode)

        # Force Annotation Mode ON while repairing
        if self.repair_mode and not self.annot_chk.isChecked():
            self.annot_chk.setChecked(True)

        # Show/Hide right label
        if hasattr(self, 'right_title'):
            self.right_title.setVisible(self.repair_mode)

        # Refresh right scalars (original base)
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
        
    def closeEvent(self, e):
        self._is_closing = True  # block any future renders

        # stop watching events on the VTK interactor
        try:
            self.plotter.interactor.removeEventFilter(self)
        except Exception:
            pass

        # close the two QtInteractor windows safely
        for view in [getattr(self, 'plotter_ref', None), getattr(self, 'plotter', None)]:
            try:
                if view is not None:
                    view.close()
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
        
    def _begin_batch(self):                                # REPLACE with:
        self._batch = True
        self._cam_pause = True          # NEW: pause cam observer effects
        # While rebuilding a file, unlink cameras so view changes don’t echo back
        try:
            if self.repair_mode:
                self._unlink_cameras()  # NEW
        except Exception:
            pass
        
        self._cam_snap_l = self._snap_camera(self.plotter)         # NEW
        self._cam_snap_r = self._snap_camera(self.plotter_ref) if self.plotter_ref.isVisible() else None  # NEW

        for view in [getattr(self, 'plotter', None), getattr(self, 'plotter_ref', None)]:
            if view:
                view.interactor.setUpdatesEnabled(False)

    def _finalize_layout(self):
        if getattr(self, '_is_closing', False):
            return

        # Restore previous camera (prevents a blank/odd first frame)
        try:
            if hasattr(self, 'plotter'):
                self.plotter.render()
            if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
                self.plotter_ref.render()
        except Exception:
            pass

        # NOW enforce the chosen view (this overrides any stale orientation)
        self.apply_view()  # schedules the debounced fit internally

        # Overlays once sizes have settled
        self._position_overlays()

        # Quick render so the user sees something immediately
        if not getattr(self, '_startup', False):
            if hasattr(self, 'plotter'):
                self.plotter.render()
            if hasattr(self, 'plotter_ref') and self.plotter_ref.isVisible():
                self.plotter_ref.render()

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

        topdown = (self.view_combo.currentText() == 'Top-Down')

        if topdown:
            # Orthographic straight down
            cam.ParallelProjectionOn()
            cam.SetViewUp(0, 1, 0)
            dop = np.array([0.0, 0.0, -1.0])
            # ParallelScale ≈ half visible height
            scale_h = 0.5 * yr
            scale_w = 0.5 * xr / max(aspect, 1e-6)
            cam.SetParallelScale(max(scale_h, scale_w) * pad)
            pos = np.array([cx, cy, cz]) - dop * (r * 2.0 + 1.0)  # any positive distance
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(*pos)
        else:
            # South-West isometric (from -X,-Y,+Z)
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([1.0, 1.0, -1.0]); dop /= np.linalg.norm(dop)
            vfov = np.deg2rad(cam.GetViewAngle())
            hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect)
            eff = max(1e-3, min(vfov, hfov))
            dist = r / np.tan(eff / 2.0) * pad
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(*(np.array([cx, cy, cz]) - dop * dist))

    def _end_batch(self):                                  # REPLACE with:
        for view in [getattr(self, 'plotter', None), getattr(self, 'plotter_ref', None)]:
            if view:
                view.interactor.setUpdatesEnabled(True)
        # Re-link cameras AFTER fit to avoid bounce
        if self.repair_mode:
            self._link_cameras()            # NEW
        self._batch = False
        self._cam_pause = False             # NEW
        # Defer fit+render until the splitter/widget sizes settle this event loop
        QtCore.QTimer.singleShot(0, self._finalize_layout)  # NEW

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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Annotator()
    win.show()
    sys.exit(app.exec_())