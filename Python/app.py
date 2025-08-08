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
  python point_cloud_annotator.py
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
from vtkmodules.vtkRenderingCore import vtkPropPicker

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
        self.brush_size = 0.05
        self.point_size = 3
        self.current_color = [255,0,0]
        self.history, self.redo_stack = [], []
        self._waiting = None
        self.directory, self.files, self.index = None, [], 0
        self._last_gamma_value = 100  # default gamma = 1.0

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
        s(QKeySequence(QtCore.Qt.Key_A), self, activated=lambda: self.annot_chk.toggle())
        s(QKeySequence('R'), self, activated=self.reset_view)
        s(QKeySequence.Undo, self, activated=self.on_undo)
        s(QKeySequence.Redo, self, activated=self.on_redo)
        s(QKeySequence(QtCore.Qt.Key_Left), self, activated=self.on_prev)
        s(QKeySequence(QtCore.Qt.Key_Right), self, activated=self.on_next)
        s(QKeySequence.Save, self, activated=self.on_save)
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
        
        # Restore state
        if STATE_FILE.exists():
            try:
                st = json.loads(STATE_FILE.read_text())
                self.directory = Path(st.get('directory',''))
                # grab all point-cloud files and do one natural sort
                self.files = self._get_sorted_files()
                self.index = max(0, min(len(self.files)-1, st.get('index',0)))
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
        Return indices of all points whose 2D screen projection
        lies within r_px pixels of (x,y), skipping voids.
        """
        picker = vtkPropPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.actor)

        ren  = self.plotter.renderer
        h    = self.plotter.interactor.height()
        r_px = self.brush_slider.value()

        # 1) pick center; abort if no hit
        if not picker.Pick(x, h - y, 0, ren):
            return []
        pt_center = np.array(picker.GetPickPosition())

        # 2) sample 8 edge directions to estimate world-space radius
        world_radii = []
        # sample at one step per pixel on your brush circumference, 
        # but at least 32 samples for small brushes:
        r_px = self.brush_slider.value()
        n_samples = min(360, max(32, int(2 * math.pi * r_px))) #max(32, int(2 * math.pi * r_px))

        for angle in np.linspace(0, 2*math.pi, n_samples, endpoint=False):
            dx = r_px * math.cos(angle)
            dy = r_px * math.sin(angle)
            if picker.Pick(int(x + dx), int(h - (y + dy)), 0, ren):
                pt_edge = np.array(picker.GetPickPosition())
                world_radii.append(np.linalg.norm(pt_edge - pt_center))

        # if nothing at any edge, fall back to brush_size in world units
        world_r = max(world_radii) if world_radii else self.brush_size

        # 3) KD-tree superset
        candidates = self.kdtree.query_ball_point(pt_center, world_r)
        if not candidates:
            return []

        # 4) final screen‐space filter
        valid = []
        for i in candidates:
            wx, wy, wz = self.cloud.points[i]
            ren.SetWorldPoint(wx, wy, wz, 1.0)
            ren.WorldToDisplay()
            dx_disp, dy_disp, _ = ren.GetDisplayPoint()
            # Qt y → VTK display y
            dy_vtk = h - y
            if (dx_disp - x)**2 + (dy_disp - dy_vtk)**2 <= r_px**2:
                valid.append(i)

        return valid

    def _build_ui(self):
        w = QtWidgets.QWidget(); self.setCentralWidget(w)
        lay = QtWidgets.QHBoxLayout(w)
        # 3D viewport
        self.plotter = QtInteractor(self)
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
        self.open_btn = QtWidgets.QPushButton('Open Folder'); self.open_btn.clicked.connect(self.open_folder)
        ctrl.addWidget(self.open_btn)
        self.annot_chk = QtWidgets.QCheckBox('Annotation Mode (A)'); self.annot_chk.stateChanged.connect(self.toggle_annotation)
        ctrl.addWidget(self.annot_chk)
        
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

        self.undo_btn.clicked.connect(self.on_undo)
        self.redo_btn.clicked.connect(self.on_redo)

        ur.addWidget(self.undo_btn)
        ur.addWidget(self.redo_btn)
        ur.addWidget(self.eraser_btn)
        nav = QtWidgets.QHBoxLayout(); bp.addLayout(nav)
        self.prev_btn = QtWidgets.QPushButton('Previous'); self.next_btn = QtWidgets.QPushButton('Next'); self.prev_btn.clicked.connect(self.on_prev); self.next_btn.clicked.connect(self.on_next); nav.addWidget(self.prev_btn); nav.addWidget(self.next_btn)
        self.save_btn = QtWidgets.QPushButton('Save (Ctrl+S)'); self.save_btn.clicked.connect(self.on_save); bp.addWidget(self.save_btn)
        for w in [self.annot_chk,self.view_combo,self.brush_slider,self.point_slider,self.color_btn]+self.swatches+[self.prev_btn,self.next_btn,self.undo_btn,self.redo_btn,self.save_btn]: w.setEnabled(False)
        

    def _enable_controls(self):
        for w in [self.annot_chk,self.view_combo,self.brush_slider,self.point_slider,self.color_btn,self.eraser_btn,self.gamma_slider,self.reset_contrast_btn,self.auto_contrast_btn,self.hist_btn]+self.swatches+[self.prev_btn,self.next_btn,self.undo_btn,self.redo_btn,self.save_btn]: w.setEnabled(True)

    def open_folder(self):
        fol = QtWidgets.QFileDialog.getExistingDirectory(self,'Select Folder')
        if not fol: return
        self.directory = Path(fol)
        # grab all point-cloud files and do one natural sort
        self.files = self._get_sorted_files()
        if not self.files:
            QtWidgets.QMessageBox.critical(self,'Error','No PLY/PCD'); return
        self.index, self.history, self.redo_stack = 0, [], []
        self._enable_controls()
        self.load_cloud()

    def load_cloud(self):
        pc = pv.read(str(self.files[self.index]))
        if 'RGB' not in pc.array_names: pc['RGB'] = np.zeros((pc.n_points,3),dtype=np.uint8)
        self.cloud, self.colors, self.kdtree = pc, pc['RGB'], cKDTree(pc.points)
        self.original_colors = self.colors.copy()
        self.enhanced_colors = self.colors.copy()
        self.plotter.clear()
        self.actor = self.plotter.add_points(self.cloud,scalars='RGB',rgb=True,point_size=self.point_size)
        self.apply_view()
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

        current = self.colors.copy()
        mask = np.all(current == self.original_colors, axis=1)
        current[mask] = self.enhanced_colors[mask]
        self.plotter.update_scalars(current, mesh=self.cloud, render=True)

        json.dump({'directory':str(self.directory),'index':self.index}, STATE_FILE.open('w'))

    def _position_overlays(self):
        h, w = self.plotter.interactor.height(), self.plotter.interactor.width()
        # bottom-left counter
        self.counter_label.move(10, h - self.counter_label.height() - 10)
        # bottom-center filename
        self.filename_label.move((w - self.filename_label.width()) // 2, h - self.filename_label.height() - 10)
        self.counter_label.raise_()
        self.filename_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # always keep overlays at bottom after resize
        if hasattr(self, 'counter_label'):
            self._position_overlays()

    def apply_view(self):
        if self.view_combo.currentText() == 'Top-Down':
            self.plotter.view_xy()
        else:
            self.plotter.view_isometric()

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
        r = self.brush_slider.value()
        d = r * 2
        pix = QPixmap(d + 4, d + 4)
        pix.fill(QtCore.Qt.transparent)
        p = QPainter(pix)
        pen = p.pen()
        pen.setColor(QColor(255, 0, 255))
        pen.setWidth(3)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(2, 2, d, d)
        p.end()
        self.plotter.interactor.setCursor(QCursor(pix, r + 2, r + 2))

    def change_brush(self, val):
        v = max(1, min(val, 200))
        self.brush_size = v / 100.0
        self.brush_slider.setValue(v)
        if self.annot_chk.isChecked():
            self.update_cursor()

    def change_point(self, val):
        v = max(1, min(val, 20))
        self.point_size = v
        self.point_slider.setValue(v)
        if hasattr(self, 'actor'):
            self.actor.GetProperty().SetPointSize(v)
            self.plotter.render()

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
        if self.current_color is None:
            self.colors[idx] = self.original_colors[idx]
        else:
            self.colors[idx] = self.current_color
        # push update back into the plot
        self.plotter.update_scalars(self.colors, mesh=self.cloud, render=True)


    def on_prev(self):
        if self.index > 0:
            self.index -= 1
            self.history.clear()
            self.redo_stack.clear()
            self.load_cloud()

    def on_next(self):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.history.clear()
            self.redo_stack.clear()
            self.load_cloud()

    def on_undo(self):
        if not self.history:
            return
        idx, old = self.history.pop()
        self.redo_stack.append((idx, self.colors[idx].copy()))
        self.colors[idx] = old
        self.plotter.update_scalars(self.colors, mesh=self.cloud, render=True)

    def on_redo(self):
        if not self.redo_stack:
            return
        idx, cols = self.redo_stack.pop()
        self.history.append((idx, self.colors[idx].copy()))
        self.colors[idx] = cols
        self.plotter.update_scalars(self.colors, mesh=self.cloud, render=True)

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

    def _on_minus(self):
        if self._waiting == 'brush':
            self.change_brush(self.brush_slider.value() - 2)
        elif self._waiting == 'point':
            self.change_point(self.point_slider.value() - 1)
        elif self._waiting == 'zoom':
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
    
    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        
    def eventFilter(self, obj, event):
        if obj is self.plotter.interactor and self.annot_chk.isChecked():
            # 1) start stroke on LeftButtonPress
            if event.type() == QtCore.QEvent.MouseButtonPress \
            and event.button() == QtCore.Qt.LeftButton:
                self._stroke_active        = True
                self._stroke_idxs.clear()
                self._colors_before_stroke = self.colors.copy()
                return True     # consume this event

            # 2) paint on LeftButton + Move
            if self._stroke_active \
            and event.type() == QtCore.QEvent.MouseMove \
            and (event.buttons() & QtCore.Qt.LeftButton):
                x, y = event.x(), event.y()
                idx   = self._compute_brush_idx(x, y)
                if idx:
                    self._stroke_idxs.update(idx)
                    if self.current_color is None:
                        self.colors[idx] = self.original_colors[idx]
                    else:
                        self.colors[idx] = self.current_color
                    self.plotter.update_scalars(self.colors, mesh=self.cloud, render=True)
                return True     # consume

            # 3) finish on LeftButtonRelease
            if self._stroke_active \
            and event.type() == QtCore.QEvent.MouseButtonRelease \
            and event.button() == QtCore.Qt.LeftButton:
                self._stroke_active = False
                if self._stroke_idxs:
                    idxs = list(self._stroke_idxs)
                    old  = self._colors_before_stroke[idxs]
                    self.history.append((idxs, old))
                    self.redo_stack.clear()
                self._colors_before_stroke = None
                return True     # consume

        # all other events (including two-finger pan, right-drag, etc.) go to QtInteractor
        return super().eventFilter(obj, event)
    
    def activate_eraser(self):
        self.current_color = None  # Special flag for erasing        
        
    def reset_contrast(self):
        self.gamma_slider.setValue(100)

        # Recompute gamma = 1.0 directly (to override auto-contrast)
        normalized = self.original_colors.astype(np.float32) / 255.0
        stretched = (normalized - normalized.min(axis=0, keepdims=True)) / (
            normalized.max(axis=0, keepdims=True) - normalized.min(axis=0, keepdims=True) + 1e-5
        )
        self.enhanced_colors = (stretched * 255).astype(np.uint8)

        current = self.colors.copy()
        mask = np.all(current == self.original_colors, axis=1)
        current[mask] = self.enhanced_colors[mask]

        self.cloud['RGB'] = current
        self.plotter.update_scalars('RGB', mesh=self.cloud, render=True)

        self.gamma_value_label.setText("1.00")

                
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
        self.plotter.update_scalars('RGB', mesh=self.cloud, render=True)

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
        self.plotter.update_scalars('RGB', mesh=self.cloud, render=True)

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



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Annotator()
    win.show()
    sys.exit(app.exec_())