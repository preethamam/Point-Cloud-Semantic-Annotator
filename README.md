# Point Cloud Annotator

Semantic color annotation and visualization tool for 3D point clouds (PLY / PCD). Provides a high‑precision, screen‑space aware brush, color + contrast enhancement utilities, and an efficient workflow for reviewing large collections. State (last folder & file) and visual contexts are preserved between sessions for continuity.

![overview](assets/overview_01.png)
![overview2](assets/overview_02.png)

---
## Table of Contents
1. Overview
2. Feature Matrix
3. Architecture & Internals
4. Installation
5. Quick Start
6. Detailed Usage
7. Controls & Shortcuts
8. Algorithms (Brush, Gamma, Auto Contrast, Histograms)
9. Data Model & File I/O
10. State & Persistence
11. Packaging as an Executable
12. Performance Tuning
13. Troubleshooting & FAQ
14. Extending & Customizing
15. Roadmap / Ideas
16. Contributing
17. License

---
## 1. Overview
The application focuses on interactive semantic recoloring of unstructured point clouds. Instead of polygonal segmentation, it lets you “paint” semantic classes (represented as RGB colors) rapidly. High‑rate feedback is achieved by combining VTK picking with a cKDTree spatial pre‑index. Contrast tools enhance visual discrimination without overwriting painted data unless explicitly requested.

---
## 2. Feature Matrix

| Category | Capabilities |
|----------|--------------|
| Formats | PLY (binary write), PCD (binary write), auto‑inject RGB channel if absent |
| Painting | Screen‑space circular brush, adjustable size, undo/redo, eraser, color swatches, custom color dialog |
| Navigation | Previous/Next file, natural sorting, view presets (Top‑Down / Isometric), zoom mode, reset camera |
| Display | Adjustable point size, persistent overlays (index & filename), magenta brush cursor |
| Color Enhancement | Gamma slider (nonlinear mapping), Auto Contrast (percentile stretch), RGB histogram (original vs enhanced) |
| Save Options | Conditional application of enhanced (unpainted) colors on save (prompt) |
| Persistence | Last folder + file index stored via `appdirs` platform path |
| Performance | cKDTree region queries, hybrid world + screen filter for precise brush footprint |

---
## 3. Architecture & Internals

High‑level components:
- UI Layer (PyQt5): Main window, controls panel, sliders, buttons, dialogs.
- Rendering Layer (PyVista / VTK): Point cloud actor, camera control, event integration.
- Annotation Engine: Maintains current color, brush size, stroke lifecycle, undo/redo stacks.
- Selection Subsystem: Combines VTK picking (projected point positions) with KD-tree radius queries and a final pixel-circle screen filter for accuracy.
- Enhancement Module: Gamma mapping, percentile stretch (Auto Contrast), histogram (KDE) visualization.
- Persistence Module: Lightweight JSON state file storing last dataset and index.

Stroke Flow:
1. Mouse press (left) in annotation mode ⇒ begin stroke; snapshot colors for undo.
2. Mouse move (while pressed) ⇒ compute indices, recolor incrementally.
3. Mouse release ⇒ push stroke diff (indices + previous colors) onto undo stack; clear redo stack.

Undo/Redo Model: Arrays of `(indices, previous_color_values)`; redo stores the overwritten colors when an undo occurs.

Data Structures:
- `self.cloud`: PyVista PolyData
- `self.colors`: Active working RGB (uint8)
- `self.original_colors`: Baseline from file load
- `self.enhanced_colors`: Contrast‑enhanced copy (only applied to untouched points)
- `self.kdtree`: cKDTree of point positions for candidate region queries

---
## 4. Installation

### 4.1 Requirements
- Python 3.8+ (tested with 3.10–3.12 recommended)
- OpenGL 2.1+ capable GPU & drivers (for VTK)
- Windows / Linux / macOS (primary development on Windows)

### 4.2 Dependencies (Runtime)
`pyvista`, `pyvistaqt`, `vtk`, `PyQt5`, `numpy`, `scipy`, `matplotlib`, `appdirs`

### 4.3 Recommended Dev Extras
`black`, `flake8`, `pyinstaller`, `auto-py-to-exe`

### 4.4 Install (Virtual Environment)
```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If no `requirements.txt` is present:
```pwsh
pip install pyvista pyvistaqt vtk PyQt5 numpy scipy matplotlib appdirs
```

---
## 5. Quick Start
```pwsh
python .\app.py
```
Click "Open Folder" → select directory with `.ply` or `.pcd` point clouds → enable Annotation Mode (A) → Paint.

---
## 6. Detailed Usage
1. Open Dataset: Loads and natural‑sorts all `*.ply` + `*.pcd` in selected folder.
2. Inspect: Choose Top‑Down for orthographic feel or Isometric for context.
3. Enable Annotation Mode: (A) toggles painting vs free camera navigation.
4. Choose Color: Swatch or dialog. Active color applied to brush; set Eraser (E) to restore original.
5. Adjust Brush: Slider or press B then +/−. Cursor ring updates live.
6. Paint: Click‑drag (left). Each stroke forms one undo unit.
7. Point Size: Adjust for density readability (1–20).
8. Contrast Handling:
	 - Gamma: Nonlinear scaling, only affecting unpainted points.
	 - Auto Contrast: Percentile stretch (2–98%) applied only to untouched points.
	 - Histograms: Compare original vs enhanced to verify adjustments.
9. Navigation: Previous / Next; overlays show progress count and filename.
10. Save: Ctrl+S → Prompt decides whether to bake enhanced colors into untouched points.

Best Practice: Apply contrast adjustments before extensive painting for consistency.

---
## 7. Controls & Shortcuts
| Action | Shortcut | Notes |
|--------|----------|-------|
| Toggle Annotation Mode | A | Enables brush & cursor |
| Brush Size | B then +/− | Pixel radius 1–200 |
| Point Size | D then +/− | Display only |
| Zoom | Z then +/− | Or UI buttons |
| Reset View | R | Re-applies preset |
| Eraser | E | Restores original colors |
| Undo / Redo | Ctrl+Z / Ctrl+Y | Stroke granularity |
| Save | Ctrl+S | Enhancement bake prompt |
| Previous / Next File | ← / → | Natural ordering |
| Pick Color Dialog | (Button) | QColorDialog |

---
## 8. Algorithms

### 8.1 Brush Selection
1. Center pick with `vtkPropPicker`.
2. Sample along circle to estimate equivalent world radius.
3. KD-tree query for candidate points.
4. Project candidates to screen; retain those inside pixel circle.

### 8.2 Gamma Mapping
`gamma = 2 ** ((slider_value - 100)/50)`; apply exponent after per-channel min/max normalization. Only untouched points updated.

### 8.3 Auto Contrast
Per-channel percentile stretch (2–98%). Untouched update policy identical to gamma.

### 8.4 Histograms
Gaussian KDE per channel for original and enhanced frames.

### 8.5 Undo/Redo
Stacks store (indices, prior_colors). Redo populated on undo; cleared on any new stroke.

---
## 9. Data Model & File I/O
- Working array: `RGB` uint8 (N×3).
- Missing `RGB`: synthesized (zeros) on load.
- Save:
	- PLY: `vtkPLYWriter` binary with `RGB`.
	- PCD: PyVista `.save(binary=True)`.
	- Enhancement prompt merges `enhanced_colors` into untouched points if confirmed.

---
## 10. State & Persistence
- JSON file using `appdirs.user_data_dir(APP_NAME)`.
- Keys: `directory`, `index`.
- Safe bounds check protects against removed files.

Remove the state file to reset startup behavior.

---
## 11. Packaging as an Executable

### 11.1 PyInstaller
```pwsh
pyinstaller --noconfirm --name PointCloudAnnotator --icon icon.png --add-data "icon.png;." --hidden-import vtkmodules.all app.py
```

### 11.2 auto-py-to-exe
Use GUI to configure the same options; include additional data/icon and hidden imports as needed.

### 11.3 Common Issues
- Missing Qt plugins → add plugin dir via `--add-data`.
- Large bundle → prune unused VTK IO modules; optionally UPX compress.

---
## 12. Performance Tuning
- Reduce brush size for massive clouds to limit candidate queries.
- Lower point size to mitigate overdraw in dense scenes.
- Downsample for annotation; propagate colors back with nearest neighbor mapping.
- Close other GPU intensive applications.

---
## 13. Troubleshooting & FAQ
| Issue | Cause | Resolution |
|-------|-------|------------|
| Blank window | OpenGL fallback | Update drivers / test simple VTK script |
| Slow painting | Very dense cloud | Downsample; ensure SciPy wheels optimized |
| Colors missing after save | File write blocked | Check permissions / disk space |
| Enhanced colors not applied to painted points | By design | Use Eraser to revert then reapply enhancement |
| Histogram window hidden | Behind main window | Alt+Tab or minimize main window temporarily |

### FAQ
Q: How to export class IDs?  
A: Future feature—current version encodes semantics as RGB only. Consider maintaining an external color→class mapping file.

Q: Can I change percentile stretch?  
A: Edit `p_low` / `p_high` values in `apply_auto_contrast()`.

Q: Add LAS/LAZ?  
A: Convert via PDAL or CloudCompare to PLY/PCD first.

---
## 14. Extending & Customizing
Hook points:
- Brush logic: `_compute_brush_idx`, painting blocks in `eventFilter`.
- Contrast: `on_gamma_change`, `apply_auto_contrast`, `reset_contrast`.
- Save pipeline: `on_save`.

Ideas:
- Class legend export (JSON/CSV).
- Multi-selection rectangle or lasso tool.
- Batch enhancement processing.
- Per-class mask export (int label array alongside RGB).

---
## 15. Roadmap / Ideas
- Export per-point integer labels.
- Batch non-interactive contrast mode.
- Palette recommendation system.
- File bookmarking & session notes.

---
## 16. Contributing
1. Fork repository & create a feature branch.
2. Keep PRs focused; include screenshots for UI changes.
3. Follow PEP8 (auto-format with `black`).
4. Describe performance implications of heavy loops.

Bug Report Template:
```
Environment:
	OS:
	Python:
	GPU/Driver:
Steps to Reproduce:
Expected:
Actual:
Sample Data (if possible):
```

---
## 17. License
MIT License © 2025 Preetham Manjunatha. See `LICENSE` for details.
