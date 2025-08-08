# Point Cloud Annotator

A desktop tool for semantic annotation of 3D point clouds (PLY/PCD). Paint colors directly onto points using a screen-space brush, adjust contrast, navigate large datasets, and save your annotations back to disk. The app remembers your last dataset and position for a smooth review workflow.

![overview](assets/overview_01.png)
![overview2](assets/overview_02.png)


## Features

### Point Cloud Format Support
- PLY (.ply)
- PCD (.pcd)
- Automatically adds an RGB channel when missing (initialized to black) so you can start painting any cloud.
- Binary PLY write using VTK for speed; PCD saved via PyVista (binary mode).

### Annotation (Painting) Tools
- Screen-space brush painting that respects perspective and depth:
  - Brush radius is defined in pixels; selection is computed by projecting candidate points to the screen for accurate circular strokes.
  - Robust selection using VTK picking + a KD-tree for fast nearest-neighbor queries.
- Adjustable brush size (1–200 px).
- Color picker (full RGB) and a quick-palette of swatches.
- Eraser tool to restore points to their original colors.
- Magenta circular cursor that matches the current brush size (shown in annotation mode).
- Point size control for display clarity (1–20 px).

### Editing Workflow
- Toggleable Annotation Mode so you can switch between 3D navigation and painting without conflicts.
- Undo/Redo with per-stroke granularity.
- Previous/Next file navigation across a folder of point clouds (with natural sorting: file_2 comes before file_10).
- Persistent window overlays:
  - Bottom-left: file index (e.g., 3/128)
  - Bottom-center: current filename

### View & Navigation
- Initial view presets: Top-Down and Isometric.
- Quick Zoom controls (+/−) with a dedicated Zoom mode to tie into keyboard shortcuts.
- Reset View returns the camera to a clean framing.
- Leverages PyVista/VTK interactor for rotate/pan/zoom when not painting.

### Contrast & Color Enhancement
- Gamma/contrast slider with a perceptual mapping for fine control.
- Auto Contrast stretches RGB channels using robust percentiles (2–98%)—applied only to unpainted points to preserve your annotations.
- RGB Histogram viewer (smoothed via KDE) shows original vs. enhanced distributions per channel.

### Smart Save Behavior
- On save, choose whether to bake in contrast-enhanced colors.
- If you select “Yes”, only untouched (unpainted) points are updated to the enhanced colors; painted points remain as you colored them.
- Saves directly over the current file using:
  - Binary PLY with named array `RGB` (uint8 Nx3)
  - Binary PCD via PyVista

### State Persistence
- Remembers last opened folder and currently selected file index across sessions.
- Stores lightweight state in a user data directory appropriate to your OS.
  - Windows example: `%APPDATA%/Point Cloud Annotator/state.json`

### Performance Notes
- Uses `scipy.spatial.cKDTree` for fast spatial lookups during brushing.
- Brush selection blends world- and screen-space techniques for accuracy and responsiveness.


## Installation

### Requirements
- Python 3.8+ (Windows 10/11 recommended)
- An OpenGL-capable GPU/driver (VTK requires OpenGL 2.1+)

### Dependencies
- PyVista (`pyvista`)
- PyVistaQt (`pyvistaqt`)
- VTK (`vtk`) – typically pulled in by PyVista, but listed explicitly here
- PyQt5 (`PyQt5`)
- NumPy (`numpy`)
- SciPy (`scipy`)
- Matplotlib (`matplotlib`)
- AppDirs (`appdirs`)

### Install Dependencies
Use a virtual environment (recommended), then install packages:

```pwsh
# (Optional) Create & activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
pip install pyvista pyvistaqt vtk PyQt5 numpy scipy matplotlib appdirs
```

If you encounter build/runtime issues with VTK/PyVista on Windows, ensure you’re on a recent Python (3.10–3.12) and update pip/setuptools/wheel:

```pwsh
python -m pip install --upgrade pip setuptools wheel
```


## How to Run

From this folder:

```pwsh
python .\app.py
```

Then click “Open Folder” and choose a directory containing `.ply` or `.pcd` point cloud files.


## Usage Guide

1. Open a folder with PLY/PCD files; files are naturally sorted and loaded into the viewer.
2. Choose an initial view (Top-Down or Isometric) if desired.
3. Toggle “Annotation Mode (A)” to paint; when off, you can freely rotate/pan/zoom via the 3D interactor.
4. Adjust brush size (slider or keyboard) and point size for display.
5. Pick a color from the dialog or click a swatch; a magenta circular cursor reflects brush radius in annotation mode.
6. Left-drag to paint; use the Eraser to restore original colors on painted points.
7. Optionally tweak contrast:
   - Gamma slider changes only unpainted points (live preview).
   - Auto Contrast stretches channels based on data percentiles (again, only on unpainted points).
   - View RGB histograms to compare original vs. enhanced distributions.
8. Navigate files with Previous/Next; overlays at the bottom show progress and the current filename.
9. Save (Ctrl+S). When prompted, choose whether to bake contrast-enhanced colors into untouched points before writing.


## Controls & Shortcuts

- Annotation Mode: A (toggle)
- Brush Size: B then +/−
- Point Size: D then +/−
- Zoom: Z then +/− (or the +/− buttons)
- Reset View: R
- Eraser: E (restores original colors when painting)
- Undo / Redo: Ctrl+Z / Ctrl+Y
- Save: Ctrl+S
- Previous / Next File: Left / Right Arrow keys

Notes:
- Painting only happens when Annotation Mode is enabled.
- Other interactions (rotate/pan/zoom) are handled by the VTK interactor when annotation mode is off.


## Data Model & Saving

- Colors are stored in an `RGB` array (uint8, shape N×3) on the point cloud.
- If an input cloud lacks `RGB`, the app adds one initialized to zeros (black).
- Save behavior:
  - PLY: Written as binary via VTK’s `vtkPLYWriter` with `RGB` as the array name.
  - PCD: Written via PyVista with `binary=True`.
  - A Yes/No dialog lets you decide if current contrast enhancement should be baked into untouched points.


## State & Config Files

- State file: remembers last folder and file index.
- Location (Windows): `%APPDATA%/Point Cloud Annotator/state.json`
- Created automatically on first successful open.


## Troubleshooting

- Black/blank render window or crash on startup:
  - Update your GPU drivers; VTK requires modern OpenGL.
  - Try a newer Python (3.10–3.12) and upgrade pip/setuptools/wheel.
- PyQt5 issues on Windows:
  - Ensure only one Qt binding is installed (avoid mixing PyQt and PySide).
- Missing `RGB` colors in saved PLY/PCD:
  - Ensure you saved from the app (Ctrl+S). The app writes `RGB` as uint8.
- Poor contrast in raw scans:
  - Use Auto Contrast to stretch channels; paint remains untouched.


## Files in This Folder

- `app.py` – Application source code.
- `app.ico`, `icon.png` – Window icons (optional; the app tries `icon.png` first, then `app.ico`).


## License

MIT License © 2025 Preetham Manjunatha. See `LICENSE` for details.
