from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtWidgets
from scipy.spatial import cKDTree
from vtkmodules.vtkIOPLY import vtkPLYWriter

from services.storage import load_state, log_gui, save_state


def natural_key(path):
    """Split filename into text/number chunks for natural sorting."""
    import re
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", path.name)]


def get_sorted_files(app):
    all_files = list(app.directory.glob("*.ply")) + list(app.directory.glob("*.pcd"))
    return sorted(all_files, key=natural_key)


def _project_pairs_for(app) -> dict:
    try:
        st = load_state()
        pairs = st.get("project_pairs", {})
    except Exception:
        pairs = {}
    if not isinstance(pairs, dict):
        pairs = {}
    if app.ann_dir and app.orig_dir:
        pairs[str(app.ann_dir)] = str(app.orig_dir)
    return pairs


def open_ann_folder(app) -> None:
    try:
        st = load_state()
        start_dir = st.get("last_ann_dir", "") or str(app.ann_dir or "")
    except Exception:
        start_dir = str(app.ann_dir or "")
    fol = QtWidgets.QFileDialog.getExistingDirectory(
        app, "Select Annotation PC Folder", start_dir
    )
    if not fol:
        return
    log_gui(f"open_ann_folder: selected={fol}")

    app.ann_dir = Path(fol)
    app.directory = app.ann_dir
    app.files = app._get_sorted_files()

    if not app.files:
        QtWidgets.QMessageBox.critical(
            app, "Error", "No PLY/PCD in Annotation folder"
        )
        return

    app.index = 0
    app.history.clear()
    app.redo_stack.clear()
    app._visited.clear()
    app._annotated.clear()
    app._dirty.clear()

    if app._pending_orig_dir is not None:
        cand = app._pending_orig_dir
        has_full_match = all((cand / p.name).exists() for p in app.files)
        app.orig_dir = cand if has_full_match else None
        app._pending_orig_dir = None
    else:
        try:
            st = load_state()
            pairs = st.get("project_pairs", {})
        except Exception:
            pairs = {}
        if not isinstance(pairs, dict):
            pairs = {}
        cand = pairs.get(str(app.ann_dir), "")
        app.orig_dir = Path(cand) if cand else None
        if app.orig_dir is not None:
            has_full_match = all((app.orig_dir / p.name).exists() for p in app.files)
            if not has_full_match:
                log_gui(f"open_ann_folder: orig_dir cleared (mismatch) orig_dir={app.orig_dir}")
                app.orig_dir = None

    app.thumbs.new_generation()
    app.thumbs.prune_ann_thumbs()
    app._populate_nav_list()

    app.load_cloud()

    # Do not auto-scan annotation state on load; only mark after session saves.

    app._expecting_ann = False
    if hasattr(app, "act_open_orig"):
        app.act_open_orig.setEnabled(True)
    if hasattr(app, "act_open_ann"):
        app.act_open_ann.setEnabled(False)

    save_state({
        "annotation_dir": str(app.ann_dir or ""),
        "original_dir": str(app.orig_dir or ""),
        "index": int(app.index),
        "nav_dock_width": int(app.nav_dock.width()),
        "last_ann_dir": str(app.ann_dir or ""),
        "project_pairs": _project_pairs_for(app),
    })
    log_gui(f"open_ann_folder: files={len(app.files)} orig_dir={app.orig_dir}")


def open_orig_folder(app) -> None:
    try:
        st = load_state()
        start_dir = st.get("last_orig_dir", "") or str(app.orig_dir or "")
    except Exception:
        start_dir = str(app.orig_dir or "")
    fol = QtWidgets.QFileDialog.getExistingDirectory(app, "Select Original PC Folder", start_dir)
    if not fol:
        return
    log_gui(f"open_orig_folder: selected={fol}")
    cand = Path(fol)
    log_gui(f"open_orig_folder: clearing ann_dir ann_dir={app.ann_dir}")
    app.ann_dir = None
    app.directory = None
    app.files = []
    app.index = 0
    app.history.clear()
    app.redo_stack.clear()
    app._visited.clear()
    app._annotated.clear()
    app._dirty.clear()

    app.orig_dir = cand
    app._pending_orig_dir = cand

    app.thumbs.new_generation()
    app._populate_nav_list()
    app._update_status_bar()

    if hasattr(app, "act_open_orig"):
        app.act_open_orig.setEnabled(False)

    save_state({
        "annotation_dir": str(app.ann_dir or ""),
        "original_dir": str(app.orig_dir or ""),
        "index": int(app.index),
        "nav_dock_width": int(app.nav_dock.width()),
        "last_orig_dir": str(app.orig_dir or ""),
        "project_pairs": _project_pairs_for(app),
    })
    log_gui(f"open_orig_folder: files={len(app.files)} orig_dir={app.orig_dir}")

    app._expecting_ann = True
    if hasattr(app, "act_open_orig"):
        app.act_open_orig.setEnabled(False)
    if hasattr(app, "act_open_ann"):
        app.act_open_ann.setEnabled(True)


def refresh_folders(app, *, reload: bool = True, show_message: bool = True) -> bool:
    if not app.ann_dir and not app.orig_dir:
        if show_message:
            QtWidgets.QMessageBox.information(
                app, "Refresh Folders", "No annotation or original folder selected."
            )
        return False

    if app.ann_dir:
        app.directory = app.ann_dir
        app.files = app._get_sorted_files()
    else:
        app.directory = None
        app.files = []

    if not app.files:
        app.index = 0
        app.history.clear()
        app.redo_stack.clear()
        app._visited.clear()
        app._annotated.clear()
        app._dirty.clear()
        app.thumbs.new_generation()
        app.thumbs.prune_ann_thumbs()
        app._populate_nav_list()
        app._update_status_bar()
        if show_message:
            QtWidgets.QMessageBox.information(
                app, "Refresh Folders", "No PLY/PCD files in the annotation folder."
            )
        return False

    if app.orig_dir is not None:
        has_full_match = all((app.orig_dir / p.name).exists() for p in app.files)
        if not has_full_match:
            log_gui(f"refresh_folders: orig_dir cleared (mismatch) orig_dir={app.orig_dir}")
            app.orig_dir = None

    if app.index < 0 or app.index >= len(app.files):
        app.index = 0

    app.thumbs.new_generation()
    app.thumbs.prune_ann_thumbs()
    app._populate_nav_list()
    app._sync_nav_selection()
    app._update_status_bar()

    if reload:
        app.history.clear()
        app.redo_stack.clear()
        app.load_cloud()
        app._position_overlays()

    save_state({
        "annotation_dir": str(app.ann_dir or ""),
        "original_dir": str(app.orig_dir or ""),
        "index": int(app.index),
        "nav_dock_width": int(app.nav_dock.width()),
        "last_ann_dir": str(app.ann_dir or ""),
        "project_pairs": _project_pairs_for(app),
    })

    log_gui(f"refresh_folders: files={len(app.files)} orig_dir={app.orig_dir}")
    return True


def load_cloud(app) -> None:
    if not app.files or app.index < 0 or app.index >= len(app.files):
        ready = refresh_folders(app, reload=False, show_message=False)
        if not ready or not app.files:
            return
        if app.index < 0 or app.index >= len(app.files):
            return

    def _read_cloud(path: Path) -> pv.PolyData:
        pc_local = pv.read(str(path))
        if getattr(pc_local, "n_points", 0) <= 0:
            raise ValueError("Point cloud contains no points.")
        return pc_local

    start_index = app.index
    pc = None
    last_error = None
    for _ in range(len(app.files)):
        path = app.files[app.index]
        try:
            pc = _read_cloud(path)
            if "RGB" not in pc.array_names:
                pc["RGB"] = np.zeros((pc.n_points, 3), dtype=np.uint8)
            break
        except Exception as exc:
            last_error = exc
            log_gui(f"load_cloud: failed index={app.index} path={path} err={exc}")
            QtWidgets.QMessageBox.warning(
                app,
                "Load Error",
                f"Failed to load point cloud:\n{path}\n\n{exc}",
            )
            if getattr(app, "_bad_files", None) is None:
                app._bad_files = set()
            app._bad_files.add(app.index)
            app.index = (app.index + 1) % len(app.files)
            if app.index == start_index:
                break

    if pc is None:
        msg = "No valid point clouds could be loaded."
        if last_error is not None:
            msg = f"{msg}\n\nLast error:\n{last_error}"
        QtWidgets.QMessageBox.information(app, "Load Error", msg)
        return

    if app.index != start_index:
        app._sync_nav_selection()

    app._visited.add(app.index)
    app._decorate_nav_item(app.index)

    app._begin_batch()

    app.cloud = pc

    app.colors = pc["RGB"].copy()

    app.original_colors = app.colors.copy()
    if app.orig_dir:
        cand = app.orig_dir / Path(app.files[app.index]).name
        if cand.exists():
            try:
                pc0 = pv.read(str(cand))
                if "RGB" in pc0.array_names and pc0.n_points == pc.n_points:
                    app.original_colors = pc0["RGB"].copy()
            except Exception as exc:
                log_gui(f"load_cloud: failed orig read path={cand} err={exc}")

    app.kdtree = cKDTree(pc.points)

    app.enhanced_colors = app.original_colors.copy()

    app.cloud["RGB"] = app.enhanced_colors.astype(np.uint8)

    app._pre_fit_camera(app.cloud, app.plotter)
    app.plotter.clear()
    app.actor = app.plotter.add_points(
        app.cloud,
        scalars="RGB",
        rgb=True,
        point_size=app.point_size,
        reset_camera=False,
        render_points_as_spheres=True,
    )

    app.plotter_ref.clear()

    ref_colors = app.original_colors.astype(np.uint8)
    app.cloud_ref = pv.PolyData(app.cloud.points.copy())
    app.cloud_ref["RGB"] = ref_colors

    app.actor_ref = app.plotter_ref.add_points(
        app.cloud_ref,
        scalars="RGB",
        rgb=True,
        point_size=app.point_size,
        reset_camera=False,
        render_points_as_spheres=True,
    )

    app._pre_fit_camera(app.cloud, app.plotter)

    if app._shared_cam_active():
        app.plotter_ref.renderer.SetActiveCamera(app.plotter.renderer.GetActiveCamera())
        app._shared_camera = app.plotter.renderer.GetActiveCamera()
        app._need_split_fit = True
    else:
        app._pre_fit_camera(app.cloud_ref, app.plotter_ref)

    app.plotter.track_click_position(lambda pos: app.on_click(pos[0], pos[1]))

    app._position_overlays()

    app._last_gamma_value = 100
    app.on_gamma_change(100)
    app.enhanced_colors = app.original_colors.copy()

    app._session_edited = np.zeros(app.cloud.n_points, dtype=bool)
    has_any_edit_now = np.any(app.colors != app.original_colors)
    app.act_toggle_annotations.setEnabled(True)

    app.annotations_visible = getattr(app, "act_toggle_annotations", None) is None or app.act_toggle_annotations.isChecked()
    app.update_annotation_visibility()
    app._end_batch()
    app._update_status_bar()


def on_save(app, _autosave: bool = False) -> None:
    out = Path(app.files[app.index])
    ext = out.suffix.lower()

    if _autosave:
        choice = QtWidgets.QMessageBox.Yes
    else:
        choice = QtWidgets.QMessageBox.question(
            app, "Save Options",
            "Save with contrast-enhanced colors (Gamma adjusted)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )

    save_colors = app.colors.copy()

    if choice == QtWidgets.QMessageBox.Yes:
        untouched_mask = ~np.any(save_colors != app.original_colors, axis=1)
        save_colors[untouched_mask] = app.enhanced_colors[untouched_mask]

    app.cloud["RGB"] = save_colors

    if ext == ".ply":
        writer = vtkPLYWriter()
        writer.SetFileName(str(out))
        writer.SetInputData(app.cloud)
        writer.SetArrayName("RGB")
        writer.SetFileTypeToBinary()
        writer.Write()
    elif ext == ".pcd":
        app.cloud.save(str(out), binary=True)
    else:
        app.cloud.save(str(out))

    try:
        app._session_edited = np.zeros(app.cloud.n_points, dtype=bool)
        if hasattr(app, "toggle_ann_chk"):
            app.toggle_ann_chk.setEnabled(True)
    except Exception:
        pass

    if not _autosave:
        QtWidgets.QMessageBox.information(
            app, "Saved",
            f"Successfully saved {ext[1:]} file with colors to and reloaded:\n{out}",
        )

    app._dirty.discard(app.index)
    app._annotated.add(app.index)
    app._decorate_nav_item(app.index)
    app._update_status_bar()
