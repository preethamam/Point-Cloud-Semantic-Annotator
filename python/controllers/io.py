from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PyQt5 import QtCore, QtWidgets
from scipy.spatial import cKDTree
from vtkmodules.vtkIOPLY import vtkPLYWriter


def natural_key(path):
    """Split filename into text/number chunks for natural sorting."""
    import re
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", path.name)]


def get_sorted_files(app):
    all_files = list(app.directory.glob("*.ply")) + list(app.directory.glob("*.pcd"))
    return sorted(all_files, key=natural_key)


def open_ann_folder(app) -> None:
    fol = QtWidgets.QFileDialog.getExistingDirectory(
        app, "Select Annotation PC Folder"
    )
    if not fol:
        return

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

    app.thumbs.prune_thumbnail_cache()

    app._populate_nav_list()

    app.load_cloud()

    if app.orig_dir is not None:
        QtCore.QTimer.singleShot(0, app._scan_annotated_files)


def open_orig_folder(app) -> None:
    fol = QtWidgets.QFileDialog.getExistingDirectory(app, "Select Original PC Folder")
    if not fol:
        return
    app.orig_dir = Path(fol)
    if app.files:
        app.load_cloud()


def load_cloud(app) -> None:
    app._visited.add(app.index)
    app._decorate_nav_item(app.index)

    pc = pv.read(str(app.files[app.index]))
    app._begin_batch()
    if "RGB" not in pc.array_names:
        pc["RGB"] = np.zeros((pc.n_points, 3), dtype=np.uint8)

    app.cloud = pc

    app.colors = pc["RGB"].copy()

    app.original_colors = app.colors.copy()
    if app.orig_dir:
        cand = app.orig_dir / Path(app.files[app.index]).name
        if cand.exists():
            pc0 = pv.read(str(cand))
            if "RGB" in pc0.array_names and pc0.n_points == pc.n_points:
                app.original_colors = pc0["RGB"].copy()

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
    app.act_toggle_annotations.setEnabled(bool(has_any_edit_now))

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
            app.toggle_ann_chk.setEnabled(bool(np.any(app._session_edited)))
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
