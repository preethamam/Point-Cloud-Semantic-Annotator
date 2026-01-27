from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QCursor, QIcon, QPainter, QPixmap
from scipy.stats import gaussian_kde
from vtkmodules.vtkRenderingCore import vtkPropPicker
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from controllers import app_helpers


def toggle_annotation(app) -> None:
    if app.act_annotation_mode.isChecked():
        app.update_cursor()
    else:
        app.plotter.interactor.unsetCursor()


def compute_brush_idx(app, x, y):
    """
    Exact WYSIWYG: points are rendered as round sprites (radius s_px).
    Paint a point only if its sprite fits fully inside the brush circle:
        ||center - cursor|| <= r_px - s_px
    Fallback to circle-circle intersection when r_px <= s_px (tiny brushes).
    """
    if not hasattr(app, "kdtree") or app.kdtree is None or not hasattr(app, "actor"):
        return []

    ren = app.plotter.renderer
    inter = app.plotter.interactor
    H = inter.height()

    picker = vtkPropPicker()
    if not picker.Pick(x, H - y, 0, ren):
        return []
    wc = np.array(picker.GetPickPosition(), dtype=float)
    if not np.isfinite(wc).all():
        return []

    ren.SetWorldPoint(wc[0], wc[1], wc[2], 1.0)
    ren.WorldToDisplay()
    xd, yd, zd = ren.GetDisplayPoint()

    ren.SetDisplayPoint(xd + 1.0, yd, zd)
    ren.DisplayToWorld()
    wx1, wy1, wz1, _ = ren.GetWorldPoint()
    ren.SetDisplayPoint(xd, yd + 1.0, zd)
    ren.DisplayToWorld()
    wx2, wy2, wz2, _ = ren.GetWorldPoint()
    px_world = max(
        float(np.linalg.norm(np.array([wx1, wy1, wz1]) - wc)),
        float(np.linalg.norm(np.array([wx2, wy2, wz2]) - wc)),
    )

    r_px = float(max(1, app.brush_size))
    s_px = 0.5 * float(max(1, app.point_size))

    inflate = float(getattr(app, "_brush_coverage", 1.15))
    world_r = max(1e-9, (r_px + s_px) * px_world * inflate)
    cand = app.kdtree.query_ball_point(wc, world_r)
    if not cand:
        return []

    cx, cy = float(x), float(H - y)
    keep = []
    SetWorldPoint = ren.SetWorldPoint
    WorldToDisplay = ren.WorldToDisplay
    GetDisplayPoint = ren.GetDisplayPoint
    pts = app.cloud.points

    r_in = r_px - s_px
    if r_in > 0.5:
        r2_in = r_in * r_in
        for i in cand:
            wx, wy, wz = pts[i]
            SetWorldPoint(wx, wy, wz, 1.0)
            WorldToDisplay()
            dx, dy, _ = GetDisplayPoint()
            if (dx - cx) * (dx - cx) + (dy - cy) * (dy - cy) <= r2_in:
                keep.append(i)
    else:
        r2_sum = (r_px + s_px) * (r_px + s_px)
        for i in cand:
            wx, wy, wz = pts[i]
            SetWorldPoint(wx, wy, wz, 1.0)
            WorldToDisplay()
            dx, dy, _ = GetDisplayPoint()
            if (dx - cx) * (dx - cx) + (dy - cy) * (dy - cy) <= r2_sum:
                keep.append(i)

    return keep


def update_cursor(app) -> None:
    """
    Cursor ring shows the exact paint footprint when using the strict brush:
    effective radius = brush_radius_px - 0.5 * point_size_px.
    """
    r_px = max(1, int(app.brush_size))
    ps_px = max(1, int(app.point_size))

    r_eff = int(round(max(1.0, r_px - 0.5 * ps_px)))
    d = 2 * r_eff

    pix = QPixmap(d + 4, d + 4)
    pix.fill(QtCore.Qt.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    pen = p.pen()
    pen.setColor(QColor(255, 0, 255))
    pen.setWidth(2)
    p.setPen(pen)
    p.setBrush(QtCore.Qt.NoBrush)
    p.drawEllipse(2, 2, d, d)
    p.end()

    if app.clone_mode:
        app.plotter_ref.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))
        app.plotter.interactor.unsetCursor()
    elif app.repair_mode:
        app.plotter.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))
        app.plotter_ref.interactor.unsetCursor()
    else:
        app.plotter.interactor.setCursor(QCursor(pix, r_eff + 2, r_eff + 2))
        app.plotter_ref.interactor.unsetCursor()


def change_brush(app, val) -> None:
    v = int(max(1, min(val, 200)))
    app.brush_size = float(v)
    if hasattr(app, "ribbon_sliders") and "brush" in app.ribbon_sliders:
        _, lbl = app.ribbon_sliders["brush"]
        lbl.setText(f"{v} px")
    if app.act_annotation_mode.isChecked():
        update_cursor(app)


def change_point(app, val) -> None:
    """Update rendered point size and keep 'round points' sticky."""
    v = max(1, min(int(val), 20))
    app.point_size = v

    render_points_as_spheres = (
        app.act_points_spheres.isChecked()
        if hasattr(app, "act_points_spheres")
        else app_helpers.render_points_as_spheres(app)
    )

    def _apply_point_size(actor, render_fn):
        try:
            prop = actor.GetProperty()
            prop.SetPointSize(v)
            try:
                prop.SetRenderPointsAsSpheres(render_points_as_spheres)
            except Exception:
                pass
            if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
                render_fn()
        except Exception:
            pass

    if hasattr(app, "actor") and app.actor is not None:
        _apply_point_size(app.actor, app.plotter.render)

    if hasattr(app, "actor_ref") and app.actor_ref is not None:
        _apply_point_size(app.actor_ref, app.plotter_ref.render)

    if app.act_annotation_mode.isChecked():
        update_cursor(app)


def pick_color(app) -> None:
    if app.clone_mode:
        return

    dialog = QtWidgets.QColorDialog(app)
    dialog.setWindowTitle("Select Color")
    if app.current_color is not None:
        dialog.setCurrentColor(QColor(*app.current_color))

    icon = app.windowIcon()
    if icon is not None and not icon.isNull():
        dialog.setWindowIcon(icon)
    else:
        from pathlib import Path

        base = Path(__file__).resolve().parents[1]
        for name in ("app.png", "app.ico"):
            candidate = base / "icons" / name
            if candidate.exists():
                dialog.setWindowIcon(QIcon(str(candidate)))
                break

    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        c = dialog.currentColor()
        app.current_color = [c.red(), c.green(), c.blue()]
        app._last_paint_color = app.current_color.copy()
        app.act_eraser.setChecked(False)


def select_swatch(app, col, btn=None) -> None:
    """
    Select paint color from menu or picker.
    UI-agnostic: no widgets, no swatches.
    """
    if app.clone_mode:
        return

    qc = QColor(col)
    app.current_color = [qc.red(), qc.green(), qc.blue()]
    app._last_paint_color = app.current_color.copy()

    if hasattr(app, "act_eraser"):
        app.act_eraser.setChecked(False)


def on_click(app, x, y) -> None:
    if not app.act_annotation_mode.isChecked():
        return
    picker = vtkPropPicker()
    h = app.plotter.interactor.height()
    picker.Pick(x, h - y, 0, app.plotter.renderer)
    pt = np.array(picker.GetPickPosition())
    if np.allclose(pt, (0, 0, 0)):
        return

    r_px = app.brush_size
    picker.ErasePickList()
    picker.Pick(x + r_px, h - y, 0, app.plotter.renderer)
    pt_edge = np.array(picker.GetPickPosition())

    world_r = np.linalg.norm(pt_edge - pt)
    idx = app.kdtree.query_ball_point(pt, world_r)

    if not idx:
        return
    old = app.colors[idx].copy()
    app.history.append((idx, old))
    app.redo_stack.clear()
    if app.clone_mode:
        app.colors[idx] = app.original_colors[idx]
    elif app.act_eraser.isChecked() or app.current_color is None:
        app.colors[idx] = app.original_colors[idx]
    else:
        app.colors[idx] = app.current_color

    app._session_edited[idx] = True
    app._mark_dirty_once()
    if hasattr(app, "toggle_ann_chk"):
        app.toggle_ann_chk.setEnabled(True)
    if hasattr(app, "act_toggle_annotations"):
        app.act_toggle_annotations.setEnabled(True)

    update_annotation_visibility(app)


def on_undo(app) -> None:
    if not app.history:
        return
    idx, old = app.history.pop()
    app.redo_stack.append((idx, app.colors[idx].copy()))
    app.colors[idx] = old
    app._session_edited[idx] = False
    if hasattr(app, "act_toggle_annotations"):
        app.act_toggle_annotations.setEnabled(True)
    if hasattr(app, "toggle_ann_chk"):
        app.toggle_ann_chk.setEnabled(True)
    if not np.any(app._session_edited):
        app._dirty.discard(app.index)
        app._decorate_nav_item(app.index)
        app._update_status_bar()
        try:
            app.statusBar().showMessage("Undo: no unsaved edits", 1500)
        except Exception:
            pass
    else:
        try:
            app.statusBar().showMessage("Undo: unsaved edits remain", 1500)
        except Exception:
            pass
    update_annotation_visibility(app)


def on_redo(app) -> None:
    if not app.redo_stack:
        return
    idx, cols = app.redo_stack.pop()
    app.history.append((idx, app.colors[idx].copy()))
    app.colors[idx] = cols
    app._session_edited[idx] = True
    if hasattr(app, "act_toggle_annotations"):
        app.act_toggle_annotations.setEnabled(True)
    if hasattr(app, "toggle_ann_chk"):
        app.toggle_ann_chk.setEnabled(True)
    app._mark_dirty_once()
    try:
        app.statusBar().showMessage("Redo: unsaved edits present", 1500)
    except Exception:
        pass
    update_annotation_visibility(app)


def on_toggle_ann_changed(app, state) -> None:
    app.annotations_visible = (state == QtCore.Qt.Checked)
    update_annotation_visibility(app)


def on_eraser_toggled(app, on: bool) -> None:
    if not app.act_annotation_mode.isChecked():
        app.act_annotation_mode.setChecked(True)
        update_cursor(app)

    if on:
        app.current_color = None
    else:
        app.current_color = app._last_paint_color.copy()


def reset_contrast(app) -> None:
    if "gamma" in app.ribbon_sliders:
        gamma_slider, gamma_lbl = app.ribbon_sliders["gamma"]
        gamma_slider.blockSignals(True)
        gamma_slider.setValue(100)
        gamma_slider.blockSignals(False)
        gamma_lbl.setText("1.00")

    current = app.colors.copy()
    untouched_mask = np.all(current == app.original_colors, axis=1)
    current[untouched_mask] = app.original_colors[untouched_mask]
    app.enhanced_colors = app.original_colors.copy()

    app.cloud["RGB"] = current
    update_annotation_visibility(app)

    if app.repair_mode and hasattr(app, "cloud_ref"):
        app.cloud_ref["RGB"] = app.original_colors.astype(np.uint8)
        if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
            app.plotter_ref.render()


def on_gamma_change(app, val) -> None:
    gamma = 2 ** ((val - 100) / 50.0)

    if hasattr(app, "ribbon_gamma_label"):
        app.ribbon_gamma_label.setText(f"{gamma:.2f}")
    elif hasattr(app, "tool_sliders"):
        try:
            _, lbl = app.tool_sliders.gamma
            lbl.setText(f"{gamma:.2f}")
        except Exception:
            pass

    original = app.original_colors.astype(np.float32) / 255.0
    min_vals = original.min(axis=0, keepdims=True)
    max_vals = original.max(axis=0, keepdims=True)
    stretched = (original - min_vals) / (max_vals - min_vals + 1e-5)

    corrected = np.power(stretched, gamma)
    app.enhanced_colors = (corrected * 255).astype(np.uint8)

    current = app.colors.copy()
    mask = np.all(current == app.original_colors, axis=1)
    current[mask] = app.enhanced_colors[mask]

    app.cloud["RGB"] = current
    update_annotation_visibility(app)

    if app.repair_mode and hasattr(app, "cloud_ref"):
        app.cloud_ref["RGB"] = app.original_colors.astype(np.uint8)
        if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
            app.plotter_ref.render()


def apply_auto_contrast(app) -> None:
    rgb = app.original_colors.astype(np.float32) / 255.0

    p_low, p_high = 2, 98
    lo = np.percentile(rgb, p_low, axis=0)
    hi = np.percentile(rgb, p_high, axis=0)

    stretched = (rgb - lo) / (hi - lo + 1e-5)
    stretched = np.clip(stretched, 0, 1)
    app.enhanced_colors = (stretched * 255).astype(np.uint8)

    current = app.colors.copy()
    mask = np.all(current == app.original_colors, axis=1)
    current[mask] = app.enhanced_colors[mask]

    app.cloud["RGB"] = current
    update_annotation_visibility(app)

    if app.repair_mode and hasattr(app, "cloud_ref"):
        app.cloud_ref["RGB"] = app.original_colors.astype(np.uint8)
        if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
            app.plotter_ref.render()

    if "gamma" in app.ribbon_sliders:
        gamma_slider, gamma_lbl = app.ribbon_sliders["gamma"]
        gamma_slider.blockSignals(True)
        gamma_slider.setValue(100)
        gamma_slider.blockSignals(False)
        gamma_lbl.setText("Auto")


def show_histograms(app) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Smoothed RGB Distributions - Original vs Enhanced")

    channels = ["Red", "Green", "Blue"]
    colors = ["r", "g", "b"]
    linestyles = ["-", "--"]

    for i, (label, color) in enumerate(zip(channels, colors)):
        orig_vals = app.original_colors[:, i].astype(np.float32)
        kde_orig = gaussian_kde(orig_vals)
        x = np.linspace(0, 255, 256)
        ax.plot(x, kde_orig(x), color=color, linestyle=linestyles[0], label=f"{label} (Original)")

        enh_vals = app.enhanced_colors[:, i].astype(np.float32)
        kde_enh = gaussian_kde(enh_vals)
        ax.plot(x, kde_enh(x), color=color, linestyle=linestyles[1], label=f"{label} (Enhanced)")

    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)


def set_annotations_visible(app, vis: bool) -> None:
    app.annotations_visible = bool(vis)
    if hasattr(app, "toggle_ann_chk"):
        app.toggle_ann_chk.blockSignals(True)
        app.toggle_ann_chk.setChecked(app.annotations_visible)
        app.toggle_ann_chk.blockSignals(False)
    update_annotation_visibility(app)


def current_base(app):
    base = getattr(app, "enhanced_colors", None)
    if base is None or len(base) != len(app.original_colors):
        base = app.original_colors
    return base


def on_alpha_change(app, val) -> None:
    app.annotation_alpha = max(0.0, min(1.0, val / 100.0))
    update_annotation_visibility(app)


def update_annotation_visibility(app) -> None:
    if getattr(app, "_is_closing", False):
        return

    if not hasattr(app, "cloud") or app.cloud is None:
        return

    base = getattr(app, "enhanced_colors", None)
    if base is None or len(base) != len(app.original_colors):
        base = app.original_colors
    base = base.astype(np.uint8)

    display = base.copy()

    if not getattr(app, "annotations_visible", True):
        app.cloud["RGB"] = display.astype(np.uint8)
        if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
            app.plotter.render()
        return

    edited_mask = np.any(app.colors != app.original_colors, axis=1)
    if not np.any(edited_mask):
        app.cloud["RGB"] = display.astype(np.uint8)
        if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
            app.plotter.render()
        return

    a = float(getattr(app, "annotation_alpha", 1.0))
    if a >= 0.999:
        display[edited_mask] = app.colors[edited_mask]
    elif a <= 0.001:
        pass
    else:
        fg = app.colors[edited_mask].astype(np.float32)
        bg = base[edited_mask].astype(np.float32)
        out = (a * fg + (1.0 - a) * bg).round().astype(np.uint8)
        display[edited_mask] = out

    app.cloud["RGB"] = display.astype(np.uint8)
    if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
        app.plotter.render()


def toggle_repair_mode(app, on: bool) -> None:
    was_split = bool(
        app.repair_mode or app.clone_mode or app.act_repair.isChecked() or app.act_clone.isChecked()
    )
    if on and not was_split:
        app._need_split_fit = True

    if on and app.clone_mode:
        app.act_clone.setChecked(False)

    app.repair_mode = bool(on)
    pending_split = bool(app.act_repair.isChecked() or app.act_clone.isChecked())
    want_split = bool(app.repair_mode or app.clone_mode or pending_split)
    app.plotter_ref.setVisible(want_split)
    app.vline.setVisible(want_split)

    if app.repair_mode and not app.act_annotation_mode.isChecked():
        app.act_annotation_mode.setChecked(True)

    if app.repair_mode:
        app.act_annotation_mode.blockSignals(True)
        app.act_annotation_mode.setChecked(True)
        app.act_annotation_mode.blockSignals(False)
        update_cursor(app)

        app.act_eraser.setChecked(True)
    elif not app.clone_mode:
        app.act_eraser.setChecked(False)

    if hasattr(app, "left_title"):
        app.left_title.setVisible(want_split)

    if app.repair_mode and hasattr(app, "cloud_ref"):
        app.cloud_ref["RGB"] = app.original_colors.astype(np.uint8)

    if want_split:
        app._link_cameras()
    else:
        app._unlink_cameras()

    if was_split != want_split:
        QtCore.QTimer.singleShot(0, app._finalize_layout)
    else:
        QtCore.QTimer.singleShot(0, app._position_overlays)
    update_annotation_visibility(app)


def toggle_clone_mode(app, on: bool) -> None:
    was_split = bool(
        app.repair_mode or app.clone_mode or app.act_repair.isChecked() or app.act_clone.isChecked()
    )
    if on and not was_split:
        app._need_split_fit = True
    app.clone_mode = bool(on)

    if on and app.repair_mode:
        app.act_repair.setChecked(False)

    pending_split = bool(app.act_repair.isChecked() or app.act_clone.isChecked())
    want_split = bool(app.repair_mode or app.clone_mode or pending_split)

    if app.clone_mode:
        app.act_annotation_mode.setChecked(True)
        app.act_toggle_annotations.setChecked(True)

        app.plotter_ref.setVisible(True)
        app.vline.setVisible(True)
        if hasattr(app, "left_title"):
            app.left_title.setVisible(True)
    elif not want_split:
        app.plotter_ref.setVisible(False)
        app.vline.setVisible(False)
        if hasattr(app, "left_title"):
            app.left_title.setVisible(False)

        app.current_color = app._last_paint_color.copy()

    if want_split:
        app._link_cameras()
    else:
        app._unlink_cameras()

    if was_split != want_split:
        QtCore.QTimer.singleShot(0, app._finalize_layout)
    else:
        QtCore.QTimer.singleShot(0, app._position_overlays)
    update_cursor(app)


def blend_into_mesh_subset(app, idx) -> None:
    """
    Update app.cloud['RGB'][idx] only, reflecting current annotation visibility/alpha.
    """
    if idx is None or len(idx) == 0:
        return

    base = getattr(app, "enhanced_colors", None)
    if base is None or len(base) != len(app.original_colors):
        base = app.original_colors

    if not getattr(app, "annotations_visible", True):
        app.cloud["RGB"][idx] = base[idx].astype(np.uint8)
        return

    a = float(getattr(app, "annotation_alpha", 1.0))
    if a >= 0.999:
        app.cloud["RGB"][idx] = app.colors[idx].astype(np.uint8)
    elif a <= 0.001:
        app.cloud["RGB"][idx] = base[idx].astype(np.uint8)
    else:
        fg = app.colors[idx].astype(np.float32)
        bg = base[idx].astype(np.float32)
        out = (a * fg + (1.0 - a) * bg).round().astype(np.uint8)
        app.cloud["RGB"][idx] = out
