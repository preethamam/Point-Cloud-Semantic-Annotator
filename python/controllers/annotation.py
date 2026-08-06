from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QCursor, QIcon, QPainter, QPixmap
from scipy.stats import gaussian_kde
from vtkmodules.vtkRenderingCore import vtkPropPicker
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from controllers import app_helpers
from services.storage import log_gui


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
    # Manual painting means the unsaved state is no longer purely a
    # "copied from Original" result, so drop the special status label.
    app._copied_from_original.discard(app.index)
    if hasattr(app, "_update_status_bar"):
        app._update_status_bar()
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
    # Provenance of the remaining unsaved edits is ambiguous once you start
    # undoing, so drop the special "copied from Original" status label.
    app._copied_from_original.discard(app.index)
    if not np.any(app._session_edited):
        app._dirty.discard(app.index)
        app._decorate_nav_item(app.index)
        app._update_status_bar()
        try:
            app.statusBar().showMessage("Undo: no unsaved edits", 1500)
        except Exception:
            pass
    else:
        app._update_status_bar()
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
    app._copied_from_original.discard(app.index)
    if hasattr(app, "act_toggle_annotations"):
        app.act_toggle_annotations.setEnabled(True)
    if hasattr(app, "toggle_ann_chk"):
        app.toggle_ann_chk.setEnabled(True)
    app._mark_dirty_once()
    app._update_status_bar()
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


def copy_color_from_original(app, r: int, g: int, b: int) -> None:
    """Copy colors from the Original Point Cloud into the Annotated Point
    Cloud for every point, except points whose current Annotated color
    already equals (r, g, b) — those are left untouched."""
    if not hasattr(app, "colors") or app.colors is None:
        _show_copy_status(app, "Copy colors: no point cloud is loaded")
        return

    ignore = np.array([r, g, b], dtype=np.uint8)
    change_mask = ~np.all(app.colors == ignore, axis=1)
    idx = np.where(change_mask)[0]

    if idx.size == 0:
        _show_copy_status(app, "Copy colors: every point already matches the ignore color")
        return

    old = app.colors[idx].copy()
    new = app.original_colors[idx]
    if np.array_equal(old, new):
        _show_copy_status(app, "Copy colors: no changes needed (already matches Original)")
        return

    app.history.append((idx, old))
    app.redo_stack.clear()
    app.colors[idx] = new
    app._session_edited[idx] = True
    app._mark_dirty_once()
    app._copied_from_original.add(app.index)

    if hasattr(app, "toggle_ann_chk"):
        app.toggle_ann_chk.setEnabled(True)
    if hasattr(app, "act_toggle_annotations"):
        app.act_toggle_annotations.setEnabled(True)

    update_annotation_visibility(app)
    if hasattr(app, "_update_status_bar"):
        app._update_status_bar()

    _show_copy_status(
        app, f"Copied original colors to {idx.size} point(s), ignoring RGB({r}, {g}, {b})"
    )


def _show_copy_status(app, message: str) -> None:
    try:
        app.statusBar().showMessage(message, 3000)
    except Exception:
        pass
    if hasattr(app, "sb_anno"):
        try:
            app.sb_anno.setToolTip(message)
        except Exception:
            pass


# Matches copy_color_panel.PANEL_STYLE so the progress dialog reads as part
# of the same feature instead of a stock OS dialog.
_BULK_PROGRESS_STYLE = """
    QProgressDialog {
        background: #ffffff;
    }
    QProgressDialog QLabel {
        color: #1a1a1a;
        background: transparent;
        font-size: 12px;
    }
    QProgressBar {
        background: #f3f4f6;
        border: 1px solid #d7d7dc;
        border-radius: 6px;
        text-align: center;
        color: #1a1a1a;
        font-size: 11px;
        min-height: 18px;
    }
    QProgressBar::chunk {
        background-color: #0078D4;
        border-radius: 5px;
    }
    QProgressDialog QPushButton {
        background: #f4f4f4;
        color: #000000;
        border: 1px solid #d7d7dc;
        border-radius: 4px;
        padding: 6px 14px;
        font-size: 12px;
    }
    QProgressDialog QPushButton:hover {
        background: #e6f1fb;
        border: 1px solid #0078D4;
    }
    QProgressDialog QPushButton:pressed {
        background: #d7ebfa;
    }
    """


class BulkCopyWorker(QtCore.QThread):
    """Runs the 'ignore RGB, copy the rest from Original' merge across many
    annotation/original file pairs using joblib worker processes, in small
    batches so progress can be reported back to the GUI between batches."""

    progress = QtCore.pyqtSignal(int, int)
    results_ready = QtCore.pyqtSignal(list)

    _BATCH_SIZE = 16

    def __init__(self, pairs: list[tuple[str, str]], ignore_rgb: tuple[int, int, int], parent=None):
        super().__init__(parent)
        self.pairs = pairs
        self.ignore_rgb = ignore_rgb
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from joblib import Parallel, delayed

        from services.bulk_copy import copy_colors_for_file

        results = []
        total = len(self.pairs)
        done = 0
        self.progress.emit(0, total)

        for start in range(0, total, self._BATCH_SIZE):
            if self._cancel:
                break
            batch = self.pairs[start:start + self._BATCH_SIZE]
            batch_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(copy_colors_for_file)(ann, orig, self.ignore_rgb)
                for ann, orig in batch
            )
            results.extend(batch_results)
            done += len(batch)
            self.progress.emit(done, total)

        self.results_ready.emit(results)


def bulk_copy_colors(app, r: int, g: int, b: int) -> None:
    """Apply the same ignore-RGB / copy-from-Original merge as Single File
    Copy, but across every annotation file that has a matching Original
    file, in parallel."""
    if not app.files or not app.orig_dir:
        QtWidgets.QMessageBox.information(
            app, "Bulk Copy",
            "Open both an Annotation folder and an Original folder first.",
        )
        return

    # Flush the currently open file to disk first so the bulk pass sees its
    # latest in-memory edits instead of a stale on-disk copy.
    edited = (
        getattr(app, "_session_edited", None) is not None
        and np.any(app._session_edited)
    )
    if edited:
        app.on_save(_autosave=True)

    pairs = []
    skipped_names = []
    for p in app.files:
        o = app.orig_dir / p.name
        if o.exists():
            pairs.append((str(p), str(o)))
        else:
            skipped_names.append(p.name)

    if not pairs:
        QtWidgets.QMessageBox.information(
            app, "Bulk Copy",
            "No annotation files have a matching file in the Original folder.",
        )
        return

    msg = (
        f"This will overwrite {len(pairs)} annotation file(s), copying colors "
        f"from the Original folder and ignoring RGB({r}, {g}, {b}) "
        "(points already at that color are left untouched)."
    )
    if skipped_names:
        msg += f"\n\n{len(skipped_names)} file(s) will be skipped (no matching Original file)."
    msg += "\n\nThis cannot be undone. Continue?"

    choice = QtWidgets.QMessageBox.question(
        app, "Bulk Copy Original Colors", msg,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
    )
    if choice != QtWidgets.QMessageBox.Yes:
        return

    progress_dlg = QtWidgets.QProgressDialog(
        f"Copying original colors... (0 / {len(pairs)})", "Cancel", 0, len(pairs), app
    )
    progress_dlg.setWindowTitle("Bulk Copy")
    progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
    progress_dlg.setMinimumDuration(0)
    progress_dlg.setAutoClose(False)
    progress_dlg.setAutoReset(False)
    progress_dlg.setStyleSheet(_BULK_PROGRESS_STYLE)
    progress_dlg.setMinimumWidth(320)
    progress_dlg.setValue(0)

    worker = BulkCopyWorker(pairs, (r, g, b), parent=app)
    app._bulk_copy_worker = worker  # keep a reference alive while it runs

    def _on_progress(done, total):
        progress_dlg.setMaximum(total)
        progress_dlg.setLabelText(f"Copying original colors... ({done} / {total})")
        progress_dlg.setValue(done)

    def _on_results(results):
        progress_dlg.close()
        _finish_bulk_copy(app, pairs, results, skipped_names)
        app._bulk_copy_worker = None

    progress_dlg.canceled.connect(worker.cancel)
    worker.progress.connect(_on_progress)
    worker.results_ready.connect(_on_results)
    worker.start()


def _finish_bulk_copy(app, pairs, results, skipped_names) -> None:
    updated = [r for r in results if r.status == "updated"]
    unchanged = [r for r in results if r.status == "unchanged"]
    errors = [r for r in results if r.status == "error"]
    not_processed = len(pairs) - len(results)

    summary = (
        f"Bulk copy finished.\n\n"
        f"Updated: {len(updated)}\n"
        f"Unchanged (nothing to copy): {len(unchanged)}\n"
        f"Skipped (no matching Original): {len(skipped_names)}\n"
        f"Errors: {len(errors)}"
    )
    if not_processed:
        summary += f"\nCancelled before processing: {not_processed}"
    if errors:
        preview = "\n".join(f"- {Path(r.path).name}: {r.error}" for r in errors[:10])
        summary += f"\n\nFirst error(s):\n{preview}"
        if len(errors) > 10:
            summary += f"\n... and {len(errors) - 10} more."

    QtWidgets.QMessageBox.information(app, "Bulk Copy Complete", summary)

    log_gui(
        f"bulk_copy_colors: updated={len(updated)} unchanged={len(unchanged)} "
        f"skipped={len(skipped_names)} errors={len(errors)} cancelled={not_processed}"
    )

    if not updated:
        return

    # Files changed on disk; refresh caches/badges and reload whatever's
    # currently on screen so the GUI matches what was just written.
    app.thumbs.new_generation()
    app.thumbs.prune_ann_thumbs()
    if hasattr(app, "_scan_annotated_files"):
        try:
            app._scan_annotated_files()
        except Exception:
            pass
    if hasattr(app, "_populate_nav_list"):
        app._populate_nav_list()
    app.history.clear()
    app.redo_stack.clear()
    app.load_cloud()


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
