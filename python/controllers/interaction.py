from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets


def event_filter(app, obj, event):
    if getattr(app, "_is_closing", False):
        return False

    is_text_input = isinstance(obj, QtWidgets.QLineEdit)

    if event.type() == QtCore.QEvent.KeyPress:
        if is_text_input:
            return False
        if event.key() == QtCore.Qt.Key_B:
            app._waiting = "brush"
            return True

        if event.key() == QtCore.Qt.Key_D:
            app._waiting = "point"
            return True

        if event.key() == QtCore.Qt.Key_A:
            app._waiting = "alpha"
            return True

        if event.key() == QtCore.Qt.Key_Z:
            app._waiting = "zoom"
            return True

        if event.key() == QtCore.Qt.Key_G:
            app._waiting = "gamma"
            return True

    if event.type() == QtCore.QEvent.KeyRelease:
        if is_text_input:
            return False
        if event.key() in (
            QtCore.Qt.Key_B,
            QtCore.Qt.Key_D,
            QtCore.Qt.Key_A,
            QtCore.Qt.Key_Z,
            QtCore.Qt.Key_G,
        ):
            app._waiting = None

    if obj is getattr(app, "nav_search", None):
        if event.type() == QtCore.QEvent.FocusIn and getattr(app, "_nav_release_pending", False):
            app._nav_release_pending = False
            if event.reason() == QtCore.Qt.MouseFocusReason:
                return False
            QtCore.QTimer.singleShot(0, lambda: app.plotter.interactor.setFocus())
            return True

    if obj is getattr(app, "nav_search", None):
        if event.type() == QtCore.QEvent.ShortcutOverride:
            k = event.key()
            if k in (
                QtCore.Qt.Key_Left,
                QtCore.Qt.Key_Right,
                QtCore.Qt.Key_Home,
                QtCore.Qt.Key_End,
                QtCore.Qt.Key_PageUp,
                QtCore.Qt.Key_PageDown,
            ):
                event.accept()
                return True

        if event.type() == QtCore.QEvent.KeyPress:
            k = event.key()
            if k == QtCore.Qt.Key_Left:
                app.on_prev()
                return True
            if k == QtCore.Qt.Key_Right:
                app.on_next()
                return True
            if k == QtCore.Qt.Key_Home:
                app.on_first()
                return True
            if k == QtCore.Qt.Key_End:
                app.on_last()
                return True
            if k == QtCore.Qt.Key_PageUp:
                app.on_page(-10)
                return True
            if k == QtCore.Qt.Key_PageDown:
                app.on_page(+10)
                return True

    if obj is getattr(app, "plotter", None) or obj is getattr(app, "plotter_ref", None):
        if event.type() == QtCore.QEvent.Wheel:
            delta_y = event.angleDelta().y()
            # Right pane wheel
            if obj is app.plotter:
                app._zoom_at_cursor_for(app.plotter, event.x(), event.y(), delta_y)
                return True
            # Left pane wheel (original view)
            if obj is app.plotter_ref:
                if not app.plotter_ref.isVisible():
                    return False
                if app._shared_cam_active():
                    app._zoom_at_cursor_for(app.plotter, event.x(), event.y(), delta_y)
                else:
                    app._zoom_at_cursor_for(app.plotter_ref, event.x(), event.y(), delta_y)
                return True

    if obj is getattr(app, "plotter", None) or obj is getattr(app, "plotter_ref", None):
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            if not hasattr(app, "colors"):
                return False
            if not app.act_annotation_mode.isChecked():
                return False
            if app.clone_mode and obj is not app.plotter_ref:
                return False
            if (not app.clone_mode and obj is not app.plotter):
                return False
            app._stroke_active = True
            app._in_stroke = True
            app._stroke_idxs = set()
            app._colors_before_stroke = app.colors.copy()
            app._last_paint_xy = None
            app._anchor_xy = (event.x(), event.y())
            app._line_len_px = 0.0
            app._constrain_line = bool(event.modifiers() & QtCore.Qt.ShiftModifier)

            return True

        if event.type() == QtCore.QEvent.MouseMove:
            if not hasattr(app, "colors"):
                return False
            if not app.act_annotation_mode.isChecked():
                return False
            if app.clone_mode and obj is not app.plotter_ref:
                return False
            if (not app.clone_mode and obj is not app.plotter):
                return False
            if not app._stroke_active:
                return False

            if app._paint_timer.elapsed() < app._min_paint_ms:
                return True

            app._paint_timer.restart()

            x, y = event.x(), event.y()

            if app._constrain_line:
                # Straight line from anchor to current
                if app._anchor_xy is None:
                    app._anchor_xy = (x, y)
                ax, ay = app._anchor_xy
                dx = x - ax
                dy = y - ay
                dist = float(np.hypot(dx, dy))
                if dist <= 1e-6:
                    return True

                step = app._paint_step_frac * app.brush_size
                next_len = app._line_len_px + step
                t0 = app._line_len_px / dist
                t1 = min(next_len / dist, 1.0)

                if t1 <= t0:
                    return True

                nx = int(ax + dx * t1)
                ny = int(ay + dy * t1)

                idxs = app._compute_brush_idx(nx, ny)
                if idxs:
                    app._stroke_idxs.update(idxs)
                    if app.clone_mode:
                        app.colors[idxs] = app._clone_source()[idxs]
                    elif app.act_eraser.isChecked() or app.current_color is None:
                        app.colors[idxs] = app.original_colors[idxs]
                    else:
                        app.colors[idxs] = app.current_color

                    app._session_edited[idxs] = True
                    app._mark_dirty_once()
                    if hasattr(app, "toggle_ann_chk"):
                        app.toggle_ann_chk.setEnabled(True)
                    if hasattr(app, "act_toggle_annotations"):
                        app.act_toggle_annotations.setEnabled(True)
                    app._blend_into_mesh_subset(idxs)

                app._line_len_px = t1 * dist
                return True

            # Freehand with interpolated stamping
            if app._last_paint_xy is None:
                app._last_paint_xy = (x, y)

            lx, ly = app._last_paint_xy
            dx = x - lx
            dy = y - ly
            dist = float(np.hypot(dx, dy))

            if dist < 1e-6:
                return True

            step = app._paint_step_frac * app.brush_size
            steps = max(1, int(dist / max(step, 1.0)))

            for i in range(1, steps + 1):
                t = float(i) / float(steps)
                nx = int(lx + dx * t)
                ny = int(ly + dy * t)
                idxs = app._compute_brush_idx(nx, ny)
                if idxs:
                    app._stroke_idxs.update(idxs)
                    if app.clone_mode:
                        app.colors[idxs] = app._clone_source()[idxs]
                    elif app.act_eraser.isChecked() or app.current_color is None:
                        app.colors[idxs] = app.original_colors[idxs]
                    else:
                        app.colors[idxs] = app.current_color

                    app._session_edited[idxs] = True
                    app._mark_dirty_once()
                    if hasattr(app, "toggle_ann_chk"):
                        app.toggle_ann_chk.setEnabled(True)
                    if hasattr(app, "act_toggle_annotations"):
                        app.act_toggle_annotations.setEnabled(True)
                    app._blend_into_mesh_subset(idxs)

            app._last_paint_xy = (x, y)
            return True

        if (app._stroke_active
                and event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton):
            if not app.act_annotation_mode.isChecked():
                return False
            app._stroke_active = False
            app._in_stroke = False
            app._last_paint_xy = None
            app._anchor_xy = None
            app._line_len_px = 0.0
            app._constrain_line = False

            if app._stroke_idxs:
                idxs = list(app._stroke_idxs)
                old = app._colors_before_stroke[idxs]
                app.history.append((idxs, old))
                app.redo_stack.clear()
            app._colors_before_stroke = None

            app.update_annotation_visibility()
            return True

    return None
