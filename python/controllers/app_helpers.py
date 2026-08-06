from __future__ import annotations

import os
import platform
from datetime import datetime, timezone

from PyQt5 import QtCore, QtWidgets

from configs.constants import APP_NAME, VERSION_NUMBER
from controllers import review
from services.storage import log_gui, save_state


def release_view_combo_focus(app) -> None:
    """Drop combo focus so the blue highlight doesn't linger."""
    def _focus():
        try:
            if hasattr(app, "plotter"):
                app.plotter.interactor.setFocus()
            else:
                app.setFocus()
        except Exception:
            app.setFocus()
    QtCore.QTimer.singleShot(0, _focus)


def on_nav_visibility_changed(app, visible: bool) -> None:
    if app.isMinimized():
        return
    if hasattr(app, "act_toggle_nav"):
        app.act_toggle_nav.setChecked(bool(visible))
    app._schedule_fit()


def on_change_event(app, event) -> None:
    if event.type() != QtCore.QEvent.WindowStateChange:
        return

    if app.isMinimized():
        if hasattr(app, "nav_dock"):
            app._nav_was_visible = app.nav_dock.isVisible()
        return

    want_visible = bool(getattr(app, "_nav_was_visible", True))
    if not hasattr(app, "nav_dock"):
        return

    if want_visible:
        app.nav_dock.setVisible(True)
        if hasattr(app, "act_toggle_nav"):
            app.act_toggle_nav.setChecked(True)
        target = int(getattr(app, "_nav_last_width", app.nav_dock.minimumWidth()))
        app.resizeDocks([app.nav_dock], [target], QtCore.Qt.Horizontal)
    else:
        if hasattr(app, "act_toggle_nav"):
            app.act_toggle_nav.setChecked(False)

def set_ribbon_display_mode(app, mode: str) -> None:
    """mode: 'fullscreen' | 'hidden' | 'shown'.
    'shown' re-docks the ribbon toolbar; anything else detaches it via
    removeToolBar() rather than just setVisible(False) -- a hidden-but-still
    -docked QToolBar leaves a residual strip of empty space in
    QMainWindow's toolbar area, whereas a fully removed one collapses to
    zero height."""
    if hasattr(app, "ribbon_tb"):
        attached = getattr(app, "_ribbon_toolbar_attached", True)
        if mode == "shown":
            if not attached:
                app.addToolBar(QtCore.Qt.TopToolBarArea, app.ribbon_tb)
            app.ribbon_tb.setVisible(True)
            app._ribbon_toolbar_attached = True
        elif attached:
            app.removeToolBar(app.ribbon_tb)
            app._ribbon_toolbar_attached = False

    app.statusBar().setVisible(True)

    if mode == "fullscreen":
        if not app.isFullScreen():
            app._pre_fullscreen_maximized = bool(app.windowState() & QtCore.Qt.WindowMaximized)
            app.showFullScreen()
    elif app.isFullScreen():
        # Windows can't go directly from a borderless fullscreen surface to
        # a bordered maximized one; restoring to normal first, in the same
        # call stack (no event-loop turn in between), avoids Qt getting the
        # window stuck in a broken in-between geometry.
        app.showNormal()
        if getattr(app, "_pre_fullscreen_maximized", True):
            app.showMaximized()

    app._ribbon_display_mode = mode

    # The toolbar attach/detach and window-state change above both resize
    # the canvas across more than one layout pass; positioning the overlay
    # titles or fitting the camera synchronously can use stale, not-yet-
    # settled widget geometry. Deferring lets Qt finish laying out before
    # we measure (same fix used elsewhere for this, see
    # controllers/annotation.py). A second, later pass is a safety net for
    # the full-screen case, whose OS-level window-state transition can
    # settle slower than a plain toolbar visibility change.
    def _refresh_canvas_layout():
        if hasattr(app, "_position_overlays"):
            app._position_overlays()
        if hasattr(app, "_schedule_fit"):
            app._schedule_fit()

    QtCore.QTimer.singleShot(0, _refresh_canvas_layout)
    QtCore.QTimer.singleShot(120, _refresh_canvas_layout)

    for attr, attr_mode in (
        ("act_ribbon_fullscreen", "fullscreen"),
        ("act_ribbon_hidden", "hidden"),
        ("act_ribbon_shown", "shown"),
    ):
        act = getattr(app, attr, None)
        if act is not None:
            act.blockSignals(True)
            act.setChecked(attr_mode == mode)
            act.blockSignals(False)


def clone_source(app):
    """Color source for Clone mode."""
    return app.original_colors


def is_split_mode(app) -> bool:
    return bool(app.repair_mode or app.clone_mode) and hasattr(app, "plotter_ref") and app.plotter_ref.isVisible()


def shared_cam_active(app) -> bool:
    return bool(is_split_mode(app) and app._shared_camera is not None)


def _env_flag(name: str):
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip().lower()
    if val in ("1", "true", "yes", "on", "y"):
        return True
    if val in ("0", "false", "no", "off", "n"):
        return False
    return None


def _get_opengl_info(app) -> tuple[str, str]:
    cached = getattr(app, "_gl_info", None)
    if isinstance(cached, tuple) and len(cached) == 2:
        return cached

    vendor = ""
    renderer = ""
    try:
        if hasattr(app, "plotter"):
            try:
                app.plotter.render()
            except Exception:
                pass
            ren_win = app.plotter.ren_win
            if hasattr(ren_win, "GetOpenGLVendor"):
                v = ren_win.GetOpenGLVendor()
                if v:
                    vendor = str(v)
            if hasattr(ren_win, "GetOpenGLRenderer"):
                r = ren_win.GetOpenGLRenderer()
                if r:
                    renderer = str(r)
            if hasattr(ren_win, "ReportCapabilities"):
                report = ren_win.ReportCapabilities()
                if isinstance(report, bytes):
                    report = report.decode(errors="ignore")
                if isinstance(report, str):
                    for line in report.splitlines():
                        if "OpenGL vendor string" in line and not vendor:
                            vendor = line.split(":", 1)[-1].strip()
                        elif "OpenGL renderer string" in line and not renderer:
                            renderer = line.split(":", 1)[-1].strip()
    except Exception:
        vendor = ""
        renderer = ""

    app._gl_info = (vendor, renderer)
    return vendor, renderer


def _set_gl_status(app, spheres_enabled: bool, source: str | None = None) -> None:
    vendor, renderer = _get_opengl_info(app)
    mode = "Spheres" if spheres_enabled else "Points"
    if source == "env":
        mode = f"{mode} (env)"
    elif source == "ui":
        mode = f"{mode} (ui)"
    if hasattr(app, "sb_gl"):
        v = vendor if vendor else "Unknown vendor"
        r = renderer if renderer else "Unknown renderer"
        app.sb_gl.setText(f"OpenGL: {v} | {r} | {mode}")


def render_points_as_spheres(app) -> bool:
    cached = getattr(app, "_render_points_as_spheres", None)
    if isinstance(cached, bool):
        return cached

    env_override = _env_flag("PCA_RENDER_POINTS_AS_SPHERES")
    if env_override is not None:
        app._render_points_source = "env"
        app._render_points_as_spheres = env_override
        _log_gl_info_once(app, env_override)
        return env_override

    decision = True

    app._render_points_source = "auto"
    app._render_points_as_spheres = decision
    _log_gl_info_once(app, decision)
    return decision


def _log_gl_info_once(app, spheres_enabled: bool) -> None:
    if getattr(app, "_gl_info_logged", False):
        return
    vendor, renderer = _get_opengl_info(app)
    source = getattr(app, "_render_points_source", "auto")
    _set_gl_status(app, spheres_enabled, source=source)
    log_gui(
        f"OpenGL: vendor='{vendor}' renderer='{renderer}' "
        f"points_as_spheres={spheres_enabled} source={source}"
    )
    app._gl_info_logged = True


def set_points_render_mode(app, on: bool) -> None:
    app._render_points_as_spheres = bool(on)
    app._render_points_source = "ui"

    if hasattr(app, "actor") and app.actor is not None:
        try:
            app.actor.GetProperty().SetRenderPointsAsSpheres(bool(on))
        except Exception:
            pass

    if hasattr(app, "actor_ref") and app.actor_ref is not None:
        try:
            app.actor_ref.GetProperty().SetRenderPointsAsSpheres(bool(on))
        except Exception:
            pass

    _set_gl_status(app, bool(on), source="ui")
    vendor, renderer = _get_opengl_info(app)
    log_gui(
        f"OpenGL: vendor='{vendor}' renderer='{renderer}' "
        f"points_as_spheres={bool(on)} source=ui"
    )

    if not getattr(app, "_is_closing", False) and not getattr(app, "_batch", False):
        try:
            app.plotter.render()
        except Exception:
            pass
        try:
            if hasattr(app, "plotter_ref") and app.plotter_ref.isVisible():
                app.plotter_ref.render()
        except Exception:
            pass


def show_about_dialog(app) -> None:
    QtWidgets.QMessageBox.about(
        app,
            f"About {APP_NAME}",
            f"""
    <b>{APP_NAME}</b><br>
    Version {VERSION_NUMBER}<br><br>

    <b>Description</b><br>
    {APP_NAME} is a professional tool for semantic annotation,
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
    </ul><b>Technology Stack</b><br>
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

    © {datetime.now(tz=timezone.utc).year} Preetham Manjunatha. All rights reserved.
    """,
    )


def close_event(app, e) -> None:
    app._is_closing = True

    try:
        app.plotter.interactor.removeEventFilter(app)
        app.plotter_ref.interactor.removeEventFilter(app)
    except Exception:
        pass

    for view in [getattr(app, "plotter_ref", None), getattr(app, "plotter", None)]:
        try:
            if view is not None:
                view.close()
        except Exception:
            pass

    try:
        app._loop_timer.stop()
    except Exception:
        pass

    try:
        app._loop_timer.stop()
        app.statusBar().clearMessage()
    except Exception:
        pass

    try:
        save_state({
            "annotation_dir": str(app.ann_dir or ""),
            "original_dir": str(app.orig_dir or ""),
            "index": int(app.index),
            "nav_dock_width": int(app.nav_dock.width()),
        })
    except Exception:
        pass

    try:
        review.flush_and_save(app)
    except Exception:
        pass

    super(type(app), app).closeEvent(e)
