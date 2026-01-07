from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from services.storage import save_state


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


def clone_source(app):
    """Color source for Clone mode."""
    return app.original_colors


def is_split_mode(app) -> bool:
    return bool(app.repair_mode or app.clone_mode) and hasattr(app, "plotter_ref") and app.plotter_ref.isVisible()


def shared_cam_active(app) -> bool:
    return bool(is_split_mode(app) and app._shared_camera is not None)


def show_about_dialog(app) -> None:
    QtWidgets.QMessageBox.about(
        app,
            "About Point Cloud Annotator",
            """
    <b>Point Cloud Annotator</b><br>
    Version 2.0.0<br><br>

    <b>Description</b><br>
    Point Cloud Annotator is a professional tool for semantic annotation,
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
    </ul>

    <b>Technology Stack</b><br>
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

    © 2026 Preetham Manjunatha. All rights reserved.
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

    super(type(app), app).closeEvent(e)
