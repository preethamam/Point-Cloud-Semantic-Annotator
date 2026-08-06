from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from configs.constants import REVIEW_FILE
from services import annotation_stats
from services.review_store import ReviewStore
from services.storage import load_state, log_gui, save_state


def init_review_state(app) -> None:
    app.review_store = ReviewStore.load(REVIEW_FILE)


def reset_for_new_project(app) -> None:
    """Start review.json over when the user picks a different annotation
    folder. review.json is a single file keyed by absolute file path, so
    without this, entries from every past project stay mixed into it
    (and into every future Excel export) forever."""
    app.review_store = ReviewStore()
    app.review_store.save(REVIEW_FILE)


def _current_key(app):
    if not app.files or app.index < 0 or app.index >= len(app.files):
        return None
    return str(app.files[app.index])


def refresh_review_comment_box(app) -> None:
    if not hasattr(app, "review_textbox"):
        return
    key = _current_key(app)
    text = app.review_store.get_comment(key) if key else ""
    app.review_textbox.blockSignals(True)
    app.review_textbox.setPlainText(text)
    app.review_textbox.blockSignals(False)


def on_review_comment_changed(app) -> None:
    key = _current_key(app)
    if key is None or not hasattr(app, "review_textbox"):
        return
    app.review_store.set_comment(
        key,
        app.review_textbox.toPlainText(),
        filename=app.files[app.index].name,
        annotation_path=key,
    )


def toggle_review_textbox(app, on: bool) -> None:
    if hasattr(app, "review_panel"):
        app.review_panel.setVisible(bool(on))

    def _refresh_canvas_layout():
        # setVisible() only queues the layout change; the plotter
        # interactors haven't actually resized yet on this call stack, so
        # positioning the overlay titles synchronously here would use
        # stale geometry (same fix as app_helpers.set_ribbon_display_mode).
        if hasattr(app, "_position_overlays"):
            app._position_overlays()
        if hasattr(app, "_schedule_fit"):
            app._schedule_fit()

    QtCore.QTimer.singleShot(0, _refresh_canvas_layout)

    if on and hasattr(app, "review_textbox"):
        app.review_textbox.setFocus()


def _record_stats(app, key: str) -> None:
    stats = annotation_stats.compute_stats(app)
    app.review_store.upsert_stats(key, stats)


def record_visit(app) -> None:
    """Snapshot stats for the file that was just loaded, in memory only
    (no disk write). Without this, a file's Excel/review.json row stays
    blank unless the user explicitly saves, saves a comment, or has it
    open at export time — even though it was visited."""
    key = _current_key(app)
    if key is None:
        return
    _record_stats(app, key)


def _show_toast(app, text: str, duration_ms: int = 1600) -> None:
    """Small borderless notification that auto-closes; doesn't block input
    the way a QMessageBox with an OK button would."""
    toast = QtWidgets.QLabel(text, app, QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint)
    toast.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    toast.setAlignment(QtCore.Qt.AlignCenter)
    toast.setStyleSheet("""
        QLabel {
            background-color: #f4f4f4;
            color: black;
            border: 1px solid #cfcfcf;
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 12px;
        }
    """)
    toast.adjustSize()

    anchor = app.mapToGlobal(QtCore.QPoint(app.width() // 2, app.height() - 90))
    toast.move(anchor.x() - toast.width() // 2, anchor.y())
    toast.show()
    QtCore.QTimer.singleShot(duration_ms, toast.close)


def save_comment(app) -> None:
    key = _current_key(app)
    if key is None:
        QtWidgets.QMessageBox.information(app, "Save Comment", "No point cloud is loaded.")
        return
    on_review_comment_changed(app)
    _record_stats(app, key)
    app.review_store.save(REVIEW_FILE)
    try:
        app.statusBar().showMessage(f"Comment saved for {app.files[app.index].name}", 2500)
    except Exception:
        pass
    _show_toast(app, "Comment saved")
    log_gui(f"save_comment: key={key}")


def record_stats_on_save(app) -> None:
    key = _current_key(app)
    if key is None:
        return
    _record_stats(app, key)
    app.review_store.save(REVIEW_FILE)


def flush_and_save(app) -> None:
    if _current_key(app) is not None:
        on_review_comment_changed(app)
    app.review_store.save(REVIEW_FILE)


def export_to_excel(app) -> None:
    key = _current_key(app)
    if key is not None:
        on_review_comment_changed(app)
        _record_stats(app, key)
    app.review_store.save(REVIEW_FILE)

    if not app.review_store.entries:
        QtWidgets.QMessageBox.information(
            app, "Export to Excel",
            "No review data to export yet. Load and review at least one point cloud first.",
        )
        return

    try:
        st = load_state()
        start_dir = st.get("last_export_dir", "") or str(app.ann_dir or Path.home())
    except Exception:
        start_dir = str(app.ann_dir or Path.home())

    default_name = f"{(app.ann_dir.name if app.ann_dir else 'review')}_review.xlsx"
    default_path = str(Path(start_dir) / default_name)

    dest, _ = QtWidgets.QFileDialog.getSaveFileName(
        app, "Export Review to Excel", default_path, "Excel Files (*.xlsx)"
    )
    if not dest:
        return
    if not dest.lower().endswith(".xlsx"):
        dest += ".xlsx"

    try:
        from services.excel_export import export_review_to_excel
    except ImportError:
        QtWidgets.QMessageBox.critical(
            app, "Export to Excel",
            "The 'openpyxl' package is required for Excel export.\n\n"
            "Install it with:\n    pip install openpyxl",
        )
        return

    count = export_review_to_excel(REVIEW_FILE, Path(dest))

    save_state({"last_export_dir": str(Path(dest).parent)})
    log_gui(f"export_to_excel: dest={dest} entries={count}")

    QtWidgets.QMessageBox.information(
        app, "Export Complete",
        f"Exported review data for {count} file(s) to:\n{dest}",
    )
