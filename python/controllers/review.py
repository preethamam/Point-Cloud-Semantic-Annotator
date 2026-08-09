from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from configs.constants import (
    REVIEW_BACKUP_DIR,
    REVIEW_BACKUP_KEEP,
    REVIEW_FILE,
    REVIEW_PREV_FILE,
)
from services import annotation_stats
from services.review_store import ReviewStore
from services.storage import load_state, log_gui, save_state

APPEND_MODE_KEY = "review_append_mode"
SHOW_TEXTBOX_KEY = "show_review_textbox"


def init_review_state(app) -> None:
    app.review_store = ReviewStore.load(REVIEW_FILE)
    # Persisted append/cumulative toggle. A missing or non-boolean value
    # degrades to OFF (today's per-project behavior), so a corrupt state
    # file can never wedge review recording.
    try:
        app._review_append_mode = bool(load_state().get(APPEND_MODE_KEY, False))
    except Exception:
        app._review_append_mode = False


def is_append_mode(app) -> bool:
    return bool(getattr(app, "_review_append_mode", False))


def restore_show_textbox(app) -> None:
    """Re-apply the persisted Show Review Textbox state on startup. Setting
    the action's checked state drives toggle_review_textbox (and keeps the
    ribbon button in sync), so the panel shows/hides to match the last run.
    Called after the menu, ribbon and review panel are built."""
    if not hasattr(app, "act_show_review_textbox"):
        return
    try:
        on = bool(load_state().get(SHOW_TEXTBOX_KEY, False))
    except Exception:
        on = False
    app.act_show_review_textbox.setChecked(on)


def _prune_backups(keep: int) -> None:
    try:
        backups = sorted(REVIEW_BACKUP_DIR.glob("review_*.json"))
        for old in backups[:-keep] if keep > 0 else backups:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _backup_review_file(reason: str) -> Path | None:
    """Copy the current review.json to a timestamped, rotating backup before
    any destructive operation, so an accidental clear is always recoverable."""
    try:
        if not REVIEW_FILE.exists():
            return None
        REVIEW_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = REVIEW_BACKUP_DIR / f"review_{ts}.json"
        shutil.copy2(REVIEW_FILE, dest)
        _prune_backups(REVIEW_BACKUP_KEEP)
        log_gui(f"review backup: reason={reason} dest={dest}")
        return dest
    except Exception as exc:
        log_gui(f"review backup failed: {exc}")
        return None


def _snapshot_prev() -> None:
    """One-deep undo snapshot written right before an OFF-mode auto-wipe.
    Overwritten each reset (not rotated) so routine folder switches don't
    spam backup files, while the immediately-prior state stays recoverable."""
    try:
        if REVIEW_FILE.exists():
            shutil.copy2(REVIEW_FILE, REVIEW_PREV_FILE)
    except Exception:
        pass


def set_append_mode(app, on: bool) -> None:
    app._review_append_mode = bool(on)
    save_state({APPEND_MODE_KEY: bool(on)})
    log_gui(f"review append mode: {'on' if on else 'off'}")


def reset_for_new_project(app) -> None:
    """Start review.json over when the user picks a different annotation
    folder. review.json is a single file keyed by absolute file path, so
    without this, entries from every past project stay mixed into it
    (and into every future Excel export) forever.

    When append (cumulative) mode is on this is a no-op: entries keep
    accumulating across folders and sessions. Destroying the record is then
    only ever done deliberately via clear_review_data()."""
    if is_append_mode(app):
        return
    if app.review_store.entries:
        _snapshot_prev()
    app.review_store = ReviewStore()
    app.review_store.save(REVIEW_FILE)


def clear_review_data(app) -> None:
    """Deliberate, confirmed wipe of the cumulative review record. Backs the
    file up first so an accidental clear can be recovered."""
    n = len(app.review_store.entries)
    reply = QtWidgets.QMessageBox.question(
        app,
        "Clear Review Data",
        (
            f"This permanently clears all review data ({n} file "
            f"entr{'y' if n == 1 else 'ies'}: comments and stats).\n\n"
            "A timestamped backup is saved first so this can be undone from\n"
            f"{REVIEW_BACKUP_DIR}\n\nContinue?"
        ),
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    if reply != QtWidgets.QMessageBox.Yes:
        return

    backup = _backup_review_file("manual clear")
    app.review_store = ReviewStore()
    app.review_store.save(REVIEW_FILE)
    refresh_review_comment_box(app)
    log_gui(f"clear_review_data: cleared={n} backup={backup}")
    _show_toast(app, "Review data cleared" + (" (backup saved)" if backup else ""))


def _entry_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data.get("entries", {}) or {})
    except Exception:
        return 0


def _backup_candidates() -> list[tuple[str, Path]]:
    """(label, path) pairs for restorable snapshots, most recent first: the
    rotating pre-clear/pre-restore backups plus the one-deep auto-wipe
    snapshot. Labels carry the filename so they stay unique."""
    items: list[tuple[float, str, Path]] = []

    def _add(path: Path, tag: str) -> None:
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        when = (
            datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            if mtime else "unknown"
        )
        n = _entry_count(path)
        label = f"{when}  |  {n} entr{'y' if n == 1 else 'ies'}{tag}  [{path.name}]"
        items.append((mtime, label, path))

    try:
        for p in REVIEW_BACKUP_DIR.glob("review_*.json"):
            _add(p, "")
    except Exception:
        pass
    if REVIEW_PREV_FILE.exists():
        _add(REVIEW_PREV_FILE, "  (last auto-cleared)")

    items.sort(key=lambda t: t[0], reverse=True)
    return [(label, path) for _, label, path in items]


def restore_review_data(app) -> None:
    """Restore review.json from a saved backup. The current data is backed
    up first, so a restore is itself undoable via another restore."""
    candidates = _backup_candidates()
    if not candidates:
        QtWidgets.QMessageBox.information(
            app, "Restore Review Data",
            f"No review backups found in\n{REVIEW_BACKUP_DIR}",
        )
        return

    labels = [label for label, _ in candidates]
    choice, ok = QtWidgets.QInputDialog.getItem(
        app, "Restore Review Data",
        "Choose a backup to restore (current data is backed up first):",
        labels, 0, False,
    )
    if not ok or not choice:
        return
    src = next(path for label, path in candidates if label == choice)

    restored = ReviewStore.load(src)
    n = len(restored.entries)
    reply = QtWidgets.QMessageBox.question(
        app, "Restore Review Data",
        f"Replace the current review data with this backup "
        f"({n} entr{'y' if n == 1 else 'ies'})?\n\n{src.name}",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    if reply != QtWidgets.QMessageBox.Yes:
        return

    _backup_review_file("pre-restore")
    app.review_store = restored
    app.review_store.save(REVIEW_FILE)
    refresh_review_comment_box(app)
    log_gui(f"restore_review_data: src={src} entries={n}")
    _show_toast(app, f"Restored {n} entr{'y' if n == 1 else 'ies'}")


def rekey_on_move(app, old_path: str, new_path: str, *, filename: str | None = None) -> None:
    """Keep a review entry pointing at its file after an in-app move, so the
    comment/stats aren't orphaned. Persists only if something actually moved."""
    try:
        if app.review_store.rekey(old_path, new_path, filename=filename):
            app.review_store.save(REVIEW_FILE)
            log_gui(f"review rekey: {old_path} -> {new_path}")
    except Exception as exc:
        log_gui(f"review rekey failed: {exc}")


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

    # Persist so the panel's shown/hidden state carries across app runs.
    save_state({SHOW_TEXTBOX_KEY: bool(on)})

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
