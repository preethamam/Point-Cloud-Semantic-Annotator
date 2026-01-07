from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from joblib import Parallel, delayed

from services.annotation_state import is_annotated_pair
from services.storage import load_nav_dock_width


def on_nav_search_entered(app) -> None:
    """Go to index or filename from nav dock."""
    if not app.files:
        return

    text = app.nav_search.text().strip()

    if not text:
        app.nav_search.clear()
        app.nav_search.clearFocus()
        try:
            app.plotter.interactor.setFocus()
        except Exception:
            app.setFocus()
        return

    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(app.files):
            app._maybe_autosave_before_nav()
            app.index = idx
            app.history.clear()
            app.redo_stack.clear()
            app.load_cloud()
            app._position_overlays()
            app._sync_nav_selection()
            app._update_status_bar()
            app._nav_release_pending = True
            QtCore.QTimer.singleShot(0, app._reset_nav_search)
        else:
            app.nav_status.setText("Index out of range")
        return

    text_low = text.lower()
    matches = [
        i for i, p in enumerate(app.files)
        if text_low in p.name.lower()
    ]

    if not matches:
        app.nav_status.setText("No matching filenames")
        return

    idx = matches[0]
    app._maybe_autosave_before_nav()
    app.index = idx
    app.history.clear()
    app.redo_stack.clear()
    app.load_cloud()
    app._position_overlays()
    app._sync_nav_selection()
    app._update_status_bar()

    app._nav_release_pending = True
    QtCore.QTimer.singleShot(0, app._reset_nav_search)


def reset_nav_search(app) -> None:
    try:
        app.nav_search.blockSignals(True)
        app.nav_search.clear()
        app.nav_search.deselect()
        app.nav_search.blockSignals(False)
    except Exception:
        pass

    try:
        app.plotter.interactor.setFocus()
    except Exception:
        app.setFocus()


def nav_row_text(app, i: int) -> str:
    """Row label: '0001 | filename.ply' (1-based index)."""
    if not app.files:
        return ""
    idx_w = max(4, len(str(len(app.files))))
    return f"{i+1:0{idx_w}d} | {app.files[i].name}"


def populate_nav_list(app) -> None:
    if not hasattr(app, "nav_list"):
        return

    app._nav_item_widgets = {}
    app.thumbs.reset_queue()

    app.nav_list.blockSignals(True)
    app.nav_list.clear()

    if not app.files:
        app.nav_list.blockSignals(False)
        return

    for i in range(len(app.files)):
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(QtCore.QSize(app.NAV_THUMB_SIZE + 16, app.NAV_THUMB_SIZE + 48))
        item.setData(QtCore.Qt.UserRole, i)

        w = app._make_nav_item_widget(i)

        app.nav_list.addItem(item)
        app.nav_list.setItemWidget(item, w)

        app.thumbs.request_thumbnail(i)

    app.nav_list.blockSignals(False)

    app._sync_nav_selection()

    max_idx = len(app.files)
    for idx in (app._dirty | app._annotated | app._visited):
        if 0 <= idx < max_idx:
            app._decorate_nav_item(idx)


def sync_nav_selection(app) -> None:
    """Keep nav list selection in sync with app.index."""
    if not hasattr(app, "nav_list") or not app.files:
        return
    i = int(getattr(app, "index", 0))
    if i < 0 or i >= app.nav_list.count():
        return

    app.nav_list.blockSignals(True)
    app.nav_list.setCurrentRow(i)
    app.nav_list.scrollToItem(app.nav_list.currentItem(), QtWidgets.QAbstractItemView.PositionAtCenter)
    app.nav_list.blockSignals(False)


def on_nav_row_changed(app, row: int) -> None:
    """Single-click navigation from nav list."""
    if not app.files:
        return
    if row < 0 or row >= len(app.files):
        return
    if row == app.index:
        return

    app._maybe_autosave_before_nav()

    app.index = row
    app.history.clear()
    app.redo_stack.clear()
    app.load_cloud()
    app._position_overlays()
    app._update_status_bar()


def scan_annotated_files(app) -> None:
    """Detect which files are annotated on disk (joblib)."""
    if not app.files or not app.orig_dir:
        return

    pairs = []
    for p in app.files:
        o = app.orig_dir / p.name
        if o.exists():
            pairs.append((p, o))

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(is_annotated_pair)(a, o) for a, o in pairs
    )

    app._annotated.clear()
    for i, is_ann in enumerate(results):
        if is_ann:
            app._annotated.add(i)

    for idx in app._annotated:
        app._decorate_nav_item(idx)


def mark_dirty_once(app) -> None:
    """Mark current cloud dirty once per session."""
    if app.index not in app._dirty:
        app._dirty.add(app.index)
        app._decorate_nav_item(app.index)


def update_loop_status(app) -> None:
    update_status_bar(app)


def update_status_bar(app) -> None:
    if app.files:
        app.sb_viewing.setText(f"Viewing: {app.files[app.index].name}")
    else:
        app.sb_viewing.setText("")

    if app.files:
        app.sb_index.setText(f"File Index: {app.index + 1} / {len(app.files)}")
    else:
        app.sb_index.setText("")

    is_dirty = app.index in getattr(app, "_dirty", set())
    is_annot = app.index in getattr(app, "_annotated", set())
    if is_dirty:
        app.sb_anno.setText("Modified")
    elif is_annot:
        app.sb_anno.setText("Annotated")
    else:
        if app.thumbs.pending_count() > 0:
            app.sb_anno.setText("Processing Thumbnails.")
        else:
            app.sb_anno.setText("Clean")

    if app.act_loop.isChecked():
        app.sb_loop.setText(f"Looping ({app.loop_delay_sec:.1f} s)")
    else:
        app.sb_loop.setText("")

    total = len(app.files) if app.files else 0
    pending = app.thumbs.pending_count()
    done = max(0, total - pending)
    if total > 0:
        app.sb_thumb.setText(f"Thumbs: {done}/{total}")
    else:
        app.sb_thumb.setText("")


def restore_nav_width(app, default_width: int) -> None:
    try:
        w = load_nav_dock_width(default_width)
        app.resizeDocks(
            [app.nav_dock],
            [w],
            QtCore.Qt.Horizontal
        )
    except Exception:
        pass
