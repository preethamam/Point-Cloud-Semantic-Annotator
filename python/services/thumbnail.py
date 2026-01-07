from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import numpy as np
import pyvista as pv
from joblib import Parallel, delayed
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap

from configs.constants import (
    PERCENTAGE_CORE_FACTOR,
    THUMB_BACKEND,
    THUMB_DIR,
    THUMB_MAX_PTS,
    THUMB_SIZE,
)

THUMB_DIR.mkdir(parents=True, exist_ok=True)
thumb_n_jobs = max(1, int((os.cpu_count() or 1) * PERCENTAGE_CORE_FACTOR))


def generate_thumbnail_job(path: Path, out_png: Path, size: int = THUMB_SIZE) -> None:
    """
    Generate a thumbnail PNG for a point cloud.
    Runs in background worker. No Qt calls allowed here.
    """
    try:
        pc = pv.read(str(path))
        if pc.n_points == 0:
            return
        if pc.n_points > THUMB_MAX_PTS:
            idx = np.linspace(0, pc.n_points - 1, THUMB_MAX_PTS).astype(int)
            pc = pc.extract_points(idx)

        plotter = pv.Plotter(off_screen=True, window_size=(size, size))
        plotter.set_background("white")

        if "RGB" in pc.array_names:
            plotter.add_points(pc, scalars="RGB", rgb=True, point_size=4)
        else:
            plotter.add_points(pc, color="gray", point_size=4)

        # Top-down orthographic view (no zoom, no border)
        cam = plotter.camera
        cam.ParallelProjectionOn()
        cam.SetViewUp(0, 1, 0)

        xmin, xmax, ymin, ymax, zmin, zmax = pc.bounds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(cx, cy, zmax + (zmax - zmin + 1e-3))

        xr = xmax - xmin
        yr = ymax - ymin
        cam.SetParallelScale(0.5 * max(xr, yr))

        plotter.reset_camera_clipping_range()

        img = plotter.screenshot(transparent_background=False)
        plotter.close()

        img = img[:, :, :3].astype(np.uint8)
        Image.fromarray(img).save(out_png)

    except Exception:
        pass


class ThumbnailService:
    def __init__(self, app, nav_thumb_size: int):
        self.app = app
        self.nav_thumb_size = nav_thumb_size
        self._thumb_lock = threading.Lock()
        self._thumb_job_set = set()        # (src_path, out_png)
        self._thumb_out_by_idx = {}        # idx -> out_png
        self._thumb_worker_running = False
        self._thumb_worker_start_pending = False

    def reset_queue(self) -> None:
        with self._thumb_lock:
            self._thumb_out_by_idx = {}
            self._thumb_job_set.clear()

    def pending_count(self) -> int:
        return len(self._thumb_out_by_idx)

    def thumb_key(self, ann_path: Path) -> str:
        """
        Stable thumbnail key.
        Hashes ONLY the ORIGINAL file if available.
        Annotation edits must NOT affect thumbnails.
        """
        if self.app.orig_dir is not None:
            orig = self.app.orig_dir / ann_path.name
            src = orig if orig.exists() else ann_path
        else:
            src = ann_path

        st = src.stat()

        h = hashlib.sha1()
        h.update(str(src.resolve()).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
        return h.hexdigest()

    def thumb_path(self, path: Path) -> Path:
        return THUMB_DIR / f"{self.thumb_key(path)}.png"

    def thumb_exists(self, path: Path) -> bool:
        return self.thumb_path(path).exists()

    def request_thumbnail(self, idx: int) -> None:
        """
        Schedule thumbnail generation.
        ALWAYS use original folder as source of truth if available.
        """
        if not self.app.files or idx < 0 or idx >= len(self.app.files):
            return

        ann_path = self.app.files[idx]

        if self.app.orig_dir is not None:
            orig_path = self.app.orig_dir / ann_path.name
            src_path = orig_path if orig_path.exists() else ann_path
        else:
            src_path = ann_path

        out_png = self.thumb_path(ann_path)
        if out_png.exists():
            return

        with self._thumb_lock:
            self._thumb_out_by_idx[idx] = out_png
            self._thumb_job_set.add((src_path, out_png))

        if not self._thumb_worker_running and not self._thumb_worker_start_pending:
            self._thumb_worker_start_pending = True
            QtCore.QTimer.singleShot(0, self._start_thumb_worker)

    def _start_thumb_worker(self) -> None:
        if getattr(self, "_thumb_worker_running", False):
            return

        self._thumb_worker_start_pending = False
        self._thumb_worker_running = True

        def worker():
            try:
                while True:
                    with self._thumb_lock:
                        jobs = list(self._thumb_job_set)
                        self._thumb_job_set.clear()

                    if not jobs:
                        break

                    Parallel(
                        n_jobs=thumb_n_jobs,
                        backend=THUMB_BACKEND,
                        verbose=0
                    )(
                        delayed(generate_thumbnail_job)(src, out, THUMB_SIZE)
                        for src, out in jobs
                    )
            finally:
                self._thumb_worker_running = False

        threading.Thread(target=worker, daemon=True).start()

    def thumb_icon_for_index(self, idx: int):
        """
        Return QIcon for thumbnail if available, else None.
        UI-safe (Qt only, no disk generation).
        """
        try:
            path = self.app.files[idx]
            png = self.thumb_path(path)
            if not png.exists():
                return None
            pix = QPixmap(str(png))
            if pix.isNull():
                return None
            pix = pix.scaled(
                THUMB_SIZE, THUMB_SIZE,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            return QIcon(pix)
        except Exception:
            return None

    def refresh_nav_thumbnail(self, idx: int) -> None:
        entry = self.app._nav_item_widgets.get(idx)
        if entry is None:
            return
        lbl = entry["img"]
        if lbl is None:
            return

        icon = self.thumb_icon_for_index(idx)
        if icon is None:
            return

        pix = icon.pixmap(self.nav_thumb_size, self.nav_thumb_size)
        lbl.setPixmap(pix)

    def poll_thumbnails(self) -> None:
        """Refresh UI icons when thumbnail files appear on disk."""
        with self._thumb_lock:
            pending = list(self._thumb_out_by_idx.items())

        updated = []
        for idx, out_png in pending[:60]:
            if out_png.exists():
                updated.append(idx)

        if updated:
            with self._thumb_lock:
                for idx in updated:
                    self._thumb_out_by_idx.pop(idx, None)

            for idx in updated:
                self.refresh_nav_thumbnail(idx)

            if hasattr(self.app, "nav_list"):
                self.app.nav_list.viewport().update()

        self.app._update_status_bar()

    def clear_thumbnail_cache(self) -> None:
        if not THUMB_DIR.exists():
            QtWidgets.QMessageBox.warning(
                self.app, "Thumbnail Cache",
                "Thumbnail cache directory is not initialized yet."
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self.app,
            "Clear Thumbnail Cache",
            "Delete the entire thumbnail cache folder?\n"
            "Thumbnails will be regenerated automatically.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        import shutil

        try:
            if THUMB_DIR.exists():
                shutil.rmtree(THUMB_DIR)

            THUMB_DIR.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.app, "Error", f"Failed to clear thumbnail cache:\n{e}"
            )
            return

        if hasattr(self.app, "_nav_item_widgets"):
            for entry in self.app._nav_item_widgets.values():
                entry["img"].clear()

        QtWidgets.QMessageBox.information(
            self.app, "Thumbnail Cache", "Thumbnail cache cleared."
        )

    def prune_thumbnail_cache(self) -> None:
        if self.app.orig_dir is None or not THUMB_DIR.exists():
            return

        valid_keys = set()
        for p in self.app.orig_dir.glob("*.ply"):
            valid_keys.add(self.thumb_key(p))

        for thumb in THUMB_DIR.glob("*.png"):
            if thumb.stem not in valid_keys:
                try:
                    thumb.unlink()
                except Exception:
                    pass
