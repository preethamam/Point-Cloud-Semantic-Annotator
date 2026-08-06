from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkIOPLY import vtkPLYWriter


@dataclass
class BulkCopyResult:
    path: str
    status: str  # "updated" | "unchanged" | "error"
    changed_points: int = 0
    error: str | None = None


def _write_cloud(pc: pv.PolyData, out_path: Path) -> None:
    ext = out_path.suffix.lower()
    if ext == ".ply":
        writer = vtkPLYWriter()
        writer.SetFileName(str(out_path))
        writer.SetInputData(pc)
        writer.SetArrayName("RGB")
        writer.SetFileTypeToBinary()
        writer.Write()
    elif ext == ".pcd":
        pc.save(str(out_path), binary=True)
    else:
        pc.save(str(out_path))


def copy_colors_for_file(ann_path: str, orig_path: str, ignore_rgb: tuple[int, int, int]) -> BulkCopyResult:
    """Worker for one annotation/original file pair: same 'ignore RGB, copy
    the rest from Original' merge as the interactive Single File Copy, then
    write the result straight back to the annotation file. Must stay a
    top-level, self-contained function (no app/GUI state) so it can be
    pickled and run in a separate joblib worker process."""
    ann_p = Path(ann_path)
    try:
        pc_ann = pv.read(str(ann_p))
        pc_orig = pv.read(str(orig_path))

        if pc_ann.n_points != pc_orig.n_points:
            return BulkCopyResult(str(ann_p), "error", error="Point count mismatch with Original")

        if "RGB" not in pc_orig.array_names:
            return BulkCopyResult(str(ann_p), "error", error="Original has no RGB data")

        if "RGB" in pc_ann.array_names:
            ann_colors = np.asarray(pc_ann["RGB"]).astype(np.uint8)
        else:
            ann_colors = np.zeros((pc_ann.n_points, 3), dtype=np.uint8)

        orig_colors = np.asarray(pc_orig["RGB"]).astype(np.uint8)

        ignore = np.array(ignore_rgb, dtype=np.uint8)
        idx = np.where(~np.all(ann_colors == ignore, axis=1))[0]

        if idx.size == 0:
            return BulkCopyResult(str(ann_p), "unchanged")

        new_colors = ann_colors.copy()
        new_colors[idx] = orig_colors[idx]

        if np.array_equal(new_colors, ann_colors):
            return BulkCopyResult(str(ann_p), "unchanged")

        pc_ann["RGB"] = new_colors
        _write_cloud(pc_ann, ann_p)
        return BulkCopyResult(str(ann_p), "updated", changed_points=int(idx.size))
    except Exception as exc:
        return BulkCopyResult(str(ann_p), "error", error=str(exc))
