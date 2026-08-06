from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAX_CLASS_ROWS = 20


def _hex_to_rgb(hexcol: str) -> tuple[int, int, int]:
    h = hexcol.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def annotation_status_label(app) -> str:
    """Clean / Modified / Annotated / Copied Original Colors for app.index."""
    is_dirty = app.index in getattr(app, "_dirty", set())
    is_annot = app.index in getattr(app, "_annotated", set())
    is_copied = app.index in getattr(app, "_copied_from_original", set())
    if is_copied and (is_dirty or is_annot):
        return "Copied Original Colors"
    if is_dirty:
        return "Modified"
    if is_annot:
        return "Annotated"
    return "Clean"


@dataclass
class AnnotationStats:
    filename: str
    annotation_path: str
    original_path: str | None
    file_index: int
    total_files: int
    file_format: str

    annotation_file_size_bytes: int | None
    annotation_file_mtime: str | None
    original_file_size_bytes: int | None
    original_file_mtime: str | None

    point_count: int
    bbox_min: list
    bbox_max: list
    bbox_extent: list
    centroid: list

    annotated_point_count: int
    annotated_percentage: float
    unannotated_point_count: int
    unannotated_percentage: float
    distinct_annotation_colors: int
    class_breakdown: list = field(default_factory=list)

    status: str = "Clean"
    is_dirty: bool = False
    is_annotated: bool = False
    is_visited: bool = False
    is_copied_from_original: bool = False

    brush_size_px: float = 0.0
    point_size_px: int = 0
    annotation_alpha_pct: int = 100
    gamma_value: float = 1.0
    render_points_as_spheres: bool = True

    stats_updated_at: str = ""


def _class_breakdown(app, colors: np.ndarray, edited_mask: np.ndarray, total: int) -> tuple[list, int]:
    if not np.any(edited_mask):
        return [], 0

    swatch_lookup = {
        _hex_to_rgb(hexcol): name for name, hexcol in getattr(app, "_SWATCHES", [])
    }

    subset = colors[edited_mask]
    uniq, counts = np.unique(subset, axis=0, return_counts=True)
    order = np.argsort(-counts)

    breakdown = []
    other_count = 0
    for rank, i in enumerate(order):
        rgb = (int(uniq[i][0]), int(uniq[i][1]), int(uniq[i][2]))
        cnt = int(counts[i])
        if rank < MAX_CLASS_ROWS:
            name = swatch_lookup.get(rgb, f"Custom RGB{rgb}")
            breakdown.append({
                "name": name,
                "hex": "#%02X%02X%02X" % rgb,
                "point_count": cnt,
                "percentage": round(100.0 * cnt / total, 3) if total else 0.0,
            })
        else:
            other_count += cnt

    if other_count > 0:
        breakdown.append({
            "name": "Other",
            "hex": "",
            "point_count": other_count,
            "percentage": round(100.0 * other_count / total, 3) if total else 0.0,
        })

    return breakdown, int(len(uniq))


def compute_stats(app) -> AnnotationStats:
    """Snapshot of the currently loaded file (app.index) for the review log."""
    path = Path(app.files[app.index])
    total = int(app.cloud.n_points)

    original_path = None
    original_size = None
    original_mtime = None
    if app.orig_dir:
        cand = app.orig_dir / path.name
        if cand.exists():
            original_path = str(cand)
            st = cand.stat()
            original_size = int(st.st_size)
            original_mtime = _iso(st.st_mtime)

    ann_size = None
    ann_mtime = None
    if path.exists():
        st = path.stat()
        ann_size = int(st.st_size)
        ann_mtime = _iso(st.st_mtime)

    pts = np.asarray(app.cloud.points)
    bbox_min = pts.min(axis=0).tolist() if total else [0.0, 0.0, 0.0]
    bbox_max = pts.max(axis=0).tolist() if total else [0.0, 0.0, 0.0]
    bbox_extent = [hi - lo for lo, hi in zip(bbox_min, bbox_max)]
    centroid = pts.mean(axis=0).tolist() if total else [0.0, 0.0, 0.0]

    edited_mask = np.any(app.colors != app.original_colors, axis=1)
    annotated_count = int(edited_mask.sum())
    unannotated_count = total - annotated_count
    class_breakdown, distinct_colors = _class_breakdown(app, app.colors, edited_mask, total)

    gamma_val = 2 ** ((int(getattr(app, "_last_gamma_value", 100)) - 100) / 50.0)

    return AnnotationStats(
        filename=path.name,
        annotation_path=str(path),
        original_path=original_path,
        file_index=app.index + 1,
        total_files=len(app.files),
        file_format=path.suffix.lower(),
        annotation_file_size_bytes=ann_size,
        annotation_file_mtime=ann_mtime,
        original_file_size_bytes=original_size,
        original_file_mtime=original_mtime,
        point_count=total,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        bbox_extent=bbox_extent,
        centroid=centroid,
        annotated_point_count=annotated_count,
        annotated_percentage=round(100.0 * annotated_count / total, 3) if total else 0.0,
        unannotated_point_count=unannotated_count,
        unannotated_percentage=round(100.0 * unannotated_count / total, 3) if total else 0.0,
        distinct_annotation_colors=distinct_colors,
        class_breakdown=class_breakdown,
        status=annotation_status_label(app),
        is_dirty=app.index in getattr(app, "_dirty", set()),
        is_annotated=app.index in getattr(app, "_annotated", set()),
        is_visited=app.index in getattr(app, "_visited", set()),
        is_copied_from_original=app.index in getattr(app, "_copied_from_original", set()),
        brush_size_px=float(getattr(app, "brush_size", 0.0)),
        point_size_px=int(getattr(app, "point_size", 0)),
        annotation_alpha_pct=int(round(float(getattr(app, "annotation_alpha", 1.0)) * 100)),
        gamma_value=round(gamma_val, 3),
        render_points_as_spheres=bool(getattr(app, "_render_points_as_spheres", True)),
        stats_updated_at=_now_iso(),
    )
