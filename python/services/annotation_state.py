from __future__ import annotations

from pathlib import Path


def is_annotated_pair(ann_path: Path, orig_path: Path) -> bool:
    try:
        import numpy as np
        import pyvista as pv

        pc_a = pv.read(str(ann_path))
        pc_o = pv.read(str(orig_path))

        if pc_a.n_points != pc_o.n_points:
            return True

        if "RGB" not in pc_a.array_names or "RGB" not in pc_o.array_names:
            return False

        return not np.array_equal(pc_a["RGB"], pc_o["RGB"])
    except Exception:
        return False
