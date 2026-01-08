from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.constants import STATE_FILE, THUMB_DIR
from services.storage import load_state, save_state
from services.thumbnail import generate_thumbnail_job

DATASETS = {
    "A": {
        "orig": r"F:\Datasets\USCPavCon3D\Original\Pavement Cracks - Broken",
        "ann": r"F:\Datasets\USCPavCon3D\Ajay Annotated Running\pavement broken 4 set",
    },
    "B": {
        "orig": r"F:\Datasets\USCPavCon3D\Original\Concrete Cracks - Broken",
        "ann": r"F:\Datasets\USCPavCon3D\Ajay Annotated Running\Concrete Cracks - Broken Anno",
    },
    "C": {
        "orig": r"F:\Datasets\USCPavCon3D\Original\Concrete Cracks",
        "ann": r"F:\Datasets\USCPavCon3D\Ajay Annotated Running\501-500",
    },
}


def _log(msg: str) -> None:
    print(msg)


def _log_line(fp, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if fp is not None:
        fp.write(line + "\n")


def _natural_key(path: Path):
    import re
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", path.name)]


def _get_sorted_files(folder: Path) -> list[Path]:
    files = list(folder.glob("*.ply")) + list(folder.glob("*.pcd"))
    return sorted(files, key=_natural_key)


def _thumb_key_for_path(src: Path) -> str:
    import hashlib
    st = src.stat()
    h = hashlib.sha1()
    h.update(str(src.resolve()).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def _thumb_out_for(ann_path: Path, orig_dir: Path | None) -> tuple[Path, Path]:
    if orig_dir is not None:
        orig = orig_dir / ann_path.name
        if orig.exists():
            return orig, THUMB_DIR / f"{_thumb_key_for_path(orig)}.png"
    return ann_path, THUMB_DIR / f"{_thumb_key_for_path(ann_path)}.png"


def _thumb_count() -> int:
    if not THUMB_DIR.exists():
        return 0
    return len(list(THUMB_DIR.glob("*.png")))


def _update_state(ctx, update_state: bool) -> None:
    if not update_state:
        return
    st = load_state()
    nav_w = int(st.get("nav_dock_width", 155))
    save_state({
        "annotation_dir": str(ctx["ann_dir"] or ""),
        "original_dir": str(ctx["orig_dir"] or ""),
        "index": 0,
        "nav_dock_width": nav_w,
        "last_ann_dir": str(ctx["ann_dir"] or ""),
        "last_orig_dir": str(ctx["orig_dir"] or ""),
        "project_pairs": ctx["project_pairs"],
    })


def _generate_thumbs(ctx) -> int:
    if not ctx["files"]:
        return 0
    created = 0
    for ann in ctx["files"]:
        src, out_png = _thumb_out_for(ann, ctx["orig_dir"])
        if out_png.exists():
            continue
        generate_thumbnail_job(src, out_png)
        if out_png.exists():
            created += 1
    _log(f"thumbs: created={created} total={_thumb_count()}")
    return created


def _open_orig(ctx, path: Path, update_state: bool, log_fp) -> None:
    ctx["orig_dir"] = path
    if ctx["files"]:
        has_full_match = all((path / p.name).exists() for p in ctx["files"])
        if not has_full_match:
            ctx["ann_dir"] = None
            ctx["files"] = []
    else:
        ctx["pending_orig"] = path
    _update_state(ctx, update_state)
    _log_line(log_fp, f"open_orig: {path} ann_dir={ctx['ann_dir']}")


def _open_ann(ctx, path: Path, update_state: bool, log_fp) -> int:
    ctx["ann_dir"] = path
    ctx["files"] = _get_sorted_files(path)
    if ctx["pending_orig"] is not None:
        cand = ctx["pending_orig"]
        ctx["pending_orig"] = None
        has_full_match = all((cand / p.name).exists() for p in ctx["files"])
        ctx["orig_dir"] = cand if has_full_match else None
    else:
        cand = ctx["project_pairs"].get(str(path), "")
        ctx["orig_dir"] = Path(cand) if cand else None
        if ctx["orig_dir"] is not None:
            has_full_match = all((ctx["orig_dir"] / p.name).exists() for p in ctx["files"])
            if not has_full_match:
                ctx["orig_dir"] = None
    if ctx["ann_dir"] and ctx["orig_dir"]:
        ctx["project_pairs"][str(ctx["ann_dir"])] = str(ctx["orig_dir"])
    _update_state(ctx, update_state)
    _log_line(log_fp, f"open_ann: {path} orig_dir={ctx['orig_dir']} files={len(ctx['files'])}")
    return _generate_thumbs(ctx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--update-state", default=True, action="store_true")
    parser.add_argument("--start", choices=["A", "B", "C"], default="A")
    parser.add_argument("--report", type=str, default="tests/folder_switch_report.txt")
    args = parser.parse_args()

    if THUMB_DIR.exists():
        shutil.rmtree(THUMB_DIR)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report) if args.report else None
    report_fp = None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_fp = report_path.open("w", encoding="utf-8")

    ctx = {
        "ann_dir": None,
        "orig_dir": None,
        "files": [],
        "pending_orig": None,
        "project_pairs": {},
    }

    order = [args.start]
    rng = random.Random(args.seed)
    for _ in range(max(0, args.iterations - 1)):
        order.append(rng.choice(["A", "B", "C"]))

    _log_line(report_fp, f"switch_order: {','.join(order)}")
    total_created = 0
    for key in order:
        ds = DATASETS[key]
        _open_orig(ctx, Path(ds["orig"]), args.update_state, report_fp)
        total_created += _open_ann(ctx, Path(ds["ann"]), args.update_state, report_fp)

    final_total = _thumb_count()
    _log_line(report_fp, f"final thumbs: {final_total} state={STATE_FILE}")
    _log_line(report_fp, f"total created: {total_created}")
    if final_total > 0:
        _log_line(report_fp, "PASS")
    else:
        _log_line(report_fp, "FAIL")
    if report_fp is not None:
        report_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
