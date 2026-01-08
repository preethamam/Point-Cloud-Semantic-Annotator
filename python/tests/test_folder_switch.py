from __future__ import annotations

import argparse
import ctypes
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.constants import STATE_FILE, THUMB_DIR
from services.thumbnail import generate_thumbnail_job

DATASETS = {
    "A": {
        "orig": r"F:\Datasets\USCPavCon3D\Original\Pavement Cracks - Broken",
        "ann": r"C:\Users\Preetham\Downloads\pavement broken 4 set",
    },
    "B": {
        "orig": r"F:\Datasets\USCPavCon3D\Original\Concrete Cracks - Broken",
        "ann": r"C:\Users\Preetham\Downloads\Concrete Cracks - Broken Anno",
    },
    "C": {
        "orig": r"F:\Datasets\USCPavCon3D\Original\Concrete Cracks",
        "ann": r"C:\Users\Preetham\Downloads\501-500",
    },
}


def _get_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = list(folder.glob("*.ply")) + list(folder.glob("*.pcd"))
    return sorted(files)


def _thumb_key_for_path(src: Path) -> str:
    st = src.stat()
    import hashlib

    h = hashlib.sha1()
    h.update(str(src.resolve()).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()


def _expected_keys_for_dataset(name: str) -> tuple[list[Path], list[str], list[str]]:
    ds = DATASETS[name]
    ann_dir = Path(ds["ann"])
    orig_dir = Path(ds["orig"])
    ann_files = _get_files(ann_dir)

    missing_orig = []
    keys = []
    for ann in ann_files:
        orig = orig_dir / ann.name
        if not orig.exists():
            missing_orig.append(ann.name)
            continue
        keys.append(_thumb_key_for_path(orig))
    return ann_files, keys, missing_orig


def _generate_missing(keys: list[str], ann_files: list[Path], name: str) -> int:
    ds = DATASETS[name]
    orig_dir = Path(ds["orig"])
    created = 0
    for ann in ann_files:
        orig = orig_dir / ann.name
        if not orig.exists():
            continue
        key = _thumb_key_for_path(orig)
        out = THUMB_DIR / f"{key}.png"
        if out.exists():
            continue
        generate_thumbnail_job(orig, out)
        if out.exists():
            created += 1
    return created


def _thumb_count() -> int:
    if not THUMB_DIR.exists():
        return 0
    return len(list(THUMB_DIR.glob("*.png")))


def _write_state(ann_dir: Path, orig_dir: Path, index: int = 0) -> None:
    try:
        state = {}
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text())
        state.update({
            "annotation_dir": str(ann_dir),
            "original_dir": str(orig_dir),
            "index": int(index),
        })
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state))
    except Exception as exc:
        print(f"[WARN] Failed to update state.json: {exc}")


def _log_line(fp, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if fp is not None:
        fp.write(line + "\n")


def _show_error_dialog(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Folder Switch Test Failed", 0x10)
    except Exception:
        pass


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

    _log_line(report_fp, "Dataset summary:")
    union_keys: set[str] = set()
    missing_any = False
    for name in ("A", "B", "C"):
        ann_files, keys, missing = _expected_keys_for_dataset(name)
        union_keys.update(keys)
        _log_line(
            report_fp,
            f"- {name}: ann={len(ann_files)} orig_keys={len(keys)} missing_orig={len(missing)}",
        )
        if missing:
            missing_any = True
            _log_line(report_fp, f"  Missing originals (first 5): {missing[:5]}")

    _log_line(report_fp, f"Expected union thumbs: {len(union_keys)}")

    order = [args.start]
    rng = random.Random(args.seed)
    for _ in range(max(0, args.iterations - 1)):
        order.append(rng.choice(["A", "B", "C"]))

    if args.update_state:
        ds = DATASETS[order[0]]
        _write_state(Path(ds["ann"]), Path(ds["orig"]), 0)
        _log_line(report_fp, f"State updated to dataset {order[0]}")

    before_total = _thumb_count()
    _log_line(report_fp, f"Thumbs before: {before_total}")

    for i, name in enumerate(order, 1):
        ann_files, keys, missing = _expected_keys_for_dataset(name)
        created = _generate_missing(keys, ann_files, name)
        after = _thumb_count()
        _log_line(
            report_fp,
            f"Switch {i}/{len(order)} -> {name}: created={created} total={after} missing_orig={len(missing)}",
        )
        time.sleep(0.2)

    after_total = _thumb_count()
    _log_line(report_fp, f"Thumbs after: {after_total}")
    _log_line(report_fp, f"Expected union thumbs: {len(union_keys)}")

    existing = {p.stem for p in THUMB_DIR.glob("*.png")}
    missing_keys = union_keys - existing
    extra_keys = existing - union_keys

    fail_reasons = []
    if missing_any:
        fail_reasons.append("Missing originals for one or more datasets.")
    if after_total != len(union_keys):
        fail_reasons.append("Cache size mismatch.")
    if missing_keys:
        fail_reasons.append(f"Missing expected thumbs: {len(missing_keys)}")
    if extra_keys:
        fail_reasons.append(f"Extra thumbs not expected: {len(extra_keys)}")

    if missing_keys:
        _log_line(report_fp, f"[FAIL] Missing expected thumbs: {len(missing_keys)}")
        for stem in list(missing_keys)[:10]:
            _log_line(report_fp, f"  missing: {stem}")
    if extra_keys:
        _log_line(report_fp, f"[FAIL] Extra thumbs not expected: {len(extra_keys)}")
        for stem in list(extra_keys)[:10]:
            _log_line(report_fp, f"  extra: {stem}")

    if args.update_state:
        ds = DATASETS[order[-1]]
        _write_state(Path(ds["ann"]), Path(ds["orig"]), 0)
        _log_line(report_fp, f"State updated to dataset {order[-1]}")

    if report_fp is not None:
        report_fp.close()

    if fail_reasons:
        _show_error_dialog("\n".join(fail_reasons))
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
