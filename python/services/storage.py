from __future__ import annotations

import json
from datetime import datetime

from configs.constants import DEBUG_GUI_LOG, STATE_DIR, STATE_FILE

STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    try:
        if not STATE_FILE.exists():
            return {}
        state = json.loads(STATE_FILE.read_text())
        if not isinstance(state, dict):
            return {}
        if "project_pairs" not in state:
            ann = state.get("annotation_dir", "")
            org = state.get("original_dir", "")
            if ann and org:
                state["project_pairs"] = {ann: org}
                STATE_FILE.write_text(json.dumps(state))
        return state
    except Exception:
        return {}


def save_state(updates: dict) -> None:
    try:
        state = load_state()
        state.update(updates)
        STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def load_nav_dock_width(default_width: int) -> int:
    try:
        state = load_state()
        return int(state.get("nav_dock_width", default_width))
    except Exception:
        return default_width


def log_gui(message: str) -> None:
    try:
        if not DEBUG_GUI_LOG:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        log_path = STATE_DIR / "gui_debug.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass
