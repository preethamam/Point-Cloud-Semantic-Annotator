from __future__ import annotations

import json

from configs.constants import STATE_DIR, STATE_FILE

STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    try:
        if not STATE_FILE.exists():
            return {}
        return json.loads(STATE_FILE.read_text())
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
