from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler

from configs.constants import (
    DEBUG_GUI_LOG,
    GUI_LOG_BACKUPS,
    GUI_LOG_MAX_BYTES,
    STATE_DIR,
    STATE_FILE,
)

STATE_DIR.mkdir(parents=True, exist_ok=True)

_GUI_LOGGER = None


def _get_gui_logger() -> logging.Logger:
    global _GUI_LOGGER
    if _GUI_LOGGER is not None:
        return _GUI_LOGGER

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / "gui_debug.log"

    logger = logging.getLogger("gui_debug")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = RotatingFileHandler(
            log_path,
            maxBytes=GUI_LOG_MAX_BYTES,
            backupCount=GUI_LOG_BACKUPS,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _GUI_LOGGER = logger
    return logger


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
        logger = _get_gui_logger()
        logger.info(message)
    except Exception:
        pass
