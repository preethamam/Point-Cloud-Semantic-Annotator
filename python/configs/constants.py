from __future__ import annotations

from pathlib import Path

from appdirs import user_data_dir

APP_NAME = "Point Cloud Annotator"

STATE_DIR = Path(user_data_dir(APP_NAME, appauthor=False))
STATE_FILE = STATE_DIR / "state.json"
THUMB_DIR = STATE_DIR / "thumbs"

THUMB_SIZE = 96   # pixels (safe, fast, clean)
THUMB_MAX_PTS = 150_000   # cap for thumb generation (fast)
PERCENTAGE_CORE_FACTOR = 0.80
THUMB_BACKEND = "threading"
DEBUG_GUI_LOG = True

NAV_THUMB_SIZE = THUMB_SIZE
NAV_NAME_MAX = 30
NAV_DOCK_WIDTH = 155
RIBBON_ENH_VIEW_HEIGHT = 88
