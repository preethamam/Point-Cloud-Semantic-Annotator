from __future__ import annotations

import numpy as np


def maybe_autosave_before_nav(app) -> None:
    if app.act_autosave.isChecked():
        edited = (
            getattr(app, "_session_edited", None) is not None
            and np.any(app._session_edited)
        )
        if edited:
            app.on_save(_autosave=True)


def _reload_after_nav(app) -> None:
    app.history.clear()
    app.redo_stack.clear()
    app.load_cloud()
    app._position_overlays()
    app._sync_nav_selection()
    app._update_status_bar()


def on_prev(app) -> None:
    if not app.files:
        return

    maybe_autosave_before_nav(app)

    if app.index > 0:
        app.index -= 1
    else:
        app.index = len(app.files) - 1

    _reload_after_nav(app)


def on_next(app) -> None:
    if not app.files:
        return

    maybe_autosave_before_nav(app)

    if app.index < len(app.files) - 1:
        app.index += 1
    else:
        app.index = 0

    _reload_after_nav(app)


def on_first(app) -> None:
    if not app.files:
        return
    maybe_autosave_before_nav(app)
    app.index = 0
    _reload_after_nav(app)


def on_last(app) -> None:
    if not app.files:
        return
    maybe_autosave_before_nav(app)
    app.index = len(app.files) - 1
    _reload_after_nav(app)


def on_page(app, delta: int) -> None:
    """Jump by +/-N (default N=10 via PgUp/PgDown). Wraps like Next/Prev."""
    if not app.files:
        return
    maybe_autosave_before_nav(app)

    n = len(app.files)
    if n <= 0:
        return

    app.index = (app.index + int(delta)) % n
    _reload_after_nav(app)


def toggle_loop(app, on: bool) -> None:
    if on:
        delay_ms = int(app.loop_delay_sec * 1000)
        app._loop_timer.start(delay_ms)
    else:
        app._loop_timer.stop()

    app._update_loop_status()


def on_loop_tick(app) -> None:
    app.on_next()


def set_loop_delay(app, val: float) -> None:
    """
    Set loop delay in seconds.
    Safe to call from ribbon, menu, or dialogs.
    """
    app.loop_delay_sec = float(val)

    if app.act_loop.isChecked():
        toggle_loop(app, False)
        toggle_loop(app, True)

    app._update_status_bar()
