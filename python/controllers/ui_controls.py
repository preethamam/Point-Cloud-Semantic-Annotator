from __future__ import annotations


def nudge_slider(app, slider, delta) -> None:
    if not slider.isEnabled():
        return
    v = slider.value() + delta
    v = max(slider.minimum(), min(slider.maximum(), v))
    slider.setValue(v)


def on_plus(app) -> None:
    if app._waiting == "brush":
        nudge_slider(app, app.ribbon_sliders["brush"][0], +2)
    elif app._waiting == "point":
        nudge_slider(app, app.ribbon_sliders["point"][0], +1)
    elif app._waiting == "alpha":
        nudge_slider(app, app.ribbon_sliders["alpha"][0], +5)
    elif app._waiting == "gamma":
        nudge_slider(app, app.ribbon_sliders["gamma"][0], +5)
    elif app._waiting == "zoom":
        app.plotter.camera.Zoom(1.1)
        app.plotter.render()


def on_minus(app) -> None:
    if app._waiting == "brush":
        nudge_slider(app, app.ribbon_sliders["brush"][0], -2)
    elif app._waiting == "point":
        nudge_slider(app, app.ribbon_sliders["point"][0], -1)
    elif app._waiting == "alpha":
        nudge_slider(app, app.ribbon_sliders["alpha"][0], -5)
    elif app._waiting == "gamma":
        nudge_slider(app, app.ribbon_sliders["gamma"][0], -5)
    elif app._waiting == "zoom":
        app.plotter.camera.Zoom(0.9)
        app.plotter.render()


def on_zoom_in(app) -> None:
    app._waiting = "zoom"
    on_plus(app)


def on_zoom_out(app) -> None:
    app._waiting = "zoom"
    on_minus(app)


def on_ribbon_delay_changed(app, val: float) -> None:
    app.loop_delay_sec = float(val)
    if app.act_loop.isChecked():
        app._toggle_loop(False)
        app._toggle_loop(True)
    app._update_status_bar()


def on_ribbon_alpha(app, v: int) -> None:
    _, lbl = app.ribbon_sliders["alpha"]
    lbl.setText(f"{int(v)}%")
    app.on_alpha_change(v)


def on_ribbon_brush(app, v: int) -> None:
    _, lbl = app.ribbon_sliders["brush"]
    lbl.setText(f"{int(v)} px")
    app.change_brush(v)


def on_ribbon_point(app, v: int) -> None:
    _, lbl = app.ribbon_sliders["point"]
    lbl.setText(f"{int(v)} px")
    app.change_point(v)


def on_ribbon_gamma(app, v: int) -> None:
    app._last_gamma_value = int(v)
    app.on_gamma_change(v)
