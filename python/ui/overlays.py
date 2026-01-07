from __future__ import annotations


def position_overlays(app) -> None:
    # RIGHT (annotated) overlays
    if hasattr(app, "right_title") and hasattr(app, "plotter"):
        h1 = app.plotter.interactor.height()
        w1 = app.plotter.interactor.width()
        app.right_title.adjustSize()
        app.right_title.move((w1 - app.right_title.width()) // 2,
                             h1 - app.right_title.height() - 2)
        app.right_title.raise_()

    # LEFT (original) label
    if hasattr(app, "left_title") and hasattr(app, "plotter_ref") and app.plotter_ref.isVisible():
        h2 = app.plotter_ref.interactor.height()
        w2 = app.plotter_ref.interactor.width()
        app.left_title.adjustSize()
        app.left_title.move((w2 - app.left_title.width()) // 2,
                            h2 - app.left_title.height() - 2)
        app.left_title.raise_()
