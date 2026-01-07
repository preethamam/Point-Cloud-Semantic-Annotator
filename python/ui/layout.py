from __future__ import annotations

from PyQt5 import QtCore, QtWidgets
from pyvistaqt import QtInteractor


def install_ribbon_toolbar(app) -> None:
    """Put the ribbon in QMainWindow's toolbar area so docks sit below it."""
    app.ribbon = app._build_ribbon()

    app.ribbon_tb = QtWidgets.QToolBar("Ribbon", app)
    app.ribbon_tb.setMovable(False)
    app.ribbon_tb.setFloatable(False)
    app.ribbon_tb.setAllowedAreas(QtCore.Qt.TopToolBarArea)
    app.ribbon_tb.setContentsMargins(0, 0, 0, 0)

    app.ribbon_tb.setIconSize(QtCore.QSize(16, 16))
    app.ribbon_tb.addWidget(app.ribbon)

    app.addToolBar(QtCore.Qt.TopToolBarArea, app.ribbon_tb)


def build_ui(app) -> None:
    """
    Build ONLY the visual layout:
    - Two 3D viewports
    - Text overlays

    ALL interaction logic is handled via:
    - QAction (menu + shortcuts)
    - Navigation dock
    """
    w = QtWidgets.QWidget(app)
    app.setCentralWidget(w)

    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)

    app.plotter_ref = QtInteractor(app)
    app.plotter_ref.set_background("white")
    app.plotter_ref.setVisible(False)
    lay.addWidget(app.plotter_ref.interactor, stretch=4)

    app.vline = QtWidgets.QFrame()
    app.vline.setFrameShape(QtWidgets.QFrame.VLine)
    app.vline.setFrameShadow(QtWidgets.QFrame.Sunken)
    app.vline.setFixedWidth(8)
    app.vline.setStyleSheet("color: #cfcfcf;")
    app.vline.setVisible(False)
    lay.addWidget(app.vline)

    app.plotter = QtInteractor(app)
    app.plotter.set_background("white")
    lay.addWidget(app.plotter.interactor, stretch=4)

    try:
        app.plotter.ren_win.SetMultiSamples(8)
        app.plotter_ref.ren_win.SetMultiSamples(8)
    except Exception:
        pass

    app.left_title = QtWidgets.QLabel(app.plotter_ref.interactor)
    app.left_title.setAutoFillBackground(True)
    app.left_title.setStyleSheet(
        "color:black; font-weight:bold; "
        "background-color:white; font-size:14px;"
    )
    app.left_title.setText("Original Point Cloud")
    app.left_title.hide()

    app.right_title = QtWidgets.QLabel(app.plotter.interactor)
    app.right_title.setAutoFillBackground(True)
    app.right_title.setStyleSheet(
        "color:black; font-weight:bold; "
        "background-color:white; font-size:14px;"
    )
    app.right_title.setAlignment(QtCore.Qt.AlignCenter)
    app.right_title.setText("Annotated Point Cloud")
    app.right_title.show()
