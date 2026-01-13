#!/usr/bin/env python3
"""
Point Cloud Semantic Annotation Tool

This application allows semantic annotation of point clouds (PCD or PLY),
and remembers last folder/index between sessions.

Features:
  - Brush tool with adjustable size (press B then +/-)
  - Toggleable annotation mode (checkbox or 'A') to switch between navigation and painting
  - Undo/Redo (buttons + Ctrl/Cmd+Z, Ctrl/Cmd+Y)
  - Reset view (button + 'R')
  - Previous/Next navigation (buttons + Left/Right arrows)
  - Save annotations (Save button + Ctrl/Cmd+S)
  - Autosave annotations
  - Open folder via button
  - Maximized window with Top view or isometric initial view
  - Magenta circular cursor matching brush size in annotation mode
  - File counter and filename overlays fixed at bottom
  - State persistence (last folder & index) via JSON in script directory

Dependencies:
  pip install pyvista pyvistaqt scipy numpy PyQt5

Usage:
  python app.py
"""
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon

from configs.constants import (
    NAV_DOCK_WIDTH,
    NAV_NAME_MAX,
    NAV_THUMB_SIZE,
)
from services.thumbnail import ThumbnailService
from controllers import annotation, app_helpers, bootstrap, interaction, io, nav_ui, navigation, ui_controls
from rendering import camera
from ui.ribbon import build_ribbon
from ui.menu import build_menubar
from ui.icons import (
    icon_clone,
    icon_contrast,
    icon_eraser,
    icon_eye,
    icon_hist,
    icon_loop,
    icon_next,
    icon_palette,
    icon_pencil,
    icon_prev,
    icon_repair,
    icon_reset_contrast,
    icon_reset_view,
    icon_zoom,
)
from ui.nav_dock import (
    build_nav_dock,
    decorate_nav_item,
    make_nav_item_widget,
    nav_display_name,
)
from ui.overlays import position_overlays
from ui.layout import build_ui, install_ribbon_toolbar


class Annotator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        icon = Path(__file__).parent / 'icons' / 'app.png'
        if not icon.exists(): icon = Path(__file__).parent / 'icons' / 'app.ico'
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.setWindowTitle('Point Cloud Annotator')

        bootstrap.bootstrap(self)

    def _install_ribbon_toolbar(self):
        return install_ribbon_toolbar(self)

    @staticmethod
    def _natural_key(path):
        return io.natural_key(path)
    
    def _get_sorted_files(self):
        return io.get_sorted_files(self)

    def _compute_brush_idx(self, x, y):
        return annotation.compute_brush_idx(self, x, y)

    def _snap_camera(self, plotter):
        return camera.snap_camera(self, plotter)

    def _restore_camera(self, plotter, snap):
        return camera.restore_camera(self, plotter, snap)

    def _build_menubar(self):
        return build_menubar(self)

    def _icon_pencil(self):
        return icon_pencil(self)

    def _icon_eraser(self):
        return icon_eraser(self)

    def _icon_repair(self):
        return icon_repair(self)

    def _icon_clone(self):
        return icon_clone(self)

    def _icon_palette(self):
        return icon_palette(self)

    def _icon_contrast(self):
        return icon_contrast(self)

    def _icon_reset_view(self):
        return icon_reset_view(self)

    def _icon_hist(self):
        return icon_hist(self)

    def _icon_prev(self):
        return icon_prev(self)

    def _icon_next(self):
        return icon_next(self)

    def _icon_loop(self):
        return icon_loop(self)

    def _icon_reset_contrast(self):
        return icon_reset_contrast(self)

    def _icon_eye(self):
        return icon_eye(self)

    def _icon_zoom(self, plus=True):
        return icon_zoom(self, plus)

    def _build_ribbon(self):
        return build_ribbon(self)

    def _release_view_combo_focus(self):
        return app_helpers.release_view_combo_focus(self)

    def _on_nav_visibility_changed(self, visible: bool):
        return app_helpers.on_nav_visibility_changed(self, visible)

    def _build_nav_dock(self):
        return build_nav_dock(self)

    def _build_ui(self):
        return build_ui(self)

    def _set_view(self, idx: int):
        return camera.set_view(self, idx)

    def open_ann_folder(self):
        return io.open_ann_folder(self)

    def open_orig_folder(self):
        return io.open_orig_folder(self)

    def load_cloud(self):
        return io.load_cloud(self)

    def _position_overlays(self):
        return position_overlays(self)

    def _view_direction(self):
        return camera.view_direction(self)

    def _fit_view(self, plotter):
        return camera.fit_view(self, plotter)

    def _mesh_bounds_in_camera_xy(self, cam, mesh):
        return camera.mesh_bounds_in_camera_xy(self, cam, mesh)

    def _fit_shared_camera_once(self, mesh):
        return camera.fit_shared_camera_once(self, mesh)

    def _fit_to_canvas(self):
        return camera.fit_to_canvas(self)

    def resizeEvent(self, event):
        return camera.resize_event(self, event)

    def apply_view(self, idx: int = None):
        return camera.apply_view(self, idx)

    def reset_view(self):
        return camera.reset_view(self)

    def toggle_annotation(self):
        return annotation.toggle_annotation(self)

    def update_cursor(self):
        return annotation.update_cursor(self)

    def change_brush(self, val):
        return annotation.change_brush(self, val)

    def change_point(self, val):
        return annotation.change_point(self, val)

    def pick_color(self):
        return annotation.pick_color(self)

    def select_swatch(self, col, btn=None):
        return annotation.select_swatch(self, col, btn)

    def on_click(self, x, y):
        return annotation.on_click(self, x, y)

    def _maybe_autosave_before_nav(self):
        return navigation.maybe_autosave_before_nav(self)

    def on_prev(self):
        return navigation.on_prev(self)

    def on_next(self):
        return navigation.on_next(self)

    def on_first(self):
        return navigation.on_first(self)

    def on_last(self):
        return navigation.on_last(self)

    def on_page(self, delta: int):
        return navigation.on_page(self, delta)

    def _toggle_loop(self, on: bool):
        return navigation.toggle_loop(self, on)

    def _on_loop_tick(self):
        return navigation.on_loop_tick(self)

    def on_undo(self):
        return annotation.on_undo(self)

    def on_redo(self):
        return annotation.on_redo(self)

    def on_save(self, _autosave: bool = False):
        return io.on_save(self, _autosave)

    def _on_plus(self):
        return ui_controls.on_plus(self)

    def _on_minus(self):
        return ui_controls.on_minus(self)

    def on_zoom_in(self):
        return ui_controls.on_zoom_in(self)

    def on_zoom_out(self):
        return ui_controls.on_zoom_out(self)

    def eventFilter(self, obj, event):
        handled = interaction.event_filter(self, obj, event)
        if handled is None:
            return super().eventFilter(obj, event)
        return handled

    def _on_eraser_toggled(self, on: bool):
        return annotation.on_eraser_toggled(self, on)

    def reset_contrast(self):
        return annotation.reset_contrast(self)

    def on_gamma_change(self, val):
        return annotation.on_gamma_change(self, val)

    def apply_auto_contrast(self):
        return annotation.apply_auto_contrast(self)

    def show_histograms(self):
        return annotation.show_histograms(self)

    def set_annotations_visible(self, vis: bool):
        return annotation.set_annotations_visible(self, vis)

    def _current_base(self):
        return annotation.current_base(self)

    def _clone_source(self):
        return app_helpers.clone_source(self)

    def _on_toggle_ann_changed(self, state):
        return annotation.on_toggle_ann_changed(self, state)

    def on_alpha_change(self, val):
        return annotation.on_alpha_change(self, val)

    def update_annotation_visibility(self):
        return annotation.update_annotation_visibility(self)

    def toggle_repair_mode(self, on: bool):
        return annotation.toggle_repair_mode(self, on)

    def toggle_clone_mode(self, on: bool):
        return annotation.toggle_clone_mode(self, on)

    def closeEvent(self, e):
        return app_helpers.close_event(self, e)

    def changeEvent(self, event):
        app_helpers.on_change_event(self, event)
        return super().changeEvent(event)

    def _sync_renders(self):
        return camera.sync_renders(self)

    def _link_cameras(self):
        return camera.link_cameras(self)

    def _unlink_cameras(self):
        return camera.unlink_cameras(self)

    def _begin_batch(self):
        return camera.begin_batch(self)

    def _finalize_layout(self):
        return camera.finalize_layout(self)

    def _pre_fit_camera(self, mesh, plotter):
        return camera.pre_fit_camera(self, mesh, plotter)

    def _end_batch(self):
        return camera.end_batch(self)

    def _schedule_fit(self, delay=None):
        return camera.schedule_fit(self, delay)

    def _render_views_once(self):
        return camera.render_views_once(self)

    def _blend_into_mesh_subset(self, idx):
        return annotation.blend_into_mesh_subset(self, idx)

    def _zoom_at_cursor_for(self, plotter, x: int, y: int, delta_y: int):
        return camera.zoom_at_cursor_for(self, plotter, x, y, delta_y)

    def _on_nav_search_entered(self):
        return nav_ui.on_nav_search_entered(self)

    def _reset_nav_search(self):
        return nav_ui.reset_nav_search(self)

    def _nav_row_text(self, i: int) -> str:
        return nav_ui.nav_row_text(self, i)

    def _populate_nav_list(self):
        return nav_ui.populate_nav_list(self)

    def _sync_nav_selection(self):
        return nav_ui.sync_nav_selection(self)

    def _on_nav_row_changed(self, row: int):
        return nav_ui.on_nav_row_changed(self, row)

    def _nav_display_name(self, name: str) -> str:
        return nav_display_name(self, name)

    def _make_nav_item_widget(self, idx: int):
        return make_nav_item_widget(self, idx)

    def _scan_annotated_files(self):
        return nav_ui.scan_annotated_files(self)

    def _decorate_nav_item(self, idx: int):
        return decorate_nav_item(self, idx)

    def _mark_dirty_once(self):
        return nav_ui.mark_dirty_once(self)

    def _nudge_slider(self, slider, delta):
        return ui_controls.nudge_slider(self, slider, delta)

    def _update_loop_status(self):
        return nav_ui.update_loop_status(self)

    def _update_status_bar(self):
        return nav_ui.update_status_bar(self)

    def _restore_nav_width(self):
        return nav_ui.restore_nav_width(self, NAV_DOCK_WIDTH)

    def _on_ribbon_delay_changed(self, val: float):
        return ui_controls.on_ribbon_delay_changed(self, val)

    def _on_ribbon_alpha(self, v: int):
        return ui_controls.on_ribbon_alpha(self, v)

    def _on_ribbon_brush(self, v: int):
        return ui_controls.on_ribbon_brush(self, v)

    def _on_ribbon_point(self, v: int):
        return ui_controls.on_ribbon_point(self, v)

    def _on_ribbon_gamma(self, v: int):
        return ui_controls.on_ribbon_gamma(self, v)

    def _set_loop_delay(self, val: float):
        return navigation.set_loop_delay(self, val)

    def _is_split_mode(self) -> bool:
        return app_helpers.is_split_mode(self)

    def _shared_cam_active(self) -> bool:
        return app_helpers.shared_cam_active(self)

    def show_about_dialog(self):
        return app_helpers.show_about_dialog(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Annotator()
    win.show()
    sys.exit(app.exec_())
