from __future__ import annotations

import numpy as np
from PyQt5 import QtCore
from vtkmodules.vtkRenderingCore import vtkCamera


def snap_camera(app, plotter):
    cam = plotter.camera
    try:
        return dict(
            pos=np.array(cam.GetPosition(), dtype=float),
            fp=np.array(cam.GetFocalPoint(), dtype=float),
            vu=np.array(cam.GetViewUp(), dtype=float),
            pp=bool(cam.GetParallelProjection()),
            ps=float(cam.GetParallelScale()),
            va=float(cam.GetViewAngle()),
        )
    except Exception:
        return None


def restore_camera(app, plotter, snap):
    if not snap:
        return
    cam = plotter.camera
    cam.SetPosition(*snap["pos"])
    cam.SetFocalPoint(*snap["fp"])
    cam.SetViewUp(*snap["vu"])
    if snap["pp"]:
        cam.ParallelProjectionOn()
        cam.SetParallelScale(snap["ps"])
    else:
        cam.ParallelProjectionOff()
        cam.SetViewAngle(snap["va"])


def set_view(app, idx: int) -> None:
    app.current_view = idx
    if hasattr(app, "ribbon_view_combo"):
        combo = app.ribbon_view_combo
        try:
            combo.blockSignals(True)
            pos = combo.findData(idx)
            if pos >= 0:
                combo.setCurrentIndex(pos)
        finally:
            combo.blockSignals(False)
    apply_view(app, idx)


def view_direction(app):
    if app.current_view == 0:
        return np.array([0.0, 0.0, -1.0])
    if app.current_view == 1:
        return np.array([0.0, 0.0, 1.0])
    if app.current_view == 2:
        return np.array([0.0, 1.0, 0.0])
    if app.current_view == 3:
        return np.array([0.0, -1.0, 0.0])
    if app.current_view == 4:
        return np.array([1.0, 0.0, 0.0])
    if app.current_view == 5:
        return np.array([-1.0, 0.0, 0.0])
    if app.current_view == 6:
        v = np.array([1.0, 1.0, -1.0])
    elif app.current_view == 7:
        v = np.array([-1.0, 1.0, -1.0])
    elif app.current_view == 8:
        v = np.array([1.0, -1.0, -1.0])
    else:
        v = np.array([-1.0, -1.0, -1.0])
    return v / max(np.linalg.norm(v), 1e-6)


def fit_view(app, plotter):
    """Fit camera of a given plotter to its mesh bounds without changing orientation."""
    if getattr(app, "_is_closing", False) or getattr(app, "_batch", False):
        return
    if plotter is None or plotter.renderer is None:
        return

    mesh = None
    if plotter is app.plotter and hasattr(app, "cloud"):
        mesh = app.cloud
    elif plotter is app.plotter_ref and hasattr(app, "cloud_ref"):
        mesh = app.cloud_ref
    if mesh is None or mesh.n_points == 0:
        return

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    xr, yr, zr = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    r = 0.5 * np.linalg.norm([xr, yr, zr])
    if r <= 0:
        return

    cam = plotter.camera
    w = max(1, plotter.interactor.width())
    h = max(1, plotter.interactor.height())
    aspect = w / float(h)

    dirp = np.array(cam.GetDirectionOfProjection(), dtype=float)
    if not np.isfinite(dirp).all() or np.linalg.norm(dirp) < 1e-6:
        dirp = view_direction(app)
    dirp /= np.linalg.norm(dirp)

    if cam.GetParallelProjection():
        half_h = 0.5 * yr
        half_w = 0.5 * xr
        scale_needed = max(half_h, half_w / max(aspect, 1e-6))
        cam.SetFocalPoint(cx, cy, cz)
        pos = np.array(cam.GetPosition(), dtype=float)
        if not np.isfinite(pos).all():
            pos = np.array([cx, cy, cz]) - dirp * (r * 2.0 + 1.0)
        cam.SetPosition(*pos)
        cam.SetParallelScale(scale_needed * app._fit_pad)
    else:
        vfov = np.deg2rad(cam.GetViewAngle())
        hfov = 2 * np.arctan(np.tan(vfov / 2) * aspect)
        eff = min(vfov, hfov)
        dist = r / np.tan(eff / 2) * app._fit_pad
        pos = np.array([cx, cy, cz]) - dirp * dist
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(*pos)

    plotter.reset_camera_clipping_range()
    if (not getattr(app, "_is_closing", False)
            and not getattr(app, "_batch", False)
            and not getattr(app, "_view_change_active", False)):
        plotter.render()


def mesh_bounds_in_camera_xy(app, cam, mesh):
    """Return (half_w, half_h) of mesh bounds measured in camera coords (x,y)."""
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    corners = np.array([
        [xmin, ymin, zmin, 1.0],
        [xmin, ymin, zmax, 1.0],
        [xmin, ymax, zmin, 1.0],
        [xmin, ymax, zmax, 1.0],
        [xmax, ymin, zmin, 1.0],
        [xmax, ymin, zmax, 1.0],
        [xmax, ymax, zmin, 1.0],
        [xmax, ymax, zmax, 1.0],
    ], dtype=float)

    M = cam.GetViewTransformMatrix()
    mat = np.array([[M.GetElement(r, c) for c in range(4)] for r in range(4)], dtype=float)

    cam_pts = (mat @ corners.T).T
    x = cam_pts[:, 0]
    y = cam_pts[:, 1]

    half_w = 0.5 * float(x.max() - x.min())
    half_h = 0.5 * float(y.max() - y.min())
    return half_w, half_h


def fit_shared_camera_once(app, mesh):
    """Fit the shared vtkCamera so the object fits in both panes."""
    if mesh is None or mesh.n_points == 0:
        return
    if app._shared_camera is None:
        return

    cam = app._shared_camera

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    center = np.array([cx, cy, cz], dtype=float)

    dop = np.array(cam.GetDirectionOfProjection(), dtype=float)
    n = float(np.linalg.norm(dop))
    if not np.isfinite(dop).all() or n < 1e-9:
        dop = np.array([0.0, 0.0, -1.0], dtype=float)
        n = 1.0
    dop /= n

    half_w, half_h = mesh_bounds_in_camera_xy(app, cam, mesh)

    w1 = max(1, int(app.plotter.interactor.width()))
    h1 = max(1, int(app.plotter.interactor.height()))
    a1 = w1 / float(h1)

    w2 = max(1, int(app.plotter_ref.interactor.width()))
    h2 = max(1, int(app.plotter_ref.interactor.height()))
    a2 = w2 / float(h2)

    pad = float(getattr(app, "_fit_pad", 1.10))

    cam.SetFocalPoint(cx, cy, cz)

    if cam.GetParallelProjection():
        s1 = max(half_h, half_w / max(a1, 1e-6))
        s2 = max(half_h, half_w / max(a2, 1e-6))
        scale = max(s1, s2) * pad

        pos = np.array(cam.GetPosition(), dtype=float)
        if not np.isfinite(pos).all():
            r = float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])) or 1.0
            pos = center - dop * (r * 2.0 + 1.0)

        cam.SetPosition(*pos)
        cam.SetParallelScale(scale)

    else:
        vfov = np.deg2rad(float(cam.GetViewAngle()))
        vfov = max(vfov, 1e-3)

        def dist_needed(aspect):
            hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * max(aspect, 1e-6))
            hfov = max(hfov, 1e-3)
            d_h = half_h / np.tan(vfov / 2.0)
            d_w = half_w / np.tan(hfov / 2.0)
            return max(d_h, d_w)

        dist = max(dist_needed(a1), dist_needed(a2)) * pad
        cam.SetPosition(*(center - dop * dist))


def fit_to_canvas(app):
    if getattr(app, "_is_closing", False) or getattr(app, "_batch", False):
        return

    split = app._is_split_mode()
    shared = bool(split and app._shared_camera is not None)

    if shared:
        mesh = getattr(app, "cloud", None)
        if mesh is None or mesh.n_points == 0:
            mesh = getattr(app, "cloud_ref", None)
        if mesh is None:
            return

        fit_shared_camera_once(app, mesh)

        try:
            app.plotter.reset_camera_clipping_range()
            app.plotter_ref.reset_camera_clipping_range()
        except Exception:
            pass

        if getattr(app, "_view_change_active", False):
            app._view_change_active = False
            sync_renders(app)
        else:
            sync_renders(app)
        return

    if hasattr(app, "plotter"):
        fit_view(app, app.plotter)
    if hasattr(app, "plotter_ref") and app.plotter_ref.isVisible():
        fit_view(app, app.plotter_ref)
    if getattr(app, "_view_change_active", False):
        app._view_change_active = False
        render_views_once(app)


def resize_event(app, event):
    super(type(app), app).resizeEvent(event)
    if hasattr(app, "right_title"):
        app._position_overlays()
    if not getattr(app, "_batch", False):
        schedule_fit(app)


def apply_view(app, idx: int = None):
    """
    Top view -> orthographic, look straight down +Z with +Y up.
    Isometric -> perspective, SOUTH-WEST isometric (from -X,-Y, +Z) with Z up.
    """
    topdown = (app.current_view == 0)
    app._view_change_active = True
    app._cam_pause = True
    views = [getattr(app, "plotter", None), getattr(app, "plotter_ref", None)]
    for view in views:
        if view:
            try:
                view.interactor.setUpdatesEnabled(False)
            except Exception:
                pass

    def _apply(plotter, mesh, topdown_local: bool):
        if plotter is None or mesh is None or mesh.n_points == 0:
            return
        cam = plotter.camera
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0

        if topdown_local:
            cam.ParallelProjectionOn()
            cam.SetViewUp(0, 1, 0)
            dop = np.array([0.0, 0.0, -1.0])
        elif app.current_view == 1:
            cam.ParallelProjectionOn()
            cam.SetViewUp(0, 1, 0)
            dop = np.array([0.0, 0.0, 1.0])
        elif app.current_view == 2:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([0.0, 1.0, 0.0])
        elif app.current_view == 3:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([0.0, -1.0, 0.0])
        elif app.current_view == 4:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([1.0, 0.0, 0.0])
        elif app.current_view == 5:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([-1.0, 0.0, 0.0])
        elif app.current_view == 6:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([1.0, 1.0, -1.0])
            dop /= np.linalg.norm(dop)
        elif app.current_view == 7:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([-1.0, 1.0, -1.0])
            dop /= np.linalg.norm(dop)
        elif app.current_view == 8:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([1.0, -1.0, -1.0])
            dop /= np.linalg.norm(dop)
        elif app.current_view == 9:
            cam.ParallelProjectionOff()
            cam.SetViewUp(0, 0, 1)
            dop = np.array([-1.0, -1.0, -1.0])
            dop /= np.linalg.norm(dop)

        cam.SetFocalPoint(cx, cy, cz)

        r = 0.5 * np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]) or 1.0
        w = max(1, plotter.interactor.width())
        h = max(1, plotter.interactor.height())
        vfov = np.deg2rad(cam.GetViewAngle())
        hfov = 2 * np.arctan(np.tan(vfov / 2) * (w / float(h)))
        eff = min(vfov, hfov)
        dist = r / np.tan(max(eff, 1e-3) / 2) * float(getattr(app, "_fit_pad", 1.12))
        pos = np.array([cx, cy, cz]) - dop * dist
        cam.SetPosition(*pos)

        plotter.reset_camera_clipping_range()

    try:
        _apply(app.plotter, getattr(app, "cloud", None), topdown)
        if hasattr(app, "plotter_ref") and app.plotter_ref.isVisible():
            _apply(app.plotter_ref, getattr(app, "cloud_ref", None), topdown)
    finally:
        for view in views:
            if view:
                try:
                    view.interactor.setUpdatesEnabled(True)
                except Exception:
                    pass
        app._cam_pause = False

    schedule_fit(app)


def reset_view(app) -> None:
    app.plotter.reset_camera()
    apply_view(app)


def sync_renders(app) -> None:
    """Render both views safely (avoid recursion / closing / batching)."""
    if (app._cam_syncing or getattr(app, "_is_closing", False)
            or getattr(app, "_batch", False) or getattr(app, "_cam_pause", False)):
        return
    app._cam_syncing = True
    try:
        if hasattr(app, "plotter"):
            app.plotter.render()
        if hasattr(app, "plotter_ref") and app.plotter_ref.isVisible():
            app.plotter_ref.render()
    finally:
        app._cam_syncing = False


def link_cameras(app) -> None:
    """Make both panels share the same vtkCamera and keep renders in sync."""
    if not hasattr(app, "plotter") or not hasattr(app, "plotter_ref"):
        return
    cam = app.plotter.renderer.GetActiveCamera()
    app.plotter_ref.renderer.SetActiveCamera(cam)
    if app._cam_observer_id is None:
        app._shared_camera = cam
        app._cam_observer_id = cam.AddObserver("ModifiedEvent", lambda *_: sync_renders(app))
    sync_renders(app)


def unlink_cameras(app) -> None:
    """Detach the right panel from the shared camera (but keep current view)."""
    if app._shared_camera is not None and app._cam_observer_id is not None:
        try:
            app._shared_camera.RemoveObserver(app._cam_observer_id)
        except Exception:
            pass
    app._cam_observer_id = None
    try:
        if hasattr(app, "plotter_ref"):
            new_cam = vtkCamera()
            new_cam.DeepCopy(app.plotter.renderer.GetActiveCamera())
            app.plotter_ref.renderer.SetActiveCamera(new_cam)
    except Exception:
        pass
    app._shared_camera = None


def begin_batch(app) -> None:
    app._batch = True
    app._cam_pause = True

    app._cam_snap_l = snap_camera(app, app.plotter)
    app._cam_snap_r = None if app._shared_cam_active() else (
        snap_camera(app, app.plotter_ref) if app.plotter_ref.isVisible() else None
    )

    for view in [getattr(app, "plotter", None), getattr(app, "plotter_ref", None)]:
        if view:
            view.interactor.setUpdatesEnabled(False)


def finalize_layout(app) -> None:
    if getattr(app, "_is_closing", False):
        return

    app._cam_pause = True

    split = app._is_split_mode()

    try:
        if hasattr(app, "_cam_snap_l") and app._cam_snap_l:
            restore_camera(app, app.plotter, app._cam_snap_l)

        if split:
            cam = app.plotter.renderer.GetActiveCamera()
            app.plotter_ref.renderer.SetActiveCamera(cam)
            app._shared_camera = cam

            if hasattr(app, "_cam_snap_l") and app._cam_snap_l:
                restore_camera(app, app.plotter, app._cam_snap_l)

            if getattr(app, "_need_split_fit", False):
                fit_to_canvas(app)
                app._need_split_fit = False

            app.plotter.reset_camera_clipping_range()
            app.plotter_ref.reset_camera_clipping_range()

        else:
            apply_view(app)
            try:
                fit_to_canvas(app)
            except Exception:
                pass

        app._position_overlays()

    finally:
        app._cam_pause = False

    sync_renders(app)

    app._cam_snap_l = None
    app._cam_snap_r = None


def pre_fit_camera(app, mesh, plotter) -> None:
    """
    Set camera orientation + an initial distance/scale using mesh bounds,
    before any actors are added. No render here.
    """
    if mesh is None or mesh.n_points == 0 or plotter is None:
        return

    cam = plotter.camera
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    xr, yr, zr = (xmax - xmin), (ymax - ymin), (zmax - zmin)
    r = 0.5 * float(np.linalg.norm([xr, yr, zr])) or 1.0

    w = max(1, plotter.interactor.width())
    h = max(1, plotter.interactor.height())
    aspect = w / float(h)
    pad = float(getattr(app, "_fit_pad", 1.12))

    is_parallel = app.current_view in (0, 1)
    dop = view_direction(app)
    dop /= max(np.linalg.norm(dop), 1e-6)

    if is_parallel:
        cam.ParallelProjectionOn()
        cam.SetViewUp(0, 1, 0)
        scale_h = 0.5 * yr
        scale_w = 0.5 * xr / max(aspect, 1e-6)
        cam.SetParallelScale(max(scale_h, scale_w) * pad)
        pos = np.array([cx, cy, cz]) - dop * (r * 2.0 + 1.0)
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(*pos)
    else:
        cam.ParallelProjectionOff()
        cam.SetViewUp(0, 0, 1)
        vfov = np.deg2rad(cam.GetViewAngle())
        hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect)
        eff = max(1e-3, min(vfov, hfov))
        dist = r / np.tan(eff / 2.0) * pad
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(*(np.array([cx, cy, cz]) - dop * dist))


def end_batch(app) -> None:
    for view in [getattr(app, "plotter", None), getattr(app, "plotter_ref", None)]:
        if view:
            view.interactor.setUpdatesEnabled(True)

    app._batch = False
    app._cam_pause = False
    QtCore.QTimer.singleShot(0, lambda: finalize_layout(app))


def schedule_fit(app, delay=None) -> None:
    """Coalesce multiple fit requests into one."""
    if getattr(app, "_is_closing", False):
        return
    if getattr(app, "_batch", False):
        return
    if delay is None:
        delay = getattr(app, "_fit_delay_ms", 33)
    app._fit_timer.stop()
    app._fit_timer.start(int(delay))


def render_views_once(app) -> None:
    if getattr(app, "_is_closing", False) or getattr(app, "_batch", False):
        return
    try:
        if hasattr(app, "plotter"):
            app.plotter.render()
        if hasattr(app, "plotter_ref") and app.plotter_ref.isVisible():
            app.plotter_ref.render()
    except Exception:
        pass


def zoom_at_cursor_for(app, plotter, x: int, y: int, delta_y: int) -> None:
    """
    Fluid, infinite zoom anchored at the cursor for a given plotter (left or right).
    Atomic when cameras are linked to avoid shake.
    """
    if plotter is None:
        return
    ren = plotter.renderer
    inter = plotter.interactor
    cam = plotter.camera
    H = inter.height()

    atomic = bool((app.repair_mode or app.clone_mode) and app._shared_camera is not None)

    if atomic:
        if app._in_zoom:
            return
        app._in_zoom = True
        app._cam_pause = True
        try:
            app.plotter.interactor.setUpdatesEnabled(False)
        except Exception:
            pass
        try:
            if hasattr(app, "plotter_ref"):
                app.plotter_ref.interactor.setUpdatesEnabled(False)
        except Exception:
            pass

    try:
        def ray_through_xy(renderer, xx, yy):
            renderer.SetDisplayPoint(float(xx), float(H - yy), 0.0)
            renderer.DisplayToWorld()
            x0, y0, z0, w0 = renderer.GetWorldPoint()
            if abs(w0) > 1e-12:
                x0, y0, z0 = x0 / w0, y0 / w0, z0 / w0

            renderer.SetDisplayPoint(float(xx), float(H - yy), 1.0)
            renderer.DisplayToWorld()
            x1, y1, z1, w1 = renderer.GetWorldPoint()
            if abs(w1) > 1e-12:
                x1, y1, z1 = x1 / w1, y1 / w1, z1 / w1

            o = np.array([x0, y0, z0], dtype=float)
            d = np.array([x1, y1, z1], dtype=float) - o
            n = float(np.linalg.norm(d))
            if n < 1e-12:
                o = np.array(cam.GetPosition(), dtype=float)
                d = np.array(cam.GetFocalPoint(), dtype=float) - o
                n = float(np.linalg.norm(d))
            return o, (d / max(n, 1e-12))

        pos0 = np.array(cam.GetPosition(), dtype=float)
        fp0 = np.array(cam.GetFocalPoint(), dtype=float)
        vu0 = np.array(cam.GetViewUp(), dtype=float)

        n0 = fp0 - pos0
        n0n = float(np.linalg.norm(n0))
        if n0n < 1e-12:
            return
        n0 /= n0n

        o0, d0 = ray_through_xy(ren, x, y)
        denom0 = float(np.dot(d0, n0))
        anchor = fp0.copy() if abs(denom0) < 1e-12 else (o0 + d0 * float(np.dot(fp0 - o0, n0) / denom0))

        factor = 1.0 if delta_y == 0 else (1.2 ** (delta_y / 120.0))
        if factor <= 0.0:
            return

        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / max(1e-6, factor))
            shift = (anchor - fp0) * (1.0 - 1.0 / factor)
            cam.SetFocalPoint(*(fp0 + shift))
            cam.SetPosition(*(pos0 + shift))
            cam.SetViewUp(*vu0)
        else:
            cam.SetPosition(*(anchor + (pos0 - anchor) / factor))
            cam.SetFocalPoint(*(anchor + (fp0 - anchor) / factor))
            cam.SetViewUp(*vu0)

        pos1 = np.array(cam.GetPosition(), dtype=float)
        fp1 = np.array(cam.GetFocalPoint(), dtype=float)
        n1 = fp1 - pos1
        n1n = float(np.linalg.norm(n1))
        if n1n >= 1e-12:
            n1 /= n1n
            o1, d1 = ray_through_xy(ren, x, y)
            denom1 = float(np.dot(d1, n1))
            if abs(denom1) > 1e-12:
                t1 = float(np.dot(anchor - o1, n1) / denom1)
                q = o1 + d1 * t1
                pan = anchor - q
                if np.isfinite(pan).all():
                    cam.SetPosition(*(pos1 + pan))
                    cam.SetFocalPoint(*(fp1 + pan))
                    cam.SetViewUp(*vu0)

    finally:
        if atomic:
            try:
                app.plotter.interactor.setUpdatesEnabled(True)
            except Exception:
                pass
            try:
                if hasattr(app, "plotter_ref"):
                    app.plotter_ref.interactor.setUpdatesEnabled(True)
            except Exception:
                pass
            app._cam_pause = False
            app._in_zoom = False

    try:
        plotter.reset_camera_clipping_range()
        if app.repair_mode and getattr(app, "plotter_ref", None) is not None and app.plotter_ref.isVisible():
            sync_renders(app)
        else:
            plotter.render()
    except Exception:
        pass
