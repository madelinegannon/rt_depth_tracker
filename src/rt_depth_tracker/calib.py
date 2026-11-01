# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Calibration math: rigid transforms, Kabsch, fixed-radius sphere fit, RANSAC plane, the floor / yaw / table
steps, sensor calibration files (sensor -> WORLD 4x4)."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

import numpy as np


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P)
    return P @ T[:3, :3].T.astype(P.dtype) + T[:3, 3].astype(P.dtype)


def euler_xyz_deg_to_R(rx, ry, rz) -> np.ndarray:
    a, b, c = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def look_at_T(cam_pos, target, up=(0, 0, 1)) -> np.ndarray:
    """Sensor->world transform for a camera at cam_pos looking at target (sensor: x right, y down, z fwd)."""
    cam_pos, target = np.asarray(cam_pos, float), np.asarray(target, float)
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))     # straight down (or up): image 'up' is world +x
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)     # columns = sensor axes in world
    return make_T(R, cam_pos)


def kabsch(A: np.ndarray, B: np.ndarray):
    """Rigid T with B ~= R A + t (least squares). Returns (T, rms)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    T = make_T(R, t)
    rms = float(np.sqrt(((apply(T, A) - B) ** 2).sum(1).mean()))
    return T, rms


def fit_sphere_fixed_radius(P: np.ndarray, radius: float, c0=None, iters: int = 12, huber: float = 0.02):
    """Gauss-Newton on centre only (radius known), Huber-weighted. Returns (centre, rms_residual, n_inliers)."""
    P = np.asarray(P, float)
    c = np.asarray(c0, float) if c0 is not None else P.mean(0) + np.array([0, 0, 0.0])
    if c0 is None:
        # push the initial centre away from the visible front hemisphere along the mean normal
        n = P.mean(0) - np.median(P, 0)
        c = P.mean(0)
    for _ in range(iters):
        d = P - c
        r = np.linalg.norm(d, axis=1)
        res = r - radius
        w = np.where(np.abs(res) < huber, 1.0, huber / np.maximum(np.abs(res), 1e-9))
        J = -d / np.maximum(r[:, None], 1e-9)          # d res / d c
        JtWJ = (J * w[:, None]).T @ J
        JtWr = (J * w[:, None]).T @ res
        try:
            step = np.linalg.solve(JtWJ + 1e-9 * np.eye(3), -JtWr)
        except np.linalg.LinAlgError:
            break
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    res = np.abs(np.linalg.norm(P - c, axis=1) - radius)
    inl = res < huber
    rms = float(np.sqrt((res[inl] ** 2).mean())) if inl.any() else float("inf")
    return c, rms, int(inl.sum())


def fit_plane_ransac(P: np.ndarray, thresh: float = 0.02, iters: int = 200, rng=None):
    """Returns (normal (unit, pointing +z-ish), d) with n.x + d = 0, inlier mask."""
    P = np.asarray(P, float)
    rng = rng or np.random.default_rng(0)
    best, best_inl = None, None
    n_pts = len(P)
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        inl = np.abs(P @ n + d) < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best, best_inl = (n, d), inl
    n, d = best
    # refine with SVD on inliers
    Q = P[best_inl]
    cq = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - cq)
    n = Vt[2]
    if n[2] < 0:
        n = -n
    d = -n @ cq
    inl = np.abs(P @ n + d) < thresh
    return n, float(d), inl


def R_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


# ---- files -----------------------------------------------------------------------------
def save_sensor_calib(path: str, cam_id: int, T_world_sensor: np.ndarray, meta: dict | None = None) -> str:
    d = {"cam_id": int(cam_id), "T_world_sensor": np.asarray(T_world_sensor, float).round(9).tolist(),
         "frame": "world: Z-up, metres, origin on the floor", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "meta": meta or {}}
    body = json.dumps(d, indent=1)
    d["sha"] = hashlib.sha256(body.encode()).hexdigest()[:12]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    return d["sha"]


def sensor_calib_files(calib_dir: str) -> list[str]:
    """The live sensor_*.json files, sorted; never the .prev.json backups the calib tool keeps for `undo`. They
    match a wider pattern and sort after the live file: a loader that globbed them ran on the PREVIOUS calibration."""
    return [f for f in sorted(glob.glob(os.path.join(calib_dir, "sensor_?.json"))) if not f.endswith(".prev.json")]


def load_sensor_calib(path: str):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["T_world_sensor"], dtype=np.float64), d


# ---- field calibration steps (floor plane, one known mark) ------------------------------------
def rotate_about(T: np.ndarray, R: np.ndarray, pivot: np.ndarray, dz: float = 0.0) -> np.ndarray:
    """World rotation R about `pivot`, then a z shift, applied to a sensor->world T."""
    Tc = np.eye(4)
    Tc[:3, :3] = R
    Tc[:3, 3] = pivot - R @ pivot + np.array([0.0, 0.0, dz])
    return Tc @ T


def floor_correction(P_world: np.ndarray, T: np.ndarray, z_band=(-1.5, 0.6), r_max=4.5, thresh=0.02, iters=400, rng=None):
    """Roll/pitch/height fix: RANSAC plane through the points the current T puts near the floor, rotate
    about the camera position so the plane normal becomes +z, lift so the floor sits at z = 0.
    Returns (T_new, report); T_new is None when there are too few floor points."""
    P = np.asarray(P_world, float)
    r = np.hypot(P[:, 0], P[:, 1])
    cand = P[(P[:, 2] > z_band[0]) & (P[:, 2] < z_band[1]) & (r < r_max)]
    rep = {"floor_candidates": int(len(cand))}
    if len(cand) < 200:
        return None, rep
    n, d, inl = fit_plane_ransac(cand, thresh, iters, rng)
    cpos = T[:3, 3].copy()
    R = R_align(n, np.array([0.0, 0.0, 1.0]))
    p0 = cpos - (n @ cpos + d) * n                    # foot of the perpendicular from the camera to the floor
    zf = float((R @ (p0 - cpos) + cpos)[2])           # where that point lands after the rotation
    rep.update(tilt_before_deg=round(float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), 2),
               height_before_m=round(float(-d / n[2]), 3), inliers=int(inl.sum()))
    return rotate_about(T, R, cpos, -zf), rep


def floor_residual(P_world: np.ndarray, thresh=0.02, iters=200, rng=None):
    """(tilt_deg, height_m) of the floor plane in already-corrected world points; for reports and tests."""
    P = np.asarray(P_world, float)
    r = np.hypot(P[:, 0], P[:, 1])
    cand = P[(np.abs(P[:, 2]) < 0.25) & (r < 4.5)]
    if len(cand) < 50:
        return None, None
    n, d, _ = fit_plane_ransac(cand, thresh, iters, rng)
    return float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))), float(-d / n[2])


def yaw_correction(seen_xy, T: np.ndarray, ref_xy):
    """Yaw fix from one known mark: rotate about the camera's vertical axis so the bearing camera->seen
    equals camera->ref. Returns (T_new, report); report['range_ratio'] (seen/ref) must be ~1, otherwise
    the mark, the camera position in the config, or the person's spot is wrong."""
    cpos = T[:3, 3].copy()
    seen, ref = np.asarray(seen_xy, float)[:2], np.asarray(ref_xy, float)[:2]
    b_seen = np.arctan2(seen[1] - cpos[1], seen[0] - cpos[0])
    b_ref = np.arctan2(ref[1] - cpos[1], ref[0] - cpos[0])
    dyaw = float((b_ref - b_seen + np.pi) % (2 * np.pi) - np.pi)
    c, s = np.cos(dyaw), np.sin(dyaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_seen, r_ref = float(np.hypot(*(seen - cpos[:2]))), float(np.hypot(*(ref - cpos[:2])))
    rep = {"yaw_deg": round(float(np.degrees(dyaw)), 2), "range_seen_m": round(r_seen, 3),
           "range_ref_m": round(r_ref, 3), "range_ratio": round(r_seen / r_ref, 3)}
    return rotate_about(T, R, cpos), rep


def stable_runs(centroids, times, min_s=5.0, jump=0.15):
    """Group per-frame centroids (x, y[, ...]) or None into runs whose centroid stays within `jump` of the
    run mean; returns [(i0, i1, mean_xy)] for runs lasting at least min_s."""
    runs, cur = [], []
    for k, c in enumerate(centroids):
        if c is None:
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur:
            m = np.mean([centroids[i][:2] for i in cur], axis=0)
            if np.hypot(c[0] - m[0], c[1] - m[1]) > jump:
                runs.append(cur)
                cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    return [(ru[0], ru[-1], np.mean([centroids[i][:2] for i in ru], axis=0))
            for ru in runs if times[ru[-1]] - times[ru[0]] >= min_s]


# ---- aiming (frustum coverage) and image-side checks ---------------------------------------------
def look_at_pan_tilt(cam_pos, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Sensor->world T for a camera at cam_pos whose optical axis points pan_deg from +x toward +y and
    tilt_deg below horizontal."""
    pan, tilt = np.radians(pan_deg), np.radians(tilt_deg)
    d = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), -np.sin(tilt)])
    return look_at_T(cam_pos, np.asarray(cam_pos, float) + d)


def pan_tilt_of(T: np.ndarray):
    """(pan_deg, tilt_deg, floor_point) of a sensor->world T: the optical axis direction and where it meets z = 0."""
    d, c = T[:3, 2], T[:3, 3]
    pan = float(np.degrees(np.arctan2(d[1], d[0])))
    tilt = float(np.degrees(np.arcsin(np.clip(-d[2], -1, 1))))
    floor = (c - d * (c[2] / d[2])) if d[2] < -1e-6 else None
    return pan, tilt, floor


def in_view(P_world, T: np.ndarray, intr, stride: int = 1, margin_px: float = 0.0) -> np.ndarray:
    """Mask of world points that land inside the image (real distortion model) in front of the camera."""
    from rt_depth_tracker.intrinsics import project
    Ps = apply(invert(T), np.asarray(P_world, float))
    u, v = project(Ps, intr, stride)
    w, h = intr.width / stride, intr.height / stride
    return (Ps[:, 2] > 0.05) & (u >= margin_px) & (u < w - margin_px) & (v >= margin_px) & (v < h - margin_px)


def zone_samples(x_min=1.2, x_max=4.0, y_half=2.0, heights=(0.2, 1.0), step=0.1) -> np.ndarray:
    """Sample points of the coverage zone (a box of floor in front of the origin, +x) at the given heights."""
    xs = np.arange(x_min, x_max + 1e-9, step)
    ys = np.arange(-y_half, y_half + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.concatenate([np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))], 1) for z in heights])


def aim_search(cam_positions: dict, intr, stride: int = 2, zone: np.ndarray | None = None, pans=None, tilts=None):
    """Per camera, the (pan, tilt) that maximises the zone seen by BOTH cameras: coarse 5 deg grid on the
    joint objective, then 1 deg refinement of each camera with the other fixed. Samples near the axis weigh
    more so ties resolve toward the space directly in front of the robot.
    Coverage is scored with nested image margins (0, 12, 24 px) so that, on a plateau where several aims cover
    everything, the aim that keeps the zone well inside both images wins.
    Returns ({cam: {pan_deg, tilt_deg, T, floor_point, own_frac}}, both_frac)."""
    Z = zone_samples() if zone is None else np.asarray(zone, float)
    w = 1.0 / (1.0 + np.abs(Z[:, 1]))
    margins = (0.0, 12.0, 24.0)
    def cover(T):
        return sum(in_view(Z, T, intr, stride, m).astype(np.float32) * (0.5 ** k) for k, m in enumerate(margins))
    pans = np.arange(-60, 61, 5.0) if pans is None else np.asarray(pans, float)
    tilts = np.arange(15, 71, 5.0) if tilts is None else np.asarray(tilts, float)
    cams = sorted(cam_positions)
    combos = [(float(p), float(t)) for p in pans for t in tilts]
    grids = {c: np.stack([cover(look_at_pan_tilt(cam_positions[c], p, t)) for p, t in combos]) for c in cams}
    best = {}
    if len(cams) == 1:
        c = cams[0]
        best[c] = combos[int((grids[c] * w).sum(1).argmax())]
    else:
        a, b = cams[0], cams[1]
        inter = (grids[a] * w) @ grids[b].T
        i, j = np.unravel_index(int(inter.argmax()), inter.shape)
        best = {a: combos[i], b: combos[j]}
        for _ in range(2):
            for c, o in ((a, b), (b, a)):
                mo = cover(look_at_pan_tilt(cam_positions[o], *best[o]))
                p0, t0 = best[c]
                cand = [(p, t) for p in np.arange(p0 - 5, p0 + 5.1, 1.0) for t in np.arange(t0 - 5, t0 + 5.1, 1.0)]
                scores = [float(((cover(look_at_pan_tilt(cam_positions[c], p, t)) * mo) * w).sum()) for p, t in cand]
                best[c] = cand[int(np.argmax(scores))]
    out, masks = {}, {}
    for c in cams:
        T = look_at_pan_tilt(cam_positions[c], *best[c])
        masks[c] = in_view(Z, T, intr, stride)
        pan, tilt, fl = pan_tilt_of(T)
        out[c] = {"pan_deg": round(pan, 1), "tilt_deg": round(tilt, 1), "T": T,
                  "floor_point": None if fl is None else np.round(fl, 3), "own_frac": round(float(masks[c].mean()), 3)}
    both = float(np.logical_and.reduce([masks[c] for c in cams]).mean())
    return out, both


def predicted_column(T: np.ndarray, intr, stride: int, point_world) -> float:
    """Image column (strided pixels) where the current calibration expects a world point."""
    from rt_depth_tracker.intrinsics import project
    Ps = apply(invert(T), np.asarray(point_world, float)[None])
    u, _ = project(Ps, intr, stride)
    return float(u[0])



# ---- table fit (no robot): yaw + xy from a rectangle of known size in the background cloud ------------
def yaw_T(yaw_deg: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """World-frame nudge: rotate about the world z axis through the origin, then shift in xy."""
    return make_T(euler_xyz_deg_to_R(0.0, 0.0, yaw_deg), np.array([dx, dy, 0.0]))


def _extents(Q, yaw_deg, pct):
    R = euler_xyz_deg_to_R(0.0, 0.0, -yaw_deg)[:2, :2]
    L = Q @ R.T
    lo, hi = np.percentile(L, pct, axis=0), np.percentile(L, 100 - pct, axis=0)
    return R, lo, hi


def rect_fit_xy(Q_xy: np.ndarray, pct: float = 0.5):
    """Minimum-area rectangle of a point set (coarse 1 deg sweep, then 0.05 deg refinement): (yaw_deg of the long
    axis in (-90, 90], centre (2,), (long, short) extents from the pct..100-pct percentiles along the axes)."""
    Q = np.asarray(Q_xy, float)
    best = None
    for step, coarse in ((1.0, True), (0.05, False)):
        angles = np.arange(-90.0, 90.0, step) if coarse else np.arange(best - 1.0, best + 1.0, step)
        areas = []
        for a in angles:
            _, lo, hi = _extents(Q, a, pct)
            areas.append((hi[0] - lo[0]) * (hi[1] - lo[1]))
        best = float(angles[int(np.argmin(areas))])
    R, lo, hi = _extents(Q, best, pct)
    ext = hi - lo
    yaw = best if ext[0] >= ext[1] else best + 90.0      # report the long axis
    yaw = (yaw + 90.0) % 180.0 - 90.0
    centre = R.T @ ((lo + hi) / 2)
    long_m, short_m = (float(ext[0]), float(ext[1])) if ext[0] >= ext[1] else (float(ext[1]), float(ext[0]))
    return yaw, centre, (long_m, short_m)


def table_correction(P_world: np.ndarray, T: np.ndarray, top_z: float, size_xy, band: float = 0.06, r_max: float | None = None,
                     thresh: float = 0.02, iters: int = 300, max_size_err: float = 0.15, min_pts: int = 300, rng=None):
    """Yaw + xy fix from a rectangular top of known size: RANSAC plane through the points near z = top_z, a rectangle
    fit of its inliers, then the world is rotated/shifted so that rectangle sits centred on the origin with its long
    side along +x (size_xy[0] >= size_xy[1]) or +y. Refuses when the measured sides disagree with size_xy by
    more than max_size_err. Returns (T_new or None, report)."""
    P = np.asarray(P_world, float)
    sx, sy = float(size_xy[0]), float(size_xy[1])
    r_max = r_max or 1.5 * max(sx, sy)
    r = np.hypot(P[:, 0], P[:, 1])
    cand = P[(np.abs(P[:, 2] - top_z) < band) & (r < r_max)]
    rep = {"top_candidates": int(len(cand))}
    if len(cand) < min_pts:
        return None, {**rep, "applied": False, "why": "too few points near the table top: check top_z, the seed height and plane"}
    n, d, inl = fit_plane_ransac(cand, thresh, iters, rng)
    Q = cand[inl]
    rep.update(inliers=int(inl.sum()), top_z_found=round(float(Q[:, 2].mean()), 3),
               top_tilt_deg=round(float(np.degrees(np.arccos(np.clip(abs(n[2]), -1, 1)))), 2))
    yaw, centre, (long_m, short_m) = rect_fit_xy(Q[:, :2])
    want_long, want_short = max(sx, sy), min(sx, sy)
    ratio = (long_m / want_long, short_m / want_short)
    rep.update(size_found=[round(long_m, 3), round(short_m, 3)], size_ratio=[round(ratio[0], 3), round(ratio[1], 3)],
               yaw_found_deg=round(yaw, 2), centre_found=[round(float(centre[0]), 3), round(float(centre[1]), 3)])
    if abs(ratio[0] - 1) > max_size_err or abs(ratio[1] - 1) > max_size_err:
        return None, {**rep, "applied": False, "why": "measured table size does not match table.size_xy (wrong table, occluded top, or wrong top_z)"}
    desired = 0.0 if sx >= sy else 90.0
    corr = (desired - yaw + 90.0) % 180.0 - 90.0           # rotation to apply, in (-90, 90]
    Rz = euler_xyz_deg_to_R(0.0, 0.0, corr)
    W = make_T(Rz, -(Rz @ np.array([centre[0], centre[1], 0.0])))
    rep.update(yaw_applied_deg=round(corr, 2), shift_applied=[round(float(-centre[0]), 3), round(float(-centre[1]), 3)], applied=True)
    return W @ T, rep
