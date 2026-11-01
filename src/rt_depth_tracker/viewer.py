# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Interactive 3D QA + calibration viewer: an orbit viewer (GLFW + OpenGL 3.3 + Dear ImGui) around the tracker
pipeline. Same config, calibration and background files as the tracker, so what it draws is what the tracker
computes. The Calibration panel nudges the world (yaw / x / y) under the loaded calibration live and saves it; the
Zones panel edits the configured zones live and saves them to the config file.

  python -m rt_depth_tracker.viewer --clip recordings/<tag>_*.npz [--calib-dir config/calib] [--learn-bg N] [--max-steps N]
  python -m rt_depth_tracker.viewer --live [--in 7013] [--q-port 7014]            beside the running tracker (mirror ports)
  python -m rt_depth_tracker.viewer --clip ... --screenshot out.png [--at 10.0]   offscreen render of one clip time, no window

Mouse: left-drag orbit, middle/right-drag pan, wheel zoom. Keys: space play/pause, left/right step one frame,
r reset view, s save a PNG, q quit. Every layer (background clouds, points per camera, cropped, robot-masked,
robot spheres, cluster boxes, track boxes, gaze/reach, target poses, trails, frustums, zones) is a checkbox.
The GL-free half (view maths, boxes, step cache, zone edits) is tested in tests/test_viewer.py; the GL half only at runtime."""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker import protocol as P
from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.intrinsics import ray_table, unproject
from rt_depth_tracker.pipeline import Pipeline
from rt_depth_tracker.segment import attach_fragments
from rt_depth_tracker.zones import zones_from_cfg

CLEAR = (0.08, 0.08, 0.10)
COL_CAM = {0: (0.25, 0.55, 1.0), 1: (1.0, 0.6, 0.2)}
COL_BG = {0: (0.28, 0.33, 0.42), 1: (0.42, 0.35, 0.28)}
COL_CROP = (0.45, 0.45, 0.45)
COL_ROBOT_PTS = (0.6, 0.15, 0.15)
COL_ROBOT = (0.1, 0.7, 0.9)
COL_REJECT = (0.45, 0.45, 0.45)
COL_TALL = (0.9, 0.55, 0.2)             # a component above max_height: a set piece, not a visitor
COL_CAND = (0.9, 0.9, 0.9)
COL_TENT = (0.6, 0.6, 0.6)
COL_CONF = (0.9, 0.8, 0.2)
COL_CUR = (0.2, 0.9, 0.3)
COL_REACH = (0.95, 0.25, 0.2)
COL_TARGET = (1.0, 0.2, 1.0)
COL_LOCAL = (0.55, 0.3, 0.55)           # this viewer's own decision when the tracker mirror is live
COL_STATIC = (0.35, 0.35, 0.4)
COL_FRUSTUM = (0.5, 0.5, 0.5)
COL_TABLE = (0.3, 0.75, 0.9)
COL_ZONE = (0.9, 0.55, 0.2)
COL_ZONE_ON = (0.95, 0.25, 0.2)
NEAR_COL, FAR_COL = np.array([0.2, 0.8, 0.9]), np.array([0.9, 0.2, 0.3])
PALETTE = np.array([[0.12, 0.47, 0.71], [1.0, 0.5, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16], [0.58, 0.4, 0.74],
                    [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.74, 0.74, 0.13], [0.09, 0.75, 0.81]])
HINT_COL = {"idle": (0.4, 0.4, 0.4), "watch": (0.9, 0.8, 0.2), "engage": (0.2, 0.9, 0.3), "reach": (0.95, 0.25, 0.2),
            "sleep": (0.25, 0.3, 0.5)}
QUIET = ("idle", "sleep")      # no target to draw: nobody engaged
LAYERS = ["bg0", "bg1", "pts0", "pts1", "crop", "robot_pts", "robot", "clusters", "tracks", "gaze", "pose", "target", "local", "sent_tracks",
          "trails", "frustums", "static", "zones"]
LAYER_LABELS = {"bg0": "background cam 0", "bg1": "background cam 1", "pts0": "points cam 0", "pts1": "points cam 1",
                "crop": "cropped (world crop)", "robot_pts": "robot-masked points", "robot": "robot spheres @ q",
                "clusters": "cluster boxes", "tracks": "track boxes", "gaze": "gaze + reach points",
                "pose": "target pose per track", "target": "target (as sent when the mirror is live)",
                "local": "this viewer's own decision", "sent_tracks": "tracks as sent (mirror)", "trails": "trails 3 s",
                "frustums": "camera frustums", "static": "floor grid / wall / rail / marks", "zones": "zones + table"}


# ---- GL-free core -----------------------------------------------------------------------------------
class ViewControl:
    """Orbit camera: yaw/pitch about a target, Z-up, degrees."""

    def __init__(self):
        self.fov_deg, self.near, self.far = 60.0, 0.01, 100.0
        self.reset()

    def reset(self):
        self.yaw, self.pitch, self.distance = 90.0, 30.0, 4.0
        self.target = np.zeros(3)

    def rotate(self, dx: float, dy: float):
        self.yaw += dx * 0.3
        self.pitch = float(np.clip(self.pitch + dy * 0.3, -89.0, 89.0))

    def pan(self, dx: float, dy: float):
        sy, cy = np.sin(np.radians(self.yaw)), np.cos(np.radians(self.yaw))
        s = self.distance * 0.002
        self.target = self.target + np.array([-sy * dx * s, cy * dx * s, dy * s])

    def zoom(self, delta: float):
        self.distance = float(np.clip(self.distance - delta * 0.3, 0.1, 50.0))

    def eye(self) -> np.ndarray:
        cy, sy = np.cos(np.radians(self.yaw)), np.sin(np.radians(self.yaw))
        cp, sp = np.cos(np.radians(self.pitch)), np.sin(np.radians(self.pitch))
        return self.target + self.distance * np.array([cp * cy, cp * sy, sp])

    def view_matrix(self) -> np.ndarray:
        """Row-major 4x4: v_cam = V @ v_world (upload with transpose=True)."""
        e = self.eye()
        f = self.target - e
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        V = np.eye(4)
        V[0, :3], V[1, :3], V[2, :3] = r, u, -f
        V[:3, 3] = [-r @ e, -u @ e, f @ e]
        return V

    def proj_matrix(self, aspect: float) -> np.ndarray:
        fl = 1.0 / np.tan(np.radians(self.fov_deg) * 0.5)
        nf = self.near - self.far
        Pm = np.zeros((4, 4))
        Pm[0, 0], Pm[1, 1] = fl / max(aspect, 1e-6), fl
        Pm[2, 2], Pm[2, 3] = (self.far + self.near) / nf, 2.0 * self.far * self.near / nf
        Pm[3, 2] = -1.0
        return Pm


def project(P: np.ndarray, V: np.ndarray, Pm: np.ndarray, w: float, h: float):
    """World (N,3) -> screen pixels (N,2), x right / y down, plus an in-front-and-on-screen mask."""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    if len(P) == 0:
        return np.zeros((0, 2)), np.zeros(0, bool)
    Ph = np.c_[P, np.ones(len(P))] @ (Pm @ V).T
    wc = Ph[:, 3]
    ok = wc > 1e-6
    ndc = Ph[:, :2] / np.where(ok, wc, 1.0)[:, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * w
    sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
    ok &= (np.abs(ndc[:, 0]) < 1.1) & (np.abs(ndc[:, 1]) < 1.1)
    return np.stack([sx, sy], 1), ok


def boxes_from_labels(P: np.ndarray, labels: np.ndarray) -> dict:
    """{label: (lo (3,), hi (3,))} axis-aligned bounds of the points carrying each label >= 0."""
    out = {}
    if len(P) == 0:
        return out
    for lab in np.unique(labels):
        if lab < 0:
            continue
        Q = P[labels == lab]
        out[int(lab)] = (Q.min(0), Q.max(0))
    return out


def circle(c, r: float, axis: int = 2, segs: int = 32) -> np.ndarray:
    """(2*segs, 3) GL_LINES vertices of a circle of radius r around c, normal along `axis`."""
    a = np.linspace(0, 2 * np.pi, segs + 1)
    ca, sa = r * np.cos(a), r * np.sin(a)
    pts = np.zeros((segs + 1, 3))
    if axis == 2:
        pts[:, 0], pts[:, 1] = ca, sa
    elif axis == 0:
        pts[:, 1], pts[:, 2] = ca, sa
    else:
        pts[:, 0], pts[:, 2] = ca, sa
    pts += np.asarray(c, float)
    return np.repeat(pts, 2, axis=0)[1:-1]


def sphere_wire(c, r: float, segs: int = 24) -> np.ndarray:
    return np.vstack([circle(c, r, 2, segs), circle(c, r, 0, segs), circle(c, r, 1, segs)])


def box_edges(lo, hi) -> np.ndarray:
    """(24, 3) GL_LINES vertices for the 12 edges of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return c[np.array(e).ravel()]


def segments(pairs) -> np.ndarray:
    """[(a, b), ...] -> (2n, 3) GL_LINES vertices."""
    if not pairs:
        return np.zeros((0, 3), np.float32)
    return np.asarray([p for ab in pairs for p in ab], np.float32)


def frustum_lines(intr, T_ws: np.ndarray, stride: int, near: float = 0.3, far: float = 4.0) -> np.ndarray:
    """Camera frustum from the real ray table corners (sensor frame, z forward) placed by T_ws."""
    rays = ray_table(intr, stride)
    corners = rays[[0, 0, -1, -1], [0, -1, -1, 0]].astype(np.float64)
    n, f = corners * near, corners * far
    o = np.zeros(3)
    pairs = [(n[i], n[(i + 1) % 4]) for i in range(4)] + [(f[i], f[(i + 1) % 4]) for i in range(4)]
    pairs += [(o, f[i]) for i in range(4)] + [(n[i], f[i]) for i in range(4)]
    return C.apply(T_ws, segments(pairs)).astype(np.float32)


def static_lines(cfg: TrackerConfig) -> np.ndarray:
    """Floor grid, plus whatever the config declares: a back wall, floor marks, a rail, a pedestal."""
    pairs = []
    for x in np.arange(-3.0, 5.01, 1.0):
        pairs.append(([x, -3, 0], [x, 3, 0]))
    for y in np.arange(-3.0, 3.01, 1.0):
        pairs.append(([-3, y, 0], [5, y, 0]))
    if cfg.wall_x is not None:
        wx = cfg.wall_x
        pairs += [([wx, -3, 0], [wx, 3, 0]), ([wx, -3, 3], [wx, 3, 3]), ([wx, -3, 0], [wx, -3, 3]), ([wx, 3, 0], [wx, 3, 3])]
    for (mx, my) in cfg.marks.values():
        pairs += [([mx - 0.1, my, 0.01], [mx + 0.1, my, 0.01]), ([mx, my - 0.1, 0.01], [mx, my + 0.1, 0.01])]
    L = [segments(pairs)]
    bx, by = cfg.base_xy
    if cfg.rail_radius > 0:
        L.append(circle((bx, by, cfg.rail_top_z), cfg.rail_radius, 2, 64))
    if cfg.pedestal_radius > 0:
        L += [circle((bx, by, 0.0), cfg.pedestal_radius, 2, 32), circle((bx, by, cfg.mount_z), cfg.pedestal_radius, 2, 32)]
        for a in np.linspace(0, 2 * np.pi, 4, endpoint=False):
            x, y = bx + cfg.pedestal_radius * np.cos(a), by + cfg.pedestal_radius * np.sin(a)
            L.append(segments([([x, y, 0], [x, y, cfg.mount_z])]))
    return np.vstack(L).astype(np.float32)


def zone_lines(cfg: TrackerConfig):
    """(xyz, colours) GL_LINES of the table rectangle (if configured) and a box per zone from its top to the top of
    the hand band; empty arrays without either."""
    xyz, col = [], []
    tb = cfg.table or {}
    if tb.get("size_xy") and tb.get("top_z") is not None:
        cx, cy = tb.get("center_xy", (0.0, 0.0))
        sx, sy = tb["size_xy"]
        e = box_edges([cx - sx / 2, cy - sy / 2, 0.0], [cx + sx / 2, cy + sy / 2, float(tb["top_z"])])
        xyz.append(e)
        col.append(np.broadcast_to(COL_TABLE, e.shape))
    for z in zones_from_cfg(cfg):
        e = box_edges([z.lo[0], z.lo[1], z.top_z], [z.hi[0], z.hi[1], z.top_z + cfg.zone_margin_z])
        xyz.append(e)
        col.append(np.broadcast_to(COL_ZONE, e.shape))
    if not xyz:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.vstack(xyz).astype(np.float32), np.vstack(col).astype(np.float32)


def zone_edits(cfg: TrackerConfig) -> list[dict]:
    """Slider state per zone from the config: centre, size, top and yaw."""
    return [dict(name=z.name, cx=float(z.centre[0]), cy=float(z.centre[1]), w=float(z.hi[0] - z.lo[0]), h=float(z.hi[1] - z.lo[1]),
                 top=float(z.top_z), yaw=float(z.yaw_deg)) for z in zones_from_cfg(cfg)]


def zones_from_edits(edits: list[dict]) -> list[dict]:
    return [{"name": e["name"], "rect": [round(e["cx"] - e["w"] / 2.0, 4), round(e["cy"] - e["h"] / 2.0, 4),
                                        round(e["cx"] + e["w"] / 2.0, 4), round(e["cy"] + e["h"] / 2.0, 4)],
             "top_z": round(e["top"], 4), "yaw_deg": round(e["yaw"], 1)} for e in edits]


def apply_zone_edits(cfg: TrackerConfig, edits: list[dict]) -> None:
    """Push the slider state into the live config (the local pipeline and the overlay read it)."""
    cfg.zones = zones_from_edits(edits)


def save_zone_edits(config_path: str, cfg: TrackerConfig, edits: list[dict]) -> str:
    """Write the zones (and the zone band knobs) to the config file the tracker reloads, keeping its other keys."""
    paths = [p for p in TrackerConfig.paths(config_path) if os.path.exists(p)]
    target = paths[-1] if paths else config_path       # the overlay wins on reload, so it is the file to edit
    raw = {}
    if os.path.exists(target):
        with open(target, "r", encoding="utf-8") as f:
            raw = json.load(f)
    raw["zones"] = zones_from_edits(edits)
    for k in ZONE_KNOBS:
        raw[k] = getattr(cfg, k)
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=1)
        f.write("\n")
    os.replace(tmp, target)
    return f"saved {len(edits)} zones -> {os.path.basename(target)}"


ZONE_KNOBS = ("zone_margin_z", "reach_z_max", "zone_near_m", "lean_bias_m", "lean_margin_m", "lean_z_lo", "lean_z_hi")


def axes_lines(origin, length: float):
    o = np.asarray(origin, float)
    xyz = np.array([o, o + [length, 0, 0], o, o + [0, length, 0], o, o + [0, 0, length]], np.float32)
    col = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], np.float32)
    return xyz, col


def bg_cloud(bg, intr, T_ws: np.ndarray, stride: int, n_max: int = 20000, rng=None) -> np.ndarray:
    """The learned background (median depth) unprojected through the tracker's ray table, subsampled."""
    if bg is None or bg.bg is None or intr is None or T_ws is None:
        return np.zeros((0, 3), np.float32)
    rays = ray_table(intr, stride)
    if rays.shape[:2] != bg.bg.shape:
        return np.zeros((0, 3), np.float32)
    P_s, _ = unproject(bg.bg, rays)
    Pw = C.apply(T_ws, P_s).astype(np.float32)
    if len(Pw) > n_max:
        rng = rng or np.random.default_rng(0)
        Pw = Pw[rng.choice(len(Pw), n_max, replace=False)]
    return Pw


def cross_lines(t, k: float, extra=None) -> np.ndarray:
    """Axis-aligned cross at t (segments), optionally joined to an extra point (the shoulder)."""
    t = np.asarray(t, np.float64)
    segs = [t - [k, 0, 0], t + [k, 0, 0], t - [0, k, 0], t + [0, k, 0], t - [0, 0, k], t + [0, 0, k]]
    if extra is not None:
        segs += [np.asarray(extra, np.float64), t]
    return np.array(segs)


def diamond_lines(p, k: float) -> np.ndarray:
    """Horizontal diamond around p (4 segments): the tracks the tracker actually sent."""
    p = np.asarray(p, np.float64)
    a, b, c, d = p - [k, 0, 0], p - [0, k, 0], p + [k, 0, 0], p + [0, k, 0]
    return np.array([a, b, b, c, c, d, d, a])


@dataclass
class Step:
    t: float
    points: np.ndarray
    cam_ids: np.ndarray
    labels: np.ndarray                 # after attach_fragments: the labels cluster_features scored
    dropped: dict
    clusters: list
    tracks: list                       # every track, tentative included
    scores: dict
    current: int | None
    decision: object
    q: np.ndarray
    q_age: float | None
    step_ms: float
    zones: list = field(default_factory=list)      # ZoneState per configured zone
    boxes: dict = field(default_factory=dict)
    sent: object = None                # P.Target the tracker sent out (mirror), if fresh
    sent_age: float | None = None
    sent_tracks: object = None         # (n, 8) rows as sent, if fresh
    sent_zones: object = None          # {idx: P.ZoneMsg} as sent, if fresh


def snapshot(pipe: Pipeline, dbg, dec, t: float, q, q_age, sent=None, sent_age=None, sent_tracks=None, sent_zones=None) -> Step:
    labels = attach_fragments(dbg.points, dbg.labels, pipe.cfg) if len(dbg.points) else dbg.labels
    st = Step(float(t), dbg.points, dbg.cam_ids, labels, dict(dbg.dropped), list(dbg.clusters),
              copy.deepcopy(pipe.tracker.tracks), dict(pipe.interest.S), pipe.interest.current, dec,
              np.asarray(q, np.float64).copy(), q_age, float(dbg.step_ms), list(getattr(dbg, "zones", [])))
    st.boxes = boxes_from_labels(st.points, st.labels)
    st.sent, st.sent_age, st.sent_tracks, st.sent_zones = sent, sent_age, sent_tracks, sent_zones
    return st


def load_calib_into(pipe: Pipeline, calib_dir: str) -> dict:
    """sensor_*.json -> set_calib, bg_*.npz -> the pipeline's background, intrinsics_*.json -> set_intrinsics."""
    cams = {}
    for f in C.sensor_calib_files(calib_dir):
        T, meta = C.load_sensor_calib(f)
        pipe.set_calib(int(meta["cam_id"]), T)
        cams[int(meta["cam_id"])] = meta.get("meta", {}).get("method", "?")
    for f in sorted(glob.glob(os.path.join(calib_dir, "bg_*.npz"))):
        pipe.cam(int(os.path.basename(f)[3:-4])).bg.load(f)
    for f in sorted(glob.glob(os.path.join(calib_dir, "intrinsics_*.json"))):
        with open(f, "r", encoding="utf-8") as fh:
            pipe.set_intrinsics(P.Intrinsics(**json.load(fh)))
    return cams


def calib_mismatch(rd, pipe: Pipeline) -> list[str]:
    """Cameras whose clip-time calibration differs from the loaded one by > 1 cm or > 0.5 deg."""
    out = []
    for cid, T in rd.calib.items():
        cur = pipe.cam(cid).T_ws
        if cur is None:
            continue
        R = T[:3, :3].T @ cur[:3, :3]
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        shift = float(np.linalg.norm(T[:3, 3] - cur[:3, 3]))
        if shift > 0.01 or ang > 0.5:
            out.append(f"cam {cid}: {shift * 100:.1f} cm / {ang:.2f} deg")
    return out


def cache_clip(rd, pipe: Pipeline, max_steps: int | None = None, learn_bg: int = 0, clip_calib: bool = False, log=print) -> list[Step]:
    """Run the whole clip through the pipeline once; the viewer then scrubs the cached steps."""
    for k in rd.intrinsics.values():
        pipe.set_intrinsics(k)
    for cid, T in rd.calib.items():
        if clip_calib or pipe.cam(cid).T_ws is None:
            pipe.set_calib(cid, T)
    if learn_bg:
        pipe.learn_background(frames=learn_bg)
    steps: list[Step] = []
    t0 = time.perf_counter()
    for t, packets, q in rd:
        pipe.set_q(q, 0.0)
        dec, dbg = pipe.step(packets, t)
        steps.append(snapshot(pipe, dbg, dec, t, q, 0.0))
        if max_steps and len(steps) >= max_steps:
            break
    if steps:
        log(f"cached {len(steps)} steps ({steps[-1].t - steps[0].t:.1f} s of clip) in {time.perf_counter() - t0:.1f} s")
    return steps


def trails(steps: list[Step], k: int, seconds: float = 3.0) -> dict:
    """{track id: (n,3) floor positions over the last `seconds` up to step k}."""
    out: dict[int, list] = {}
    t_k = steps[k].t
    for s in steps[max(0, k - 400):k + 1]:
        if t_k - s.t > seconds:
            continue
        for tr in s.tracks:
            if tr.confirmed:
                out.setdefault(tr.id, []).append([tr.xy[0], tr.xy[1], 0.03])
    return {i: np.asarray(v, np.float32) for i, v in out.items() if len(v) > 1}


def track_style(tr, score: float, current: int | None, hint: str):
    if not tr.confirmed:
        return COL_TENT, f"#{tr.id} tentative {tr.hits}/{tr.misses}"
    col = COL_CONF
    if tr.id == current:
        col = COL_REACH if (hint == "reach" or tr.over_rail) else COL_CUR
    elif tr.over_rail:
        col = COL_REACH
    return col, f"#{tr.id} S={score:.2f} E={tr.energy:.2f} h={tr.height:.2f}"


def depth_ramp(P: np.ndarray, origin) -> np.ndarray:
    """Near-to-far colour ramp over 5 m from the camera."""
    d = np.linalg.norm(P - np.asarray(origin, np.float32), axis=1)
    t = np.clip(d / 5.0, 0, 1)[:, None]
    return (NEAR_COL * (1 - t) + FAR_COL * t).astype(np.float32)


# ---- sources ---------------------------------------------------------------------------------------
class ClipSource:
    def __init__(self, steps: list[Step], name: str):
        self.steps, self.name = steps, name
        self.times = np.array([s.t for s in steps])
        self.k, self.playing, self.speed = 0, True, 1.0
        self._t = 0.0

    def tick(self, dt: float):
        if not self.playing or len(self.steps) < 2:
            return
        self._t += dt * self.speed
        span = self.times[-1] - self.times[0]
        if self._t > span:
            self._t = 0.0
        self.k = int(np.clip(np.searchsorted(self.times, self.times[0] + self._t), 0, len(self.steps) - 1))

    def seek(self, k: int):
        self.k = int(np.clip(k, 0, len(self.steps) - 1))
        self._t = float(self.times[self.k] - self.times[0])

    def current(self) -> Step | None:
        return self.steps[self.k] if self.steps else None

    def trails(self) -> dict:
        return trails(self.steps, self.k) if self.steps else {}

    def stop(self):
        pass


class LiveSource:
    """The tracker's own receive loop (mirror ports) feeding the viewer's pipeline, plus a mirror of what the
    tracker sent out (target_port) and a watch on the calibration/background files (relearn, absorption
    save, recalibration reload within a second)."""

    CALIB_PATTERNS = ("sensor_*.json", "bg_*.npz", "intrinsics_*.json")

    def __init__(self, pipe: Pipeline, in_port: int, q_port: int, target_port: int | None = None, calib_dir: str | None = None):
        from rt_depth_tracker.pacing import Pacer
        from rt_depth_tracker.run import DepthRx, RobotRx, TargetRx
        self.pipe, self.name = pipe, f"live /depth:{in_port} /robot/q:{q_port}" + (f" /tracker:{target_port}" if target_port else "")
        self.rx, self.qrx, self.pacer = DepthRx(in_port), RobotRx(q_port), Pacer()
        self.trx = TargetRx(target_port) if target_port else None
        self.target_port = target_port
        self.mirror_seen = False
        self.calib_dir = calib_dir
        self._calib_stamp = self._stamp_calib()
        self._last_calib_check = 0.0
        self.calib_reloaded_at: str | None = None
        self.calib_reloads = 0
        self.ring: deque = deque(maxlen=120)
        self.latest: Step | None = None
        self.new_intrinsics: dict = {}
        self.playing, self.speed = True, 1.0

    def _stamp_calib(self) -> dict:
        if not self.calib_dir:
            return {}
        out = {}
        for pat in self.CALIB_PATTERNS:
            for f in glob.glob(os.path.join(self.calib_dir, pat)):
                try:
                    out[f] = os.path.getmtime(f)
                except OSError:
                    pass
        return out

    def _maybe_reload_calib(self, now: float):
        if not self.calib_dir or now - self._last_calib_check < 1.0:
            return
        self._last_calib_check = now
        stamp = self._stamp_calib()
        if stamp == self._calib_stamp:
            return
        try:
            load_calib_into(self.pipe, self.calib_dir)
        except Exception:  # noqa: BLE001  (a file still being written: try again next second)
            return
        self._calib_stamp = stamp
        self.calib_reloaded_at = time.strftime("%H:%M:%S")
        self.calib_reloads += 1

    def tick(self, dt: float):
        self.rx.event.wait(0.0)
        self.rx.event.clear()
        pk, intr = self.rx.take(max_age=0.5)
        for k in intr.values():
            self.pipe.set_intrinsics(k)
            self.new_intrinsics[k.cam_id] = k
        q, age = self.qrx.get()
        self.pipe.set_q(q, age)
        now = time.monotonic()
        self._maybe_reload_calib(now)
        fresh = self.pacer.offer(pk, now)
        if not fresh:
            return
        dec, dbg = self.pipe.step(pk, now)
        sent = sent_tracks = sent_zones = sent_age = None
        if self.trx is not None:
            tgt, sent_age, rows, rage = self.trx.get()
            zones, zage = self.trx.get_zones()
            self.mirror_seen = self.mirror_seen or sent_age is not None
            sent = tgt if (sent_age is not None and sent_age < 0.5) else None
            sent_tracks = rows if (rage is not None and rage < 0.5) else None
            sent_zones = zones if (zage is not None and zage < 0.5) else None
        self.latest = snapshot(self.pipe, dbg, dec, now, q, age, sent=sent, sent_age=sent_age, sent_tracks=sent_tracks, sent_zones=sent_zones)
        self.ring.append(self.latest)

    def current(self) -> Step | None:
        return self.latest

    def trails(self) -> dict:
        return trails(list(self.ring), len(self.ring) - 1) if self.ring else {}

    def stop(self):
        self.rx.stop()
        self.qrx.stop()
        if self.trx is not None:
            self.trx.stop()


# ---- GL renderer -------------------------------------------------------------------------------------
VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCol;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uPointSize;
out vec3 vCol;
void main() {
    gl_Position = uProj * uView * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vCol = aCol;
}
"""
FRAG = """#version 330 core
in vec3 vCol;
out vec4 FragColor;
void main() { FragColor = vec4(vCol, 1.0); }
"""


class Renderer:
    def __init__(self):
        from OpenGL import GL as gl
        self.gl = gl
        self.prog = self._program(VERT, FRAG)
        self.u_view = gl.glGetUniformLocation(self.prog, "uView")
        self.u_proj = gl.glGetUniformLocation(self.prog, "uProj")
        self.u_ps = gl.glGetUniformLocation(self.prog, "uPointSize")
        self.bufs: dict[str, list] = {}       # name -> [vao, vbo, count, mode, point_size]
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _program(self, vs_src, fs_src):
        gl = self.gl
        prog = gl.glCreateProgram()
        for kind, src in ((gl.GL_VERTEX_SHADER, vs_src), (gl.GL_FRAGMENT_SHADER, fs_src)):
            sh = gl.glCreateShader(kind)
            gl.glShaderSource(sh, src)
            gl.glCompileShader(sh)
            if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
                raise RuntimeError(f"shader error: {gl.glGetShaderInfoLog(sh).decode()}")
            gl.glAttachShader(prog, sh)
            gl.glDeleteShader(sh)
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            raise RuntimeError(f"link error: {gl.glGetProgramInfoLog(prog).decode()}")
        return prog

    def upload(self, name: str, xyz: np.ndarray, color, mode: str = "points", point_size: float = 2.0):
        import ctypes
        gl = self.gl
        xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
        col = np.asarray(color, np.float32)
        col = np.broadcast_to(col.reshape(-1, 3), xyz.shape) if col.size in (3, xyz.size) else col.reshape(-1, 3)
        data = np.ascontiguousarray(np.hstack([xyz, col]).astype(np.float32))
        if name not in self.bufs:
            vao, vbo = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
            gl.glBindVertexArray(0)
            self.bufs[name] = [vao, vbo, 0, gl.GL_POINTS, point_size]
        b = self.bufs[name]
        b[2], b[3], b[4] = len(xyz), (gl.GL_LINES if mode == "lines" else gl.GL_POINTS), point_size
        if len(xyz):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b[1])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def clear(self, name: str):
        if name in self.bufs:
            self.bufs[name][2] = 0

    def draw(self, view: np.ndarray, proj: np.ndarray, visible, point_scale: float = 1.0):
        gl = self.gl
        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(self.u_view, 1, gl.GL_TRUE, np.ascontiguousarray(view, np.float32))
        gl.glUniformMatrix4fv(self.u_proj, 1, gl.GL_TRUE, np.ascontiguousarray(proj, np.float32))
        for name, (vao, _vbo, n, mode, ps) in self.bufs.items():
            if n == 0 or not visible(name):
                continue
            gl.glUniform1f(self.u_ps, ps * point_scale if mode == gl.GL_POINTS else 1.0)
            gl.glBindVertexArray(vao)
            gl.glDrawArrays(mode, 0, n)
        gl.glBindVertexArray(0)


# ---- app -----------------------------------------------------------------------------------------------
class App:
    PANEL_W, PAD = 320.0, 10.0

    def __init__(self, cfg: TrackerConfig, pipe: Pipeline, source, calib_methods: dict, hidden: bool = False,
                 title: str = "Depth Tracker Viewer", calib_dir: str | None = None, config_path: str | None = None):
        import glfw
        from imgui_bundle import imgui
        from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
        self.glfw, self.imgui = glfw, imgui
        self.cfg, self.pipe, self.source, self.calib_methods = cfg, pipe, source, calib_methods
        self.calib_dir, self.config_path = calib_dir, config_path
        if not glfw.init():
            raise SystemExit("glfwInit failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        if hidden:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(1400, 860, title, None, None)
        if not self.window:
            glfw.terminate()
            raise SystemExit("glfwCreateWindow failed (no display / no OpenGL 3.3?)")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        imgui.style_colors_dark()
        self.rend = Renderer()
        self.view = ViewControl()
        self._home_view()
        self.layers = {k: True for k in LAYERS}
        self.layers["local"] = False        # the mirror is the truth; the local guess is opt-in
        self.color_by = 0                       # camera | depth | cluster
        self.point_scale = 1.0
        self.quit = False
        self.snap_n = 0
        self._last = time.perf_counter()
        self._static_cams: set = set()
        self._labels: list = []
        self._reloads_seen = 0
        self.nudge = [0.0, 0.0, 0.0]            # yaw deg, dx, dy applied live on top of the loaded calibration
        self._nudge_base: dict = {}             # cam -> T_ws the sliders act on
        self.nudge_msg = ""
        self.zedit = zone_edits(cfg)
        self.zedit_msg = ""
        self.rend.upload("static", static_lines(cfg), COL_STATIC, "lines")
        ax, ac = axes_lines((0, 0, 0), 0.5)
        bx, bc = axes_lines((cfg.base_xy[0], cfg.base_xy[1], cfg.mount_z), 0.3)
        self.rend.upload("axes", np.vstack([ax, bx]), np.vstack([ac, bc]), "lines")
        zx, zc = zone_lines(cfg)
        self.rend.upload("zones", zx, zc, "lines")
        self._refresh_static()

    def _home_view(self):
        """From +x looking back at the origin, +y to the right, 35 deg down."""
        self.view.reset()
        self.view.yaw, self.view.pitch, self.view.distance = 0.0, 35.0, 7.0
        self.view.target = np.array([1.2, 0.0, 0.8])

    # ---- static geometry that needs intrinsics + calibration ----
    def _refresh_static(self):
        reloads = getattr(self.source, "calib_reloads", 0)
        if reloads != self._reloads_seen:                      # new calibration on disk: re-place the clouds and frustums
            self._reloads_seen = reloads
            self._static_cams.clear()
            self._nudge_base = {}
            self.nudge = [0.0, 0.0, 0.0]
        for c, cs in self.pipe.cams.items():
            if c in self._static_cams or cs.intr is None or cs.T_ws is None:
                continue
            self._static_cams.add(c)
            self.rend.upload(f"frustum{c}", frustum_lines(cs.intr, cs.T_ws, self.cfg.stride, 0.3, 2.5), COL_FRUSTUM, "lines")
            self.rend.upload(f"bg{c}", bg_cloud(cs.bg, cs.intr, cs.T_ws, self.cfg.stride), COL_BG.get(c, (0.35, 0.35, 0.35)), "points", 1.5)

    def _visible(self, name: str) -> bool:
        if name.startswith("frustum"):
            return self.layers["frustums"]
        if name == "axes":
            return self.layers["static"]
        if name.startswith("track") or name == "tracks":
            return self.layers["tracks"]
        if name == "zones_on":
            return self.layers["zones"]
        return self.layers.get(name, True)

    # ---- per-step dynamic geometry ----
    def _upload_step(self, st: Step | None):
        r = self.rend
        self._labels = []
        for name in ("pts0", "pts1", "crop", "robot_pts", "robot", "clusters", "tracks", "gaze", "pose", "target", "local",
                     "sent_tracks", "trails", "zones_on"):
            r.clear(name)
        if st is None:
            return
        for c in (0, 1):
            m = st.cam_ids == c
            Pc = st.points[m]
            if self.color_by == 1:
                cs = self.pipe.cams.get(c)
                col = depth_ramp(Pc, cs.T_ws[:3, 3] if cs is not None and cs.T_ws is not None else (0, 0, 0)) if len(Pc) else COL_CAM[c]
            elif self.color_by == 2:
                lab = st.labels[m]
                col = np.where((lab >= 0)[:, None], PALETTE[np.abs(lab) % len(PALETTE)], 0.35) if len(Pc) else COL_CAM[c]
            else:
                col = COL_CAM[c]
            r.upload(f"pts{c}", Pc, col, "points", 2.0)
        crop = [d["crop"] for d in st.dropped.values() if len(d["crop"])]
        rob = [d["robot"] for d in st.dropped.values() if len(d["robot"])]
        r.upload("crop", np.vstack(crop) if crop else np.zeros((0, 3)), COL_CROP, "points", 2.0)
        r.upload("robot_pts", np.vstack(rob) if rob else np.zeros((0, 3)), COL_ROBOT_PTS, "points", 2.0)
        if st.q_age is not None and self.pipe.robot is not None:
            cs_, rs_ = self.pipe.robot.sphere_world(st.q)
            r.upload("robot", np.vstack([sphere_wire(c, rr, 16) for c, rr in zip(cs_, rs_)]), COL_ROBOT, "lines")
        # cluster boxes: grey rejected component, white scored cluster
        scored = {cl.label for cl in st.clusters}
        cxyz, ccol = [], []
        for lab, (lo, hi) in st.boxes.items():
            e = box_edges(lo, hi)
            cxyz.append(e)
            col = COL_CAND if lab in scored else COL_REJECT
            if lab not in scored:
                z95 = float(np.percentile(st.points[st.labels == lab, 2], 95))
                if z95 > self.cfg.max_height:
                    col = COL_TALL
                    self._labels.append((hi, f"set piece h95={z95:.2f}", COL_TALL, 0))
            ccol.append(np.broadcast_to(col, e.shape))
        for cl in st.clusters:
            if cl.label in st.boxes:
                self._labels.append((st.boxes[cl.label][1], f"n={cl.n} h={cl.height_p95:.2f} r={cl.r:.2f}", COL_CAND, 0))
        if cxyz:
            r.upload("clusters", np.vstack(cxyz), np.vstack(ccol), "lines")
        # tracks
        hint = st.decision.hint if st.decision is not None else "idle"
        txyz, tcol, gaze, gcol, pose, pcol = [], [], [], [], [], []
        shoulder = np.array([self.cfg.base_xy[0], self.cfg.base_xy[1], self.cfg.mount_z])    # the pose line runs from the base
        for tr in st.tracks:
            col, text = track_style(tr, st.scores.get(tr.id, 0.0), st.current, hint)
            box = st.boxes.get(tr.label)
            if box is not None and tr.misses == 0:
                lo, hi = box
                e = box_edges(lo - 0.02, hi + 0.02)
                txyz.append(e)
                tcol.append(np.broadcast_to(col, e.shape))
                self._labels.append((hi, text, col, 1))
            else:
                self._labels.append((np.array([tr.xy[0], tr.xy[1], tr.height]), text, col, 1))
            if tr.confirmed:
                gaze += [tr.top, tr.reach_pt]
                gcol += [COL_CUR, COL_REACH]
                pose += [np.array([tr.xy[0], tr.xy[1], 0.0]), tr.top, shoulder, tr.top]
                pcol += [col, col, col, col]
        if txyz:
            r.upload("tracks", np.vstack(txyz), np.vstack(tcol), "lines")
        if gaze:
            r.upload("gaze", np.asarray(gaze), np.asarray(gcol), "points", 8.0)
        if pose:
            r.upload("pose", np.asarray(pose), np.asarray(pcol), "lines")
        sent = getattr(st, "sent", None)
        if sent is not None:                                    # the mirror: what the consumer receives is the truth
            if sent.hint not in QUIET:
                t = np.array([sent.x, sent.y, sent.z], np.float64)
                r.upload("target", cross_lines(t, 0.08, shoulder), COL_TARGET, "lines")
                self._labels.append((t + [0, 0, 0.12], f"-> out {sent.hint} S={sent.score:.2f} id={sent.track_id}", COL_TARGET, 2))
            if st.decision is not None and hint not in QUIET:
                t = np.asarray(st.decision.target, np.float64)
                r.upload("local", cross_lines(t, 0.05), COL_LOCAL, "lines")
                self._labels.append((t + [0, 0, 0.12], f"local {hint} S={st.decision.score:.2f}", COL_LOCAL, 3))
        elif st.decision is not None and hint not in QUIET:
            t = np.asarray(st.decision.target, np.float64)
            r.upload("target", cross_lines(t, 0.08, shoulder), COL_TARGET, "lines")
            self._labels.append((t + [0, 0, 0.12], f"{hint} S={st.decision.score:.2f} id={st.decision.track_id} (local)", COL_TARGET, 2))
        rows = getattr(st, "sent_tracks", None)
        if rows is not None and len(rows):
            dx, dcol = [], []
            for row in np.asarray(rows, np.float64):
                tid, top, S, flags = int(row[0]), row[1:4], float(row[5]), int(row[7])
                col = COL_REACH if (flags & 1 or (flags & 2 and sent is not None and sent.hint == "reach")) else (COL_CUR if flags & 2 else COL_CONF)
                dx.append(diamond_lines(top, 0.06))
                dcol.append(np.broadcast_to(col, (8, 3)))
                self._labels.append((top + [0, 0, 0.2], f"#{tid} S={S:.2f}" + ("*" if flags & 2 else "") + ("R" if flags & 1 else ""), col, 4))
            r.upload("sent_tracks", np.vstack(dx), np.vstack(dcol), "lines")
        tr_lines = []
        for tid, pts in self.source.trails().items():
            tr_lines.append(np.repeat(pts, 2, axis=0)[1:-1])
        if tr_lines:
            r.upload("trails", np.vstack(tr_lines), COL_CONF, "lines")
        # zones lit by the local pipeline (the mirror's view is in the panel)
        zx, zc = [], []
        for z, zone in zip(st.zones, zones_from_cfg(self.cfg)):
            if z.occupied or z.reaching or z.leaning:
                e = box_edges([zone.lo[0], zone.lo[1], zone.top_z + 0.005],
                              [zone.hi[0], zone.hi[1], zone.top_z + self.cfg.zone_margin_z + (0.3 if z.reaching or z.leaning else 0.0)])
                zx.append(e)
                zc.append(np.broadcast_to(COL_ZONE_ON if z.reaching or z.leaning else COL_CONF, e.shape))
        if zx:
            r.upload("zones_on", np.vstack(zx), np.vstack(zc), "lines")

    # ---- input ----
    def _handle_input(self):
        imgui, glfw = self.imgui, self.glfw
        io = imgui.get_io()
        if not io.want_capture_mouse:
            d = io.mouse_delta
            if imgui.is_mouse_down(imgui.MouseButton_.left):
                self.view.rotate(d.x, d.y)
            if imgui.is_mouse_down(imgui.MouseButton_.middle) or imgui.is_mouse_down(imgui.MouseButton_.right):
                self.view.pan(-d.x, d.y)
            if io.mouse_wheel:
                self.view.zoom(io.mouse_wheel)
        if io.want_capture_keyboard:
            return
        if imgui.is_key_pressed(imgui.Key.space, False):
            self.source.playing = not self.source.playing
        if imgui.is_key_pressed(imgui.Key.left_arrow, True) and hasattr(self.source, "seek"):
            self.source.playing = False
            self.source.seek(self.source.k - 1)
        if imgui.is_key_pressed(imgui.Key.right_arrow, True) and hasattr(self.source, "seek"):
            self.source.playing = False
            self.source.seek(self.source.k + 1)
        if imgui.is_key_pressed(imgui.Key.r, False):
            self._home_view()
        if imgui.is_key_pressed(imgui.Key.s, False):
            self.snap_n += 1
            fn = f"qa_viewer_{time.strftime('%Y%m%d_%H%M%S')}.png"
            self.screenshot(fn)
            print("saved", fn)
        if imgui.is_key_pressed(imgui.Key.q, False) or imgui.is_key_pressed(imgui.Key.escape, False):
            self.quit = True

    # ---- panels ----
    def _panel(self, name: str, y: float, body, x: float | None = None) -> float:
        imgui = self.imgui
        imgui.set_next_window_pos(imgui.ImVec2(self.PAD if x is None else x, y), imgui.Cond_.always)
        imgui.set_next_window_size_constraints(imgui.ImVec2(self.PANEL_W, 0), imgui.ImVec2(self.PANEL_W, 1e6))
        imgui.begin(name, None, imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move)
        body()
        h = imgui.get_window_size().y
        imgui.end()
        return h

    def _source_panel(self, st: Step | None):
        imgui, src = self.imgui, self.source
        imgui.text(src.name[:44])
        if isinstance(src, ClipSource):
            n = len(src.steps)
            t0 = src.times[0] if n else 0.0
            imgui.text(f"step {src.k + 1}/{n}   t = {(st.t - t0) if st else 0:.2f} s")
            if imgui.button("pause" if src.playing else "play"):
                src.playing = not src.playing
            imgui.same_line()
            _, src.speed = imgui.slider_float("speed", src.speed, 0.1, 4.0, "%.1fx")
            # hint strip over the clip
            dl = imgui.get_window_draw_list()
            p = imgui.get_cursor_screen_pos()
            w, hh = self.PANEL_W - 20, 8.0
            for i in range(int(w)):
                s = src.steps[min(n - 1, int(i / w * n))]
                hcol = HINT_COL.get(s.decision.hint if s.decision is not None else "idle", (0.4, 0.4, 0.4))
                dl.add_rect_filled(imgui.ImVec2(p.x + i, p.y), imgui.ImVec2(p.x + i + 1, p.y + hh),
                                   imgui.color_convert_float4_to_u32(imgui.ImVec4(*hcol, 1.0)))
            kx = p.x + (src.k / max(1, n - 1)) * w
            dl.add_rect_filled(imgui.ImVec2(kx - 1, p.y - 2), imgui.ImVec2(kx + 1, p.y + hh + 2),
                               imgui.color_convert_float4_to_u32(imgui.ImVec4(1, 1, 1, 1)))
            imgui.dummy(imgui.ImVec2(w, hh + 4))
            changed, k = imgui.slider_int("frame", src.k, 0, max(0, n - 1))
            if changed:
                src.playing = False
                src.seek(k)
        else:
            imgui.text(f"rx: {self.source.rx.rates()}")
        imgui.separator()
        cams = sorted(self.pipe.cams)
        imgui.text("cams: " + ", ".join(f"{c} {self.calib_methods.get(c, '?')}{'' if self.pipe.ready(c) else ' NOT READY'}" for c in cams))
        imgui.text("bg: " + ", ".join(f"{c} {'ready' if cs.bg.ready else 'MISSING'}" for c, cs in sorted(self.pipe.cams.items())))
        if st is not None:
            if self.pipe.robot is None:
                imgui.text("no robot in the config: self-mask off")
            elif st.q_age is None:
                imgui.text_colored(imgui.ImVec4(0.95, 0.3, 0.2, 1), "no /robot/q: robot mask OFF")
            else:
                stale = st.q_age > 0.5
                imgui.text_colored(imgui.ImVec4(0.95, 0.3, 0.2, 1) if stale else imgui.ImVec4(0.8, 0.8, 0.8, 1),
                                   f"q deg {np.round(np.degrees(st.q), 0).astype(int).tolist()}" + (f" STALE {st.q_age:.1f}s" if stale else ""))
            imgui.text(f"points {len(st.points)}  step {st.step_ms:.1f} ms")

    def _layers_panel(self):
        imgui = self.imgui
        _, self.color_by = imgui.combo("colour by", self.color_by, ["camera", "depth", "cluster"])
        _, self.point_scale = imgui.slider_float("point size", self.point_scale, 0.5, 4.0, "%.1f")
        for k in LAYERS:
            _, self.layers[k] = imgui.checkbox(LAYER_LABELS[k], self.layers[k])

    def _tracks_panel(self, st: Step | None):
        imgui = self.imgui
        if st is None:
            imgui.text("no step yet")
            return
        src = self.source
        if getattr(src, "trx", None) is not None:
            sent = getattr(st, "sent", None)
            if sent is not None:
                imgui.text_colored(imgui.ImVec4(*HINT_COL.get(sent.hint, (0.6, 0.6, 0.6)), 1.0),
                                   f"-> out: {sent.hint}  {[round(sent.x, 2), round(sent.y, 2), round(sent.z, 2)]}  S={sent.score:.2f} id={sent.track_id}  (mirror {src.target_port})")
                sz = getattr(st, "sent_zones", None)
                if sz:
                    on = [f"{z.name}:{'L' if z.leaning else 'R' if z.reaching else 'o' if z.occupied else '-'}" for z in sorted(sz.values(), key=lambda z: z.idx)]
                    imgui.text_colored(imgui.ImVec4(0.2, 0.9, 0.3, 1.0), "-> out zones " + " ".join(on))
            elif getattr(src, "mirror_seen", False):
                imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1), f"-> out: mirror {src.target_port} quiet")
            else:
                imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.2, 1), f"LOCAL pipeline only: no tracker mirror on {src.target_port} (tracker --out ...,{src.target_port})")
        rel = getattr(src, "calib_reloaded_at", None)
        if rel:
            imgui.text(f"calibration / background reloaded {rel}")
        hint = st.decision.hint if st.decision is not None else "idle"
        imgui.text_colored(imgui.ImVec4(*HINT_COL.get(hint, (0.6, 0.6, 0.6)), 1.0),
                           f"local: {hint}  target {np.round(st.decision.target, 2).tolist() if st.decision is not None else '-'}")
        if imgui.begin_table("tracks", 7, imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.sizing_fixed_fit):
            for h in ("id", "S", "E", "hit/miss", "h", "reach", "flags"):
                imgui.table_setup_column(h)
            imgui.table_headers_row()
            for tr in sorted(st.tracks, key=lambda t: -st.scores.get(t.id, 0.0)):
                col, _ = track_style(tr, st.scores.get(tr.id, 0.0), st.current, hint)
                imgui.table_next_row()
                cells = (f"#{tr.id}", f"{st.scores.get(tr.id, 0.0):.2f}", f"{tr.energy:.2f}", f"{tr.hits}/{tr.misses}",
                         f"{tr.height:.2f}", f"{tr.reach:.2f}",
                         ("C" if tr.confirmed else "t") + ("R" if tr.over_rail else "") + ("*" if tr.id == st.current else ""))
                for c in cells:
                    imgui.table_next_column()
                    imgui.text_colored(imgui.ImVec4(*col, 1.0), c)
            imgui.end_table()
        imgui.text(f"clusters {len(st.clusters)} of {len(st.boxes)} components")
        if st.zones:
            imgui.separator()
            for z in st.zones:
                col = imgui.ImVec4(*COL_ZONE_ON, 1.0) if (z.reaching or z.leaning) else imgui.ImVec4(*COL_CONF, 1.0) if z.occupied else imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
                imgui.text_colored(col, f"{z.idx} {z.name:8s} {'LEAN' if z.leaning else 'REACH' if z.reaching else 'occupied' if z.occupied else '-':8s} "
                                        f"#{z.track_id if z.track_id >= 0 else '-'} {z.dwell_s:5.1f}s")

    def _calib_panel(self):
        """Yaw / x / y sliders on top of the loaded calibration, applied to the pipeline at once; Save writes the files."""
        imgui = self.imgui
        cams = [c for c, cs in sorted(self.pipe.cams.items()) if cs.T_ws is not None]
        if not cams:
            imgui.text("no calibrated camera")
            return
        for c in cams:
            if c not in self._nudge_base:
                self._nudge_base[c] = self.pipe.cams[c].T_ws.copy()
        imgui.text("nudge the world about the origin")
        ch_y, self.nudge[0] = imgui.slider_float("yaw deg", self.nudge[0], -180.0, 180.0, "%.1f")
        ch_x, self.nudge[1] = imgui.slider_float("x m", self.nudge[1], -1.0, 1.0, "%.3f")
        ch_z, self.nudge[2] = imgui.slider_float("y m", self.nudge[2], -1.0, 1.0, "%.3f")
        if ch_y or ch_x or ch_z:
            self.apply_nudge()
        if imgui.button("save"):
            self.save_nudge()
        imgui.same_line()
        if imgui.button("reset"):
            self.nudge = [0.0, 0.0, 0.0]
            self.apply_nudge()
        if self.nudge_msg:
            imgui.text(self.nudge_msg[:44])

    def _zones_panel(self):
        """Sliders for every zone's footprint (centre, size, top, yaw) and the hand/head bands; applied live, Save writes
        the config file (the tracker reloads it)."""
        imgui = self.imgui
        changed = False
        imgui.push_item_width(170)
        if imgui.collapsing_header("bands"):
            for k, lo, hi, lab in (("zone_margin_z", 0.0, 0.2, "hand band from +m"), ("reach_z_max", 0.05, 0.6, "hand band to +m"),
                                   ("lean_z_lo", 0.05, 0.8, "head band from +m"), ("lean_z_hi", 0.2, 1.2, "head band to +m"),
                                   ("lean_bias_m", -0.3, 0.3, "head rect bias to the near side m"), ("lean_margin_m", 0.0, 0.4, "head rect grow m"),
                                   ("zone_near_m", 0.1, 1.0, "occupied within m")):
                ch, v = imgui.slider_float(lab, float(getattr(self.cfg, k)), lo, hi, "%.3f")
                if ch:
                    setattr(self.cfg, k, float(v))
                    changed = True
        for e in self.zedit:
            if imgui.collapsing_header(f"{e['name']}##{e['name']}"):
                for k, lo, hi, lab, fmt in (("cx", -3.0, 3.0, "centre x m", "%.3f"), ("cy", -3.0, 3.0, "centre y m", "%.3f"),
                                            ("w", 0.05, 3.0, "size x m", "%.3f"), ("h", 0.05, 3.0, "size y m", "%.3f"),
                                            ("top", 0.0, 1.5, "top z m", "%.3f"), ("yaw", -180.0, 180.0, "yaw deg", "%.0f")):
                    ch, e[k] = imgui.slider_float(f"{lab}##{e['name']}", float(e[k]), lo, hi, fmt)
                    changed |= ch
        imgui.pop_item_width()
        if changed:
            apply_zone_edits(self.cfg, self.zedit)
            zx, zc = zone_lines(self.cfg)
            self.rend.upload("zones", zx, zc, "lines")
        if imgui.button("save zones"):
            if not self.config_path:
                self.zedit_msg = "no --config: cannot save"
            else:
                try:
                    self.zedit_msg = save_zone_edits(self.config_path, self.cfg, self.zedit)
                except (OSError, ValueError) as ex:
                    self.zedit_msg = f"save failed: {ex}"
        imgui.same_line()
        if imgui.button("reload"):
            self.cfg.__dict__.update(TrackerConfig.load(self.config_path).__dict__)
            self.zedit = zone_edits(self.cfg)
            zx, zc = zone_lines(self.cfg)
            self.rend.upload("zones", zx, zc, "lines")
            self.zedit_msg = "reloaded from disk"
        if self.zedit_msg:
            imgui.text(self.zedit_msg[:60])

    def apply_nudge(self):
        W = C.yaw_T(self.nudge[0], self.nudge[1], self.nudge[2])
        for c, T0 in self._nudge_base.items():
            self.pipe.set_calib(c, W @ T0)
        self._static_cams.clear()        # the background clouds and frustums move with the pose

    def save_nudge(self):
        if not self.calib_dir:
            self.nudge_msg = "no --calib-dir: cannot save"
            return
        for c in self._nudge_base:
            path = os.path.join(self.calib_dir, f"sensor_{'abcdefgh'[c]}.json")
            prev = ""
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    prev = f.read()
                with open(path[:-5] + ".prev.json", "w", encoding="utf-8") as f:
                    f.write(prev)
            method = (json.loads(prev).get("meta", {}).get("method", "?") if prev else "?") + "+nudge"
            sha = C.save_sensor_calib(path, c, self.pipe.cams[c].T_ws, {"method": method, "yaw_deg": self.nudge[0], "dx": self.nudge[1], "dy": self.nudge[2]})
            self.nudge_msg = f"saved cam {c} {sha} (tracker reloads it within 1 s)"
            self.calib_methods[c] = method
        self._nudge_base = {c: self.pipe.cams[c].T_ws.copy() for c in self._nudge_base}
        self.nudge = [0.0, 0.0, 0.0]
        if isinstance(self.source, LiveSource):
            self.source._calib_stamp = self.source._stamp_calib()      # our own write is not a reload

    def _tracker_panel(self):
        imgui, c = self.imgui, self.cfg
        keys = ["crop", "z_max", "min_height", "max_height", "min_pts_at_3m", "min_pts", "gate", "confirm_hits",
                "bg_tol_mm", "bg_tol_sig", "interest_model", "engage_at", "still_energy", "near_pref", "kid_pref"]
        if self.pipe.robot is not None:
            keys[2:2] = ["robot_margin", "pedestal_radius"]
        if c.zones:
            keys += ["zone_near_m", "zone_hold_s"]
        for k in keys:
            imgui.text(f"{k}: {getattr(c, k)}")

    def _draw_labels(self, V, Pm, win_w, win_h):
        imgui = self.imgui
        if not self._labels:
            return
        show = {0: self.layers["clusters"], 1: self.layers["tracks"], 2: self.layers["target"], 3: self.layers["local"],
                4: self.layers["sent_tracks"]}
        pts = np.array([l[0] for l in self._labels], np.float64)
        scr, ok = project(pts, V, Pm, win_w, win_h)
        dl = imgui.get_foreground_draw_list()
        for (p3, text, col, kind), (sx, sy), vis in zip(self._labels, scr, ok):
            if vis and show[kind] and sx > self.PANEL_W + 2 * self.PAD:
                dl.add_text(imgui.ImVec2(float(sx) + 4, float(sy) - 14), imgui.color_convert_float4_to_u32(imgui.ImVec4(*col, 1.0)), text)

    # ---- frame ----
    def frame(self, draw_gui: bool = True):
        glfw, imgui, gl = self.glfw, self.imgui, self.rend.gl
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        self.impl.process_inputs()
        imgui.new_frame()
        self._handle_input()
        self.source.tick(min(dt, 0.25))
        self._refresh_static()
        st = self.source.current()
        self._upload_step(st)
        fw, fh = glfw.get_framebuffer_size(self.window)
        ww, wh = glfw.get_window_size(self.window)
        gl.glViewport(0, 0, fw, fh)
        gl.glClearColor(*CLEAR, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        V, Pm = self.view.view_matrix(), self.view.proj_matrix(fw / max(fh, 1))
        self.rend.draw(V, Pm, self._visible, self.point_scale)
        if draw_gui:
            y = self.PAD
            y += self._panel("Source", y, lambda: self._source_panel(st)) + self.PAD
            y += self._panel("Layers", y, self._layers_panel) + self.PAD
            y += self._panel("Tracks", y, lambda: self._tracks_panel(st)) + self.PAD
            y += self._panel("Calibration", y, self._calib_panel) + self.PAD
            self._panel("Tracker", y, self._tracker_panel)
            if self.cfg.zones:
                self._panel("Zones", 50, self._zones_panel, x=ww - self.PANEL_W - self.PAD)
            imgui.set_next_window_pos(imgui.ImVec2(ww - 230, 10), imgui.Cond_.always)
            imgui.set_next_window_bg_alpha(0.4)
            imgui.begin("##stats", None, imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize
                        | imgui.WindowFlags_.always_auto_resize | imgui.WindowFlags_.no_move)
            imgui.text(f"{imgui.get_io().framerate:.0f} FPS | {len(st.points) if st else 0} pts | {st.step_ms if st else 0:.1f} ms")
            imgui.end()
            self._draw_labels(V, Pm, ww, wh)
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)

    def screenshot(self, path: str):
        """Read the last rendered frame back (the front buffer after swap) and write a PNG."""
        import cv2
        gl, glfw = self.rend.gl, self.glfw
        fw, fh = glfw.get_framebuffer_size(self.window)
        gl.glReadBuffer(gl.GL_FRONT)
        buf = gl.glReadPixels(0, 0, fw, fh, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        gl.glReadBuffer(gl.GL_BACK)
        img = np.frombuffer(buf, np.uint8).reshape(fh, fw, 3)[::-1, :, ::-1]
        cv2.imwrite(path, np.ascontiguousarray(img))

    def run(self):
        glfw = self.glfw
        try:
            while not glfw.window_should_close(self.window) and not self.quit:
                glfw.poll_events()
                self.frame()
        finally:
            self.close()

    def close(self):
        try:
            self.source.stop()
        finally:
            if getattr(self.impl, "shutdown", None):
                self.impl.shutdown()
            self.glfw.destroy_window(self.window)
            self.glfw.terminate()


# ---- entry -------------------------------------------------------------------------------------------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs="*", default=None, help="recordings/<tag>_*.npz (globs ok)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--in", dest="in_port", type=int, default=7013, help="live: /depth mirror port")
    ap.add_argument("--q-port", type=int, default=7014, help="live: /robot/q mirror port (a joint feed sent to 7011 and 7014)")
    ap.add_argument("--target-port", type=int, default=P.PORT_TRACKER_MIRROR,
                    help="live: mirror of the tracker's output (tracker --out ...,7016): what the consumer receives; 0 = off")
    ap.add_argument("--config", default="config/tracker.json")
    ap.add_argument("--calib-dir", default="config/calib", help="sensor_*.json, bg_*.npz, intrinsics_*.json (overrides a clip's stored calibration)")
    ap.add_argument("--learn-bg", type=int, default=0, help="clip: learn the background from the first N frames instead of bg_*.npz")
    ap.add_argument("--clip-calib", action="store_true", help="clip: use the calibration stored in the recording, not --calib-dir's")
    ap.add_argument("--max-steps", type=int, default=0, help="clip: cache at most N steps (memory guard)")
    ap.add_argument("--screenshot", default=None, help="render one frame offscreen to this PNG and exit")
    ap.add_argument("--at", type=float, default=None, help="screenshot: seconds into the clip (default: the step with the most points)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    cfg = TrackerConfig.load(a.config)
    pipe = Pipeline(cfg, keep_dropped=True)
    methods = load_calib_into(pipe, a.calib_dir)
    if a.clip:
        from rt_depth_tracker.record import Reader
        paths = sorted(set(sum((glob.glob(p) for p in a.clip), [])))
        if not paths:
            raise SystemExit("no clip files match --clip")
        rd = Reader(paths)
        if not a.clip_calib:
            bad = calib_mismatch(rd, pipe)
            if bad:
                print("WARNING: the clip was recorded under a different calibration than " + a.calib_dir + " (" + "; ".join(bad) + ").\n"
                      "         If the cameras moved since, the current background will not match its frames either: "
                      "use --clip-calib --learn-bg N, or --calib-dir <backup from that time>.")
        steps = cache_clip(rd, pipe, a.max_steps or None, a.learn_bg, a.clip_calib)
        if not steps:
            raise SystemExit("clip produced no steps")
        source = ClipSource(steps, os.path.basename(paths[0]))
        if a.screenshot:
            if a.at is not None:
                source.seek(int(np.argmin(np.abs(source.times - (source.times[0] + a.at)))))
            else:
                first = min(a.learn_bg, len(steps) - 1)      # frames spent learning the background are all foreground
                source.seek(first + int(np.argmax([len(s.points) for s in steps[first:]])))
            source.playing = False
    elif a.live:
        source = LiveSource(pipe, a.in_port, a.q_port, a.target_port or None, a.calib_dir)
    else:
        raise SystemExit("one of --clip, --live")
    app = App(cfg, pipe, source, methods, hidden=bool(a.screenshot), calib_dir=a.calib_dir, config_path=a.config)
    if a.screenshot:
        try:
            for _ in range(3):          # ImGui lays out on the second frame; the third is clean
                app.glfw.poll_events()
                app.frame()
            app.screenshot(a.screenshot)
            st = source.current()
            k = getattr(source, "k", None)                       # live sources have no step index or time base
            t_rel = st.t - source.times[0] if getattr(source, "times", None) is not None else 0.0
            print(f"wrote {a.screenshot}: step {'live' if k is None else k} t={t_rel:.2f}s points={len(st.points)} "
                  f"clusters={len(st.clusters)} tracks={len(st.tracks)} hint={st.decision.hint}"
                  + (f" zones={[(z.name, z.occupied, z.reaching) for z in st.zones]}" if st.zones else ""))
        finally:
            app.close()
        return 0
    print(f"{source.name}: left-drag orbit, middle/right-drag pan, wheel zoom; space play/pause, arrows step, r reset, s PNG, q quit")
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
