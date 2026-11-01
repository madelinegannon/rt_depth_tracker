# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Synthetic depth renderer: floor, optional back wall, rail and pedestal, boxes (a table slab, plinths), robot
spheres (FK) when a robot is given, people as vertical capsules -> uint16 depth images through the real camera
model. Used by the tests and to develop the pipeline before any hardware exists."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker.intrinsics import ray_table
from rt_depth_tracker.protocol import DepthPacket, Intrinsics
from rt_depth_tracker.robot_model import RobotModel

FEMTO_NFOV_BINNED = Intrinsics(0, 320, 288, 251.5, 251.5, 160.0, 144.0)   # a 320x288 ToF depth mode, ~75x65 deg


@dataclass
class Person:
    xy: tuple
    height: float = 1.7
    radius: float = 0.18
    arm: tuple | None = None       # (x, y, z, radius) extra sphere for a reaching hand
    extra: list = field(default_factory=list)   # more spheres (x, y, z, r), e.g. a waving arm

    def with_arm(self, azimuth_deg: float, elevation_deg: float, length: float = 0.6, n: int = 5) -> "Person":
        """Return a copy with an arm of n spheres from the shoulder in the given direction."""
        sh = np.array([self.xy[0], self.xy[1], self.height - 0.35])
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pts = [tuple(sh + d * (self.radius + length * k / (n - 1))) + (0.06,) for k in range(n)]
        return Person(self.xy, self.height, self.radius, self.arm, list(self.extra) + pts)


@dataclass
class Scene:
    people: list = field(default_factory=list)
    q: np.ndarray = field(default_factory=lambda: np.zeros(6))
    robot: RobotModel | None = None    # rendered at q when given; its base height is the model's mount_z
    rail_radius: float = 0.0           # a ring of tube at rail_z around the origin; 0 = none
    rail_z: float = 0.8
    rail_tube: float = 0.03
    wall_x: float | None = None        # a vertical wall plane at x = wall_x; None = none
    pedestal_radius: float = 0.0       # a column from the floor to pedestal_z under the robot; 0 = none
    pedestal_z: float = 0.0
    table: dict | None = None          # {"center_xy", "size_xy", "top_z"}: rendered as a 0.1 m slab under top_z
    boxes: list = field(default_factory=list)   # [(lo (3,), hi (3,))] axis-aligned boxes, e.g. plinths on the table
    yaw_deg: float = 0.0               # rotate the table and boxes about the world z axis (calibration tests)
    offset_xy: tuple = (0.0, 0.0)      # ... and shift them
    noise_mm: float = 4.0
    balls: list = field(default_factory=list)      # (x, y, z, r) calibration balls (rendered in depth + coloured in render_rgb)


def _ray_plane(o, d, n, p0):
    denom = d @ n
    t = ((p0 - o) @ n) / np.where(np.abs(denom) < 1e-9, np.inf, denom)
    t[t <= 0] = np.inf
    return t


def _ray_sphere(o, d, c, r):
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    t = np.where(disc >= 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
    t[t <= 0] = np.inf
    return t


def _ray_vcyl(o, d, cx, cy, r, z0, z1):
    """Vertical cylinder (finite), side surface only."""
    ox, oy, oz = o
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (d[:, 0] * (ox - cx) + d[:, 1] * (oy - cy))
    c = (ox - cx) ** 2 + (oy - cy) ** 2 - r * r
    disc = b * b - 4 * a * c
    with np.errstate(all="ignore"):
        t = np.where((disc >= 0) & (a > 1e-12), (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    z = oz + t * d[:, 2]
    t = np.where((t > 0) & (z >= z0) & (z <= z1), t, np.inf)
    return t


def _ray_torus_approx(o, d, R, r, z, n_seg=48):
    """A rail as a ring of small spheres (cheap, good enough for a 3 cm tube)."""
    t = np.full(len(d), np.inf)
    for a in np.linspace(0, 2 * np.pi, n_seg, endpoint=False):
        t = np.minimum(t, _ray_sphere(o, d, np.array([R * np.cos(a), R * np.sin(a), z]), r * 1.5))
    return t


def _ray_aabb(o, d, lo, hi):
    """Slab test against an axis-aligned box; the entry distance, inf when missed or started inside."""
    with np.errstate(all="ignore"):
        inv = 1.0 / d
        t1, t2 = (lo - o) * inv, (hi - o) * inv
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)


def _ray_box_yawed(o, d, lo, hi, yaw_deg, offset_xy):
    """A box defined in the table frame, placed in the world by yaw about z and an xy shift."""
    if yaw_deg == 0.0 and tuple(offset_xy) == (0.0, 0.0):
        return _ray_aabb(o, d, lo, hi)
    Rz = C.euler_xyz_deg_to_R(0, 0, yaw_deg)
    off = np.array([offset_xy[0], offset_xy[1], 0.0])
    o_t = Rz.T @ (o - off)
    d_t = d @ Rz           # rows: Rz.T @ d_i
    return _ray_aabb(o_t, d_t, lo, hi)


def scene_boxes(scene: Scene) -> list:
    """[(lo, hi)] in the table frame: the slab plus the boxes."""
    out = []
    if scene.table:
        cx, cy = scene.table.get("center_xy", (0.0, 0.0))
        sx, sy = scene.table["size_xy"]
        top = float(scene.table["top_z"])
        out.append((np.array([cx - sx / 2, cy - sy / 2, top - 0.1]), np.array([cx + sx / 2, cy + sy / 2, top])))
    for lo, hi in scene.boxes:
        out.append((np.asarray(lo, float), np.asarray(hi, float)))
    return out


def render(scene: Scene, T_world_sensor: np.ndarray, intr: Intrinsics = FEMTO_NFOV_BINNED, stride: int = 2,
           rng=None, far_m: float = 6.0) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    rays = ray_table(intr, stride).reshape(-1, 3).astype(np.float64)
    R, o = T_world_sensor[:3, :3], T_world_sensor[:3, 3]
    d = rays @ R.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = _ray_plane(o, d, np.array([0, 0, 1.0]), np.array([0, 0, 0.0]))              # floor
    if scene.wall_x is not None:
        t = np.minimum(t, _ray_plane(o, d, np.array([1.0, 0, 0]), np.array([scene.wall_x, 0, 0])))
    if scene.rail_radius > 0:
        t = np.minimum(t, _ray_torus_approx(o, d, scene.rail_radius, scene.rail_tube, scene.rail_z))
    if scene.pedestal_radius > 0 and scene.pedestal_z > 0:
        t = np.minimum(t, _ray_vcyl(o, d, 0, 0, scene.pedestal_radius, 0, scene.pedestal_z))
    if scene.robot is not None:
        cs, rs = scene.robot.sphere_world(scene.q)
        for c, r in zip(cs, rs):
            t = np.minimum(t, _ray_sphere(o, d, c, r))
    for lo, hi in scene_boxes(scene):
        t = np.minimum(t, _ray_box_yawed(o, d, lo, hi, scene.yaw_deg, scene.offset_xy))
    for b in scene.balls:
        t = np.minimum(t, _ray_sphere(o, d, np.array(b[:3]), b[3]))
    for p in scene.people:
        t = np.minimum(t, _ray_vcyl(o, d, p.xy[0], p.xy[1], p.radius, 0.0, p.height - p.radius))
        t = np.minimum(t, _ray_sphere(o, d, np.array([p.xy[0], p.xy[1], p.height - p.radius]), p.radius))
        if p.arm is not None:
            t = np.minimum(t, _ray_sphere(o, d, np.array(p.arm[:3]), p.arm[3]))
        for ex in p.extra:
            t = np.minimum(t, _ray_sphere(o, d, np.array(ex[:3]), ex[3]))
    # ToF depth = distance along the optical axis, not the ray range: z = t * cos(angle) = t / |ray|
    z_cam = t / np.linalg.norm(rays, axis=1)
    z_cam[~np.isfinite(t) | (t > far_m)] = 0
    z_mm = z_cam * 1000.0
    z_mm[z_mm > 0] += rng.normal(0, scene.noise_mm, size=int((z_mm > 0).sum()))
    return np.clip(z_mm, 0, 65535).astype(np.uint16).reshape(intr.height // stride, intr.width // stride)


def render_rgb(scene: Scene, T_world_sensor: np.ndarray, intr: Intrinsics = FEMTO_NFOV_BINNED, stride: int = 2,
               rng=None, ball_bgr=(0, 90, 255), bg_bgr=(140, 140, 140)):
    """Depth image plus a BGR image (grey scene, orange balls) aligned pixel-for-pixel with the depth."""
    rng = rng or np.random.default_rng(0)
    depth = render(scene, T_world_sensor, intr, stride, rng=rng)
    no_balls = Scene(**{**scene.__dict__, "balls": []})
    depth_nb = render(no_balls, T_world_sensor, intr, stride, rng=np.random.default_rng(0))
    h, w = depth.shape
    bgr = np.empty((h, w, 3), np.uint8)
    bgr[...] = bg_bgr
    ball_px = (depth > 0) & (depth_nb == 0) | ((depth > 0) & (depth_nb > 0) & (depth.astype(np.int32) < depth_nb.astype(np.int32) - 30))
    bgr[ball_px] = ball_bgr
    noise = rng.integers(-12, 13, size=bgr.shape, dtype=np.int16)
    return depth, np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8), ball_px


def two_camera_rig(spacing: float = 3.1, height: float = 3.0, x: float = -0.9, aim=(1.5, 0.0, 0.9)):
    """Two cameras `spacing` apart along y at x behind the origin, `height` up, both aimed at the same point in
    front of it (+x): cam 0 at -y, cam 1 at +y. Returns {cam_id: T_ws}."""
    return {0: C.look_at_T([x, -spacing / 2, height], aim),
            1: C.look_at_T([x, +spacing / 2, height], aim)}


def overhead_camera(height: float = 4.0, xy=(0.0, 0.0), aim_xy=None) -> np.ndarray:
    """One camera above the scene looking straight down (image 'up' = world +x). Returns T_ws."""
    aim = (xy[0], xy[1], 0.0) if aim_xy is None else (aim_xy[0], aim_xy[1], 0.0)
    return C.look_at_T([xy[0], xy[1], height], aim)


def packet(cam_id: int, depth: np.ndarray, seq: int, ts: float, stride: int = 2, intr: Intrinsics = FEMTO_NFOV_BINNED) -> DepthPacket:
    return DepthPacket(cam_id, seq, intr.width, intr.height, stride, ts, depth)


def intrinsics_for(cam_id: int, base: Intrinsics = FEMTO_NFOV_BINNED) -> Intrinsics:
    k = Intrinsics(**base.__dict__)
    k.cam_id = cam_id
    return k
