# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Tracker parameters (config/tracker.json overrides these defaults; live-reloadable).

Frames: WORLD is Z-up, in metres, with its origin on the floor. A robot base, if there is one, stands at
base_xy and mount_z above the floor; the crop box and the cluster grid are centred on base_xy either way.
Sensor calibration files map sensor -> WORLD. A gitignored tracker.local.json next to tracker.json
overrides it (per-machine geometry)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class TrackerConfig:
    # scene: a plain default, the origin on the floor, nothing around it; everything below is opt-in
    mount_z: float = 0.0                # robot base height above the floor (0 = standing on it)
    base_xy: tuple = (0.0, 0.0)
    base_yaw_deg: float = 0.0           # robot base yaw in the world (0 = the base frame is the world frame)
    wall_x: float | None = None         # a back wall plane at x = wall_x, drawn by the viewers; None = none
    cams: dict = field(default_factory=lambda: {"0": {"pos": [-1.0, 1.5, 2.5], "aim": [1.5, 0.0, 0.8]},
                                                "1": {"pos": [-1.0, -1.5, 2.5], "aim": [1.5, 0.0, 0.8]}})
    rail_radius: float = 0.0            # a barrier at this horizontal distance from base_xy; 0 = none
    rail_top_z: float = 0.85            # anything above this and inside the rail is "reaching over"
    over_rail_inset: float = 0.15       # a torso leaning on the rail still has points ~0.1 m inside it
    pedestal_radius: float = 0.0        # a column under the robot base, masked with it; 0 = none
    marks: dict = field(default_factory=dict)   # {"name": [x, y]} floor marks for the yaw step, drawn by the viewers
    table: dict = field(default_factory=dict)   # {"center_xy", "size_xy", "top_z"} a rectangle of known size for the table step
    robot: dict = field(default_factory=dict)   # {"urdf", "robot_description"} paths (relative to this file); empty = no robot, no self-mask
    crop: bool = True                   # world crop below (z/area/x_min); off = the background model is the only spatial filter
    area_xy: float = 4.5                # drop points with |x| or |y| beyond this from base_xy
    x_min: float = -4.5                 # drop points with x - base_x below this (a wall behind the base); -area_xy = off
    z_min: float = 0.05                 # floor residue
    z_max: float = 2.9                  # above any raised hand, so a tall set piece keeps its top for max_height
    # zones: axis-aligned regions with a top (a table, a plinth, a case): who stands at one, who reaches over it
    zones: list = field(default_factory=list)   # [{"name", "rect": [x0, y0, x1, y1], "top_z", "yaw_deg"}]; empty = no zone output
    zone_margin_z: float = 0.02         # a point this far above a zone's top counts as reaching over it
    reach_z_max: float = 0.20           # ... up to this high: the hand band. Above it is the head band (lean) or nothing
    zone_near_m: float = 0.45           # a person whose nearest zone is closer than this "occupies" it
    zone_hold_s: float = 0.3            # occupied / reaching / leaning are held this long after the last hit (flicker)
    lean_bias_m: float = 0.12           # the lean rect is the footprint shifted this far along the zone's -Y (where people stand)
    lean_margin_m: float = 0.15         # and grown by this on every side: a head over the near edge is leaning in
    lean_z_lo: float = 0.20             # the head band above the zone top: a head this high ...
    lean_z_hi: float = 0.60             # ... to this high over the lean rect is leaning in (higher is standing)
    lean_min_s: float = 1.0             # a lean has to last this long before the zone reports it
    # capture
    stride: int = 2
    pair_window_s: float = 0.05         # max device-time gap to pool the two cameras
    # background
    bg_learn_frames: int = 300
    bg_tol_mm: float = 80.0            # re-placement and drift margin; only a hand flat on a table hides in it
    bg_tol_sig: float = 3.0
    bg_tol_frac: float = 0.01
    bg_valid_frac: float = 0.02        # sparse-return surfaces (glossy/dark) get a median instead of 'any return is fg'
    bg_ema_alpha: float = 0.002
    bg_empty_s: float = 120.0
    bg_absorb_s: float = 300.0          # a surface that holds still off the learned depth this long is healed into the SESSION
                                        # background (in memory, never saved; a person still that long too, until they move)
    bg_agree_frac: float = 0.5          # only a return inside this fraction of tol resets the absorb age: a surface nudged onto the tolerance must still heal
    bg_invalid_max_mm: float = 4500.0   # a return on a never-learned pixel counts only inside this range
    bg_unheal_s: float = 2.0            # the learned surface seen this long at a healed pixel un-heals it (a wall rolled away)
    # robot mask (only with a robot configured)
    robot_margin: float = 0.08
    robot_q_stale_s: float = 1.0        # no joint feed for this long: the pose we hold is fiction
    robot_stale_margin: float = 0.95    # ... so mask the whole reach instead
    robot_sphere_scale: float = 1.0     # multiplies every collision-sphere radius (2.0 = the crude "fat arm" option)
    robot_spheres_extra: dict = field(default_factory=dict)   # link -> [{center (link frame), radius}] on top of the YAML (a tool, the calibration ball)
    # clustering
    cell: float = 0.10
    min_pts_cell: int = 3
    dilate_cells: int = 0               # 1 bridges 0.2 m gaps (merges shoulder-to-shoulder people); 0 = off
    r_min_pct: float = 2.0              # percentile for the 'closest point' radius (single-point min jitters)
    min_pts_at_3m: int = 40             # points a person must have at 3 m from base_xy, scaled by 1/range^2 (cameras beside the base); 0 = off
    min_pts: int = 0                    # flat floor on the same count (an overhead camera sees every spot at the same range); 0 = off
    fragment_attach_m: float = 0.7      # undersized fragments (a hand, a sleeve) join the nearest person within this
    min_height: float = 0.6
    max_height: float = 2.44            # p95 above this (8 ft) is taller than any visitor: a set piece, not a person
    top_band: float = 0.15
    # tracking
    gate: float = 0.6
    alpha: float = 0.4
    beta: float = 0.1
    ema_tau: float = 0.3
    energy_tau: float = 0.5
    deriv_tau: float = 0.1              # fast smoothing for the motion-energy derivatives
    confirm_hits: int = 5
    lost_misses: int = 15
    delete_misses: int = 45
    # interest model
    interest_model: bool = True
    w_approach: float = 0.6
    w_motion: float = 0.5
    w_reach: float = 0.0               # reach is shape, not motion: a chair corner must not earn attention
    w_over_rail: float = 1.5
    over_rail_jump: float = 0.7
    over_rail_sticky: bool = True       # over-rail pins the score every frame, so a still leaner is never let go
    novelty: float = 0.5
    novelty_tau: float = 4.0
    proximity: float = 0.2
    prox_near: float = 1.4              # closeness ramp (gain multiplier and the near_pref rank): 1 at or inside this radius
    prox_far: float = 3.5               # ... 0 at or beyond this
    still_energy: float = 0.3
    decay_still_s: float = 8.0
    decay_active_s: float = 25.0
    bored_lo_s: float = 18.0
    bored_hi_s: float = 35.0
    bored_factor: float = 0.3
    refractory_s: float = 8.0
    refractory_cap: float = 0.3
    engage_at: float = 0.35
    watch_below: float = 0.6
    switch_margin: float = 0.25
    switch_hold_s: float = 1.0
    min_engaged_s: float = 3.0
    near_pref: float = 0.5            # rank boost at prox_near and closer, 0 at prox_far: the same activity counts more up close
    kid_pref: float = 0.4             # ... and from a child (0 = off)
    kid_h: float = 1.2                # full boost at or under this tracked height (head p95, ~4 cm under stature)
    adult_h: float = 1.7              # no boost at or over
    disengage_below: float = 0.05     # a still person is watched, not dropped: idle is the absolute floor
    disengage_hold_s: float = 10.0
    reach_thresh: float = 0.25
    # presence gate: nobody interesting -> sleep; an empty room -> short idle sweeps between sleeps
    sleep_gate: bool = False            # off by default so the interest scenarios keep asserting on raw hints
    doze_empty_s: float = 30.0          # look around this long after the last person leaves, then sleep
    doze_present_s: float = 0.0         # someone still here but boring: no sweeping at them, just settle
    sleep_lo_s: float = 120.0
    sleep_hi_s: float = 480.0           # asleep U(lo, hi) before each sweep: never a mechanical cadence
    sweep_lo_s: float = 45.0
    sweep_hi_s: float = 120.0           # 0 = no sweeps at all (with doze_empty_s 0: never idle, asleep at once)
    # output
    idle_target: tuple = (1.8, 0.0, 1.6)
    extra: dict = field(default_factory=dict)
    config_dir: str = field(default="", repr=False)   # where the file lives: relative paths in it (the robot files) resolve here

    @staticmethod
    def paths(path: str | None) -> list[str]:
        """[tracker.json, tracker.local.json] for a config path (the overlay is applied last)."""
        if not path:
            return []
        base, ext = os.path.splitext(path)
        return [path, base + ".local" + ext]

    @classmethod
    def load(cls, path: str | None) -> "TrackerConfig":
        cfg = cls()
        for p in cls.paths(path):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    cfg.apply(json.load(f))
                cfg.config_dir = os.path.dirname(os.path.abspath(p))
        return cfg

    def cam_pose(self, cam_id: int) -> tuple[list[float], list[float]]:
        c = self.cams.get(str(cam_id)) or self.cams.get(cam_id)
        if c is None:
            raise KeyError(f"no cams[{cam_id}] entry in the tracker config")
        return [float(v) for v in c["pos"]], [float(v) for v in c["aim"]]

    def resolve(self, path: str | None) -> str | None:
        """A path from the file, made absolute against the file's directory."""
        if not path:
            return None
        return path if os.path.isabs(path) or not self.config_dir else os.path.join(self.config_dir, path)

    def apply(self, d: dict):
        for k, v in d.items():
            if k in self.__dataclass_fields__ and k not in ("extra", "config_dir"):
                cur = getattr(self, k)
                if isinstance(cur, dict) and isinstance(v, dict):
                    setattr(self, k, {**cur, **v})        # overlays add or replace keys (a link's spheres, one camera), never wipe the rest
                elif isinstance(cur, list):
                    setattr(self, k, list(v))             # zones are replaced whole: an overlay cannot merge a list sensibly
                elif v is None or cur is None:
                    setattr(self, k, v)
                else:
                    setattr(self, k, tuple(v) if isinstance(cur, tuple) else type(cur)(v))
            else:
                self.extra[k] = v

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "config_dir"}


def save_defaults(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in TrackerConfig().to_dict().items() if k != "extra"}, f, indent=1)
