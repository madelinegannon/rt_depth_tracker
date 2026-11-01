# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Zones: axis-aligned rectangular regions in the world frame with a top height (a table, a plinth, a display
case). Per zone: who stands at it, whether a hand hangs over it above its top, whether a head leans in over
its near edge. Optional: nothing here runs unless the config lists zones."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rt_depth_tracker.config import TrackerConfig


@dataclass
class Zone:
    idx: int
    name: str
    lo: np.ndarray      # (2,) x0, y0
    hi: np.ndarray      # (2,) x1, y1
    top_z: float
    yaw_deg: float = 0.0    # world bearing of the zone's +X; its -Y side (6 o'clock) is where people stand

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    def minus_y(self) -> np.ndarray:
        """Unit vector of the zone's -Y in the world."""
        a = np.radians(self.yaw_deg)
        return np.array([np.sin(a), -np.cos(a)])


@dataclass
class ZoneState:
    idx: int
    name: str
    occupied: bool
    reaching: bool
    track_id: int       # the reaching track, else the leaning one, else the nearest occupying one, else -1
    dwell_s: float      # time occupied, reached or leaned over without a gap longer than zone_hold_s
    leaning: bool = False   # a head over the person's own edge of the zone


def zones_from_cfg(cfg: TrackerConfig) -> list[Zone]:
    out = []
    for i, z in enumerate(cfg.zones):
        x0, y0, x1, y1 = [float(v) for v in z["rect"]]
        lo, hi = np.array([min(x0, x1), min(y0, y1)]), np.array([max(x0, x1), max(y0, y1)])
        top = float(z.get("top_z", (cfg.table or {}).get("top_z", 0.9)))
        out.append(Zone(i, str(z.get("name", f"z{i}")), lo, hi, top, float(z.get("yaw_deg", 0.0))))
    return out


def rect_distance(p_xy, lo, hi) -> float:
    """Horizontal distance from a point to an axis-aligned rectangle (0 inside)."""
    d = np.maximum(np.maximum(lo - p_xy, p_xy - hi), 0.0)
    return float(np.hypot(d[0], d[1]))


def lean_rect(z: Zone, cfg: TrackerConfig) -> tuple[np.ndarray, np.ndarray]:
    """The footprint shifted lean_bias_m along the zone's -Y (where people stand) and grown by lean_margin_m:
    a head over the near edge is leaning in without hanging over the whole zone."""
    shift = z.minus_y() * cfg.lean_bias_m
    return z.lo + shift - cfg.lean_margin_m, z.hi + shift + cfg.lean_margin_m


def cluster_zones(Q: np.ndarray, floor_xy, zones: list[Zone], cfg: TrackerConfig, top_xy=None, z95: float | None = None
                  ) -> tuple[int, int, int, float]:
    """(reach_mask, lean_mask, nearest_zone, nearest_dist). Two height bands above a zone's top: the hand band
    (zone_margin_z .. reach_z_max) sets the reach bit when any point over the footprint is in it; the head band
    (lean_z_lo .. lean_z_hi) sets the lean bit when the head (top_xy) is over the lean rect at that height. A leaning
    person's arms are over the zone too, so a lean suppresses that zone's reach bit."""
    if not zones:
        return 0, 0, -1, float("inf")
    mask, lean, best, best_d = 0, 0, -1, float("inf")
    head = None if top_xy is None or z95 is None else np.asarray(top_xy, float)
    for z in zones:
        d = rect_distance(np.asarray(floor_xy, float), z.lo, z.hi)
        if d < best_d:
            best, best_d = z.idx, d
        if head is not None and z.top_z + cfg.lean_z_lo < z95 <= z.top_z + cfg.lean_z_hi:
            lo, hi = lean_rect(z, cfg)
            if (lo <= head).all() and (head <= hi).all():
                lean |= 1 << z.idx
                continue
        inside = (Q[:, 0] >= z.lo[0]) & (Q[:, 0] <= z.hi[0]) & (Q[:, 1] >= z.lo[1]) & (Q[:, 1] <= z.hi[1])
        if (inside & (Q[:, 2] > z.top_z + cfg.zone_margin_z) & (Q[:, 2] <= z.top_z + cfg.reach_z_max)).any():
            mask |= 1 << z.idx
    return mask, lean, best, best_d


class ZoneTracker:
    """Per-zone occupied / reaching / leaning with a short hold and a dwell clock."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._held: dict[tuple, list] = {}        # (zone, "occ"|"reach"|"lean") -> [seconds since the last hit, track id]
        self._dwell: dict[int, float] = {}
        self._lean_s: dict[int, float] = {}       # zone -> how long the (held) lean has lasted: reported after lean_min_s

    def _hold(self, key: tuple, hit_id: int, dt: float) -> int:
        """Latch a hit for zone_hold_s; returns the held track id or -1."""
        if hit_id >= 0:
            self._held[key] = [0.0, hit_id]
            return hit_id
        h = self._held.get(key)
        if h is None:
            return -1
        h[0] += dt
        if h[0] > self.cfg.zone_hold_s:
            del self._held[key]
            return -1
        return h[1]

    def update(self, tracks, zones: list[Zone], dt: float) -> list[ZoneState]:
        c = self.cfg
        out = []
        for z in zones:
            occ_id, reach_id, lean_id = -1, -1, -1
            occ_d = float("inf")
            for t in tracks:
                if not t.confirmed:
                    continue
                if t.reach_mask & (1 << z.idx) and reach_id < 0:
                    reach_id = t.id
                if t.lean_mask & (1 << z.idx) and lean_id < 0:
                    lean_id = t.id
                if t.nearest_zone == z.idx and t.zone_dist < c.zone_near_m and t.zone_dist < occ_d:
                    occ_id, occ_d = t.id, t.zone_dist
            occ_id = self._hold((z.idx, "occ"), occ_id, dt)
            reach_id = self._hold((z.idx, "reach"), reach_id, dt)
            lean_id = self._hold((z.idx, "lean"), lean_id, dt)
            self._lean_s[z.idx] = self._lean_s.get(z.idx, 0.0) + dt if lean_id >= 0 else 0.0
            if self._lean_s[z.idx] < c.lean_min_s:
                lean_id = -1
            if occ_id >= 0 or reach_id >= 0 or lean_id >= 0:
                self._dwell[z.idx] = self._dwell.get(z.idx, 0.0) + dt
            else:
                self._dwell[z.idx] = 0.0
            tid = reach_id if reach_id >= 0 else lean_id if lean_id >= 0 else occ_id
            out.append(ZoneState(z.idx, z.name, occ_id >= 0, reach_id >= 0, tid, self._dwell.get(z.idx, 0.0), lean_id >= 0))
        return out
