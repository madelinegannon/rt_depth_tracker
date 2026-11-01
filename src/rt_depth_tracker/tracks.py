# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Multi-target tracking on the floor plane: gated nearest-neighbour association, alpha-beta
position/velocity, EMA features, tentative/confirmed/lost lifecycle, motion energy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.segment import Cluster


@dataclass
class Track:
    id: int
    xy: np.ndarray
    v: np.ndarray
    height: float
    reach: float
    spread: float
    top: np.ndarray            # (3,) gaze point
    reach_pt: np.ndarray       # (3,)
    over_rail: bool = False
    energy: float = 0.0
    hits: int = 1
    misses: int = 0
    age_s: float = 0.0
    confirmed: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    n: int = 0
    r_centroid: float = 0.0
    extent: float = 0.0
    label: int = -1            # cluster label of the last hit (viewers draw the track's box from it)
    reach_mask: int = 0        # zones: bit i = a hand over zone i
    lean_mask: int = 0         # zones: bit i = the head over zone i's near edge
    nearest_zone: int = -1
    zone_dist: float = float("inf")
    hand_pt: np.ndarray | None = field(default=None, repr=False)
    _f_reach: float = field(default=0.0, repr=False)     # fast-EMA features for derivatives
    _f_spread: float = field(default=0.0, repr=False)
    _f_extent: float = field(default=0.0, repr=False)

    @property
    def r(self) -> float:
        return float(np.hypot(self.xy[0], self.xy[1]))

    @property
    def lost(self) -> bool:
        return self.misses > 0


class Tracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, clusters: list[Cluster], dt: float, now: float) -> list[Track]:
        c = self.cfg
        dt = max(float(dt), 1e-3)
        # predict
        for t in self.tracks:
            t.xy = t.xy + t.v * dt
            t.age_s += dt
        # associate (greedy nearest, gated); many-to-one allowed so a merged pair keeps its older id
        unassigned = list(range(len(clusters)))
        pairs = []
        if self.tracks and clusters:
            D = np.array([[np.linalg.norm(t.xy - cl.floor_xy) for cl in clusters] for t in self.tracks])
            order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
            used_t = set()
            for ti, ci in order:
                if D[ti, ci] > c.gate:
                    break
                if ti in used_t or ci not in unassigned:
                    continue
                used_t.add(ti)
                unassigned.remove(ci)
                pairs.append((ti, ci))
        matched_t = {ti for ti, _ in pairs}
        for ti, ci in pairs:
            self._correct(self.tracks[ti], clusters[ci], dt, now)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.misses += 1
                t.energy *= np.exp(-dt / c.energy_tau)
        for ci in unassigned:
            cl = clusters[ci]
            self.tracks.append(Track(self._next_id, cl.floor_xy.astype(float).copy(), np.zeros(2), cl.height_p95, cl.reach,
                                     cl.spread, np.array([cl.top_xy[0], cl.top_xy[1], cl.height_p95]), cl.reach_pt.copy(),
                                     cl.over_rail, first_seen=now, last_seen=now, n=cl.n, r_centroid=cl.r, extent=cl.extent,
                                     label=cl.label, reach_mask=cl.reach_mask, lean_mask=cl.lean_mask, nearest_zone=cl.nearest_zone,
                                     zone_dist=cl.zone_dist, hand_pt=cl.hand_pt, _f_reach=cl.reach, _f_spread=cl.spread, _f_extent=cl.extent))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses < c.delete_misses]
        return [t for t in self.tracks if t.confirmed and t.misses < c.lost_misses]

    def _correct(self, t: Track, cl: Cluster, dt: float, now: float):
        c = self.cfg
        z = cl.floor_xy.astype(float)
        r = z - t.xy
        t.xy = t.xy + c.alpha * r
        t.v = t.v + (c.beta / dt) * r
        speed = float(np.linalg.norm(t.v))
        if speed > 3.0:
            t.v *= 3.0 / speed
        k = 1.0 - np.exp(-dt / c.ema_tau)
        t.height += k * (cl.height_p95 - t.height)
        t.reach += k * (cl.reach - t.reach)
        t.spread += k * (cl.spread - t.spread)
        t.extent += k * (cl.extent - t.extent)
        # derivatives from a FAST EMA: kills single-frame edge jitter, keeps 1-3 Hz gestures
        kf = 1.0 - np.exp(-dt / c.deriv_tau)
        f_reach = t._f_reach + kf * (cl.reach - t._f_reach)
        f_spread = t._f_spread + kf * (cl.spread - t._f_spread)
        f_extent = t._f_extent + kf * (cl.extent - t._f_extent)
        d_reach = (f_reach - t._f_reach) / dt
        d_spread = (f_spread - t._f_spread) / dt
        d_extent = (f_extent - t._f_extent) / dt
        t._f_reach, t._f_spread, t._f_extent = f_reach, f_spread, f_extent
        t.top = np.array([cl.top_xy[0], cl.top_xy[1], cl.height_p95])
        t.reach_pt = cl.reach_pt.copy()
        t.over_rail = cl.over_rail
        t.r_centroid = cl.r
        t.n = cl.n
        t.label = cl.label
        t.reach_mask, t.lean_mask, t.nearest_zone, t.zone_dist, t.hand_pt = cl.reach_mask, cl.lean_mask, cl.nearest_zone, cl.zone_dist, cl.hand_pt
        inst = min(2.0, float(np.sqrt(speed ** 2 + d_reach ** 2 + d_spread ** 2 + d_extent ** 2)))
        ke = 1.0 - np.exp(-dt / c.energy_tau)
        t.energy += ke * (inst - t.energy)
        t.hits += 1
        t.misses = 0
        t.last_seen = now
        if not t.confirmed and t.hits >= c.confirm_hits:
            t.confirmed = True
