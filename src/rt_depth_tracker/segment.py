# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""World-frame point filtering and floor-grid clustering with per-cluster features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.zones import Zone, cluster_zones


@dataclass
class Cluster:
    centroid: np.ndarray      # (3,)
    floor_xy: np.ndarray      # (2,) mean xy of the lower body band (or centroid xy)
    height_p95: float         # z of the 95th percentile (top of head-ish)
    top_xy: np.ndarray        # (2,) mean xy of the top band (gaze point)
    r_min: float              # smallest horizontal distance from base_xy
    reach_pt: np.ndarray      # (3,) the point closest to base_xy
    reach: float              # r_centroid - r_min
    spread: float             # xy std
    extent: float             # 98th pct horizontal distance from centroid (arms out -> large)
    over_rail: bool
    n: int
    n_cam: dict
    label: int = -1
    reach_mask: int = 0       # bit i: some point hangs over zone i above its top
    lean_mask: int = 0        # bit i: the head is over zone i's near edge
    nearest_zone: int = -1
    zone_dist: float = float("inf")
    hand_pt: np.ndarray | None = field(default=None, repr=False)   # (3,) the point above the nearest zone's top closest to its centre

    @property
    def r(self) -> float:
        return float(np.hypot(self.centroid[0], self.centroid[1]))


def crop_world(P: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    """Boolean keep-mask for points in the tracked volume."""
    if not cfg.crop:
        return np.ones(len(P), bool)
    return ((P[:, 2] > cfg.z_min) & (P[:, 2] < cfg.z_max) & (np.abs(P[:, 0] - cfg.base_xy[0]) < cfg.area_xy)
            & (np.abs(P[:, 1] - cfg.base_xy[1]) < cfg.area_xy) & (P[:, 0] - cfg.base_xy[0] > cfg.x_min))


def min_pts_at(cfg: TrackerConfig, d: float) -> float:
    """Points a person must have at horizontal distance d from base_xy: the range-scaled rule (cameras beside the
    base see a far person smaller) and the flat floor (an overhead camera sees everyone at the same range)."""
    scaled = cfg.min_pts_at_3m * (3.0 / max(d, 0.5)) ** 2 if cfg.min_pts_at_3m > 0 else 0.0
    return max(scaled, float(cfg.min_pts))


def cluster_floor_grid(P: np.ndarray, cfg: TrackerConfig):
    """Project to the floor grid, dilate one cell, connected components. Returns labels (N,) with -1 for
    points in cells below min_pts_cell."""
    if len(P) == 0:
        return np.zeros(0, dtype=np.int32), 0
    half = cfg.area_xy
    cell = cfg.cell
    n_cells = int(np.ceil(2 * half / cell)) + 1
    ix = np.clip(((P[:, 0] - cfg.base_xy[0] + half) / cell).astype(np.int32), 0, n_cells - 1)
    iy = np.clip(((P[:, 1] - cfg.base_xy[1] + half) / cell).astype(np.int32), 0, n_cells - 1)
    flat = ix * n_cells + iy
    counts = np.bincount(flat, minlength=n_cells * n_cells).reshape(n_cells, n_cells)
    occ = counts >= cfg.min_pts_cell
    occ_d = ndimage.binary_dilation(occ, structure=np.ones((3, 3), bool), iterations=cfg.dilate_cells) if cfg.dilate_cells > 0 else occ
    lab, n = ndimage.label(occ_d, structure=np.ones((3, 3), int))
    labels = lab[ix, iy].astype(np.int32) - 1
    labels[~occ[ix, iy]] = -1
    return labels, int(n)


def attach_fragments(P: np.ndarray, labels: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    """Relabel undersized / too-low components onto the nearest full-size person within
    fragment_attach_m (horizontal). Keeps a reaching hand or a dark sleeve with its body.
    Components taller than max_height are set pieces: nothing attaches to them and they attach to
    nobody, so their label survives for cluster_features to reject and the viewer to draw."""
    n_lab = labels.max() + 1 if len(labels) else 0
    if n_lab <= 1:
        return labels
    bx, by = cfg.base_xy
    stats = []
    for lab in range(n_lab):
        m = labels == lab
        n = int(m.sum())
        if n == 0:
            stats.append(None)
            continue
        Q = P[m]
        c = Q.mean(0)
        d = float(np.hypot(c[0] - bx, c[1] - by))
        z95 = float(np.percentile(Q[:, 2], 95))
        tall = z95 > cfg.max_height
        keep = not tall and n >= min_pts_at(cfg, d) and z95 >= cfg.min_height
        stats.append((c, keep, tall))
    keepers = [(lab, s[0]) for lab, s in enumerate(stats) if s is not None and s[1]]
    if not keepers:
        return labels
    out = labels.copy()
    for lab, s in enumerate(stats):
        if s is None or s[1] or s[2]:   # a keeper or a set piece attaches to nobody
            continue
        dists = [(float(np.hypot(s[0][0] - c[0], s[0][1] - c[1])), kl) for kl, c in keepers]
        dmin, kl = min(dists)
        if dmin <= cfg.fragment_attach_m:
            out[labels == lab] = kl
    return out


def cluster_features(P: np.ndarray, labels: np.ndarray, cam_ids: np.ndarray | None, cfg: TrackerConfig,
                     zones: list[Zone] | None = None) -> list[Cluster]:
    out = []
    zones = zones or []
    bx, by = cfg.base_xy
    labels = attach_fragments(P, labels, cfg)
    for lab in range(labels.max() + 1 if len(labels) else 0):
        m = labels == lab
        n = int(m.sum())
        if n == 0:
            continue
        Q = P[m]
        r_all = np.hypot(Q[:, 0] - bx, Q[:, 1] - by)
        centroid = Q.mean(0)
        d = float(np.hypot(centroid[0] - bx, centroid[1] - by))
        z95 = float(np.percentile(Q[:, 2], 95))
        if n < min_pts_at(cfg, d) or z95 < cfg.min_height or z95 > cfg.max_height:
            continue
        top = Q[Q[:, 2] > z95 - cfg.top_band]
        top_xy = top[:, :2].mean(0) if len(top) else centroid[:2]
        low = Q[Q[:, 2] < min(1.0, z95 * 0.6)]
        floor_xy = low[:, :2].mean(0) if len(low) >= 5 else centroid[:2]
        i_min = int(np.argmin(r_all))
        r_min = float(np.percentile(r_all, cfg.r_min_pct))
        reach = float(np.hypot(centroid[0] - bx, centroid[1] - by) - r_min)
        over_rail = cfg.rail_radius > 0 and bool(((r_all < cfg.rail_radius - cfg.over_rail_inset) & (Q[:, 2] > cfg.rail_top_z)).any())
        reach_mask, lean_mask, nearest_zone, zone_dist = cluster_zones(Q, floor_xy, zones, cfg, top_xy, z95)
        hand_pt = None
        if nearest_zone >= 0:
            z = zones[nearest_zone]
            c_xy = (z.lo + z.hi) / 2.0
            above = np.flatnonzero(Q[:, 2] > z.top_z + cfg.zone_margin_z)
            k = above[int(np.argmin(np.hypot(Q[above, 0] - c_xy[0], Q[above, 1] - c_xy[1])))] if len(above) else i_min
            hand_pt = Q[k].copy()
        n_cam = {}
        if cam_ids is not None:
            ids, cnt = np.unique(cam_ids[m], return_counts=True)
            n_cam = {int(i): int(c) for i, c in zip(ids, cnt)}
        extent = float(np.percentile(np.hypot(Q[:, 0] - centroid[0], Q[:, 1] - centroid[1]), 98))
        out.append(Cluster(centroid=centroid, floor_xy=np.asarray(floor_xy), height_p95=z95, top_xy=np.asarray(top_xy),
                           r_min=r_min, reach_pt=Q[i_min].copy(), reach=reach, spread=float(Q[:, :2].std()), extent=extent,
                           over_rail=over_rail, n=n, n_cam=n_cam, label=lab, reach_mask=reach_mask, lean_mask=lean_mask,
                           nearest_zone=nearest_zone, zone_dist=zone_dist, hand_pt=hand_pt))
    return out
