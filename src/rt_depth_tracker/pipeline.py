# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Pipeline: depth frames (per camera) [+ robot q] -> Decision. No sockets, no timing dependencies:
the same object runs live (run.py), on recordings (replay.py) and on synthetic scenes (tests)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker.background import BackgroundModel
from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.interest import Decision, InterestModel
from rt_depth_tracker.intrinsics import ray_table, unproject
from rt_depth_tracker.presence import SleepGate
from rt_depth_tracker.protocol import DepthPacket, Intrinsics
from rt_depth_tracker.robot_model import RobotModel
from rt_depth_tracker.segment import Cluster, cluster_features, cluster_floor_grid, crop_world
from rt_depth_tracker.tracks import Track, Tracker
from rt_depth_tracker.zones import ZoneState, ZoneTracker, zones_from_cfg


@dataclass
class CamState:
    cam_id: int
    T_ws: np.ndarray | None = None
    intr: Intrinsics | None = None
    rays: np.ndarray | None = None
    bg: BackgroundModel | None = None
    last: DepthPacket | None = None
    frames_seen: int = 0


@dataclass
class Debug:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    cam_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    clusters: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    zones: list = field(default_factory=list)      # ZoneState per configured zone (empty without zones)
    robot_pts_removed: int = 0
    bg_pts_removed: int = 0
    step_ms: float = 0.0
    scores: dict = field(default_factory=dict)
    dropped: dict = field(default_factory=dict)   # cam -> {"crop": (n,3), "robot": (n,3)}; only with keep_dropped
    pix: dict = field(default_factory=dict)       # cam -> (fg pixel idx, keep mask, robot mask); only with keep_dropped


class Pipeline:
    def __init__(self, cfg: TrackerConfig, robot: RobotModel | None = None, rng=None, keep_dropped: bool = False):
        self.cfg = cfg
        self.robot = robot if robot is not None else RobotModel.from_config(cfg)    # None: no robot, no self-mask
        self.keep_dropped = keep_dropped      # viewers keep what the crop and the robot mask removed; the live tracker never does
        self.cams: dict[int, CamState] = {}
        self.tracker = Tracker(cfg)
        self.zone_tracker = ZoneTracker(cfg)
        self.interest = InterestModel(cfg, rng=rng)
        self.sleep = SleepGate(cfg, rng=self.interest.rng)
        self.q = np.zeros(self.robot.n_dof if self.robot is not None else 6)
        self.q_age = None
        self._last_t = None
        self._empty_since = None
        self.absorbed = 0            # pixels healed since the runtime last logged it (session overlay, never saved)
        self.absorbed_total = 0
        self.unhealed = 0
        self.unhealed_total = 0

    # ---- setup ---------------------------------------------------------------------------------
    def cam(self, cam_id: int) -> CamState:
        if cam_id not in self.cams:
            self.cams[cam_id] = CamState(cam_id, bg=BackgroundModel(self.cfg.bg_tol_mm, self.cfg.bg_tol_sig, self.cfg.bg_tol_frac,
                                                                     self.cfg.bg_valid_frac, self.cfg.bg_ema_alpha, self.cfg.bg_empty_s,
                                                                     self.cfg.bg_absorb_s, self.cfg.bg_invalid_max_mm,
                                                                     agree_frac=self.cfg.bg_agree_frac, unheal_s=self.cfg.bg_unheal_s))
        return self.cams[cam_id]

    def set_calib(self, cam_id: int, T_world_sensor: np.ndarray):
        self.cam(cam_id).T_ws = np.asarray(T_world_sensor, np.float64)

    def set_intrinsics(self, intr: Intrinsics):
        cs = self.cam(intr.cam_id)
        if cs.intr is None or cs.intr.__dict__ != intr.__dict__:
            cs.intr = intr
            cs.rays = None

    def set_q(self, q, age: float | None = 0.0):
        self.q = np.asarray(q, np.float64)
        self.q_age = age

    def ready(self, cam_id: int) -> bool:
        cs = self.cam(cam_id)
        return cs.T_ws is not None and cs.intr is not None

    # ---- per frame --------------------------------------------------------------------------------
    def _rays_for(self, cs: CamState, stride: int):
        if cs.rays is None or cs.rays.shape[:2] != (cs.intr.height // stride, cs.intr.width // stride):
            cs.rays = ray_table(cs.intr, stride)
        return cs.rays

    def points_for(self, pkt: DepthPacket, dbg: Debug | None = None):
        """World points for one packet after background subtraction, crop and (with a robot) robot masking."""
        cs = self.cam(pkt.cam_id)
        cs.last, cs.frames_seen = pkt, cs.frames_seen + 1
        if cs.bg.learning:
            cs.bg.feed_learning(pkt.depth)
        if not self.ready(pkt.cam_id) or cs.bg.bg is None:
            # no calibration or no background yet: blind, not 'everything is foreground' (the furniture would become people);
            # same contract as the geometry mismatch below: the mask is always depth-shaped
            return np.zeros((0, 3), np.float32), np.zeros(pkt.depth.shape, bool)
        rays = self._rays_for(cs, pkt.stride)
        if rays.shape[:2] != pkt.depth.shape or cs.bg.bg.shape != pkt.depth.shape:
            # rays or a saved background disagree with the frame (wrong stride/mode): never unproject garbage
            return np.zeros((0, 3), np.float32), np.zeros(pkt.depth.shape, bool)
        fg = cs.bg.foreground(pkt.depth)
        P_s, idx = unproject(pkt.depth, rays, fg)
        if dbg is not None:
            dbg.bg_pts_removed += int(np.count_nonzero(pkt.depth > 0) - len(P_s))
        P_all = C.apply(cs.T_ws, P_s).astype(np.float32)
        keep = crop_world(P_all, self.cfg)
        P = P_all[keep]
        rob = np.zeros(len(P), bool)
        if self.robot is not None and len(P):
            # a pose nobody has confirmed for a while, or none ever received, is not a pose: grow every
            # sphere to the arm's reach so the arm cannot leak through wherever it actually went
            stale = self.q_age is None or self.q_age > self.cfg.robot_q_stale_s
            margin = self.cfg.robot_stale_margin if stale else self.cfg.robot_margin
            rob = self.robot.inside_mask(P, self.q, margin, self.cfg.pedestal_radius)
            if dbg is not None:
                dbg.robot_pts_removed += int(rob.sum())
            P = P[~rob]
        if self.keep_dropped and dbg is not None:
            dbg.dropped[pkt.cam_id] = {"crop": P_all[~keep], "robot": P_all[keep][rob]}
            dbg.pix[pkt.cam_id] = (idx, keep, rob)
        return P, fg

    def step(self, packets: dict[int, DepthPacket], t: float | None = None) -> tuple[Decision, Debug]:
        t0 = time.perf_counter()
        now = time.monotonic() if t is None else float(t)
        dt = 1 / 30.0 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now
        dbg = Debug()
        pts, ids, fgs = [], [], {}
        for cam_id, pkt in packets.items():
            P, fg = self.points_for(pkt, dbg)
            fgs[cam_id] = (pkt, fg)
            if len(P):
                pts.append(P)
                ids.append(np.full(len(P), cam_id, np.int8))
        P = np.vstack(pts) if pts else np.zeros((0, 3), np.float32)
        cam_ids = np.concatenate(ids) if ids else np.zeros(0, np.int8)
        labels, _ = cluster_floor_grid(P, self.cfg)
        zones = zones_from_cfg(self.cfg)
        clusters = cluster_features(P, labels, cam_ids, self.cfg, zones)
        tracks = self.tracker.update(clusters, dt, now)
        zone_states = self.zone_tracker.update(tracks, zones, dt) if zones else []
        decision = self.interest.update(tracks, dt, now)
        decision.hint = self.sleep.update(decision.hint, len(tracks), dt)
        scene_empty = len(clusters) == 0
        for cam_id, (pkt, fg) in fgs.items():
            cs = self.cams[cam_id]
            if cs.bg.ready and fg.shape == pkt.depth.shape:
                cs.bg.slow_update(pkt.depth, fg, scene_empty, now)
                n = cs.bg.absorb(pkt.depth, dt)
                self.absorbed += n
                self.absorbed_total += n
                u = cs.bg.pop_unhealed()
                self.unhealed += u
                self.unhealed_total += u
        dbg.points, dbg.cam_ids, dbg.labels, dbg.clusters, dbg.tracks, dbg.zones = P, cam_ids, labels, clusters, tracks, zone_states
        dbg.scores = dict(self.interest.S)
        dbg.step_ms = (time.perf_counter() - t0) * 1e3
        return decision, dbg

    # ---- background management -------------------------------------------------------------------
    def learn_background(self, cam_id: int | None = None, frames: int | None = None):
        n = frames or self.cfg.bg_learn_frames
        for cid, cs in self.cams.items():
            if cam_id is None or cid == cam_id:
                cs.bg.begin_learning(n)

    def bg_status(self) -> dict:
        return {cid: {"ready": cs.bg.ready, "learning": cs.bg.learning, "age_s": cs.bg.age_s(), "healed_px": cs.bg.healed_px}
                for cid, cs in self.cams.items()}

    def tracks_rows(self, tracks: list[Track]) -> np.ndarray:
        rows = [[t.id, t.top[0], t.top[1], t.top[2], t.height, self.interest.S.get(t.id, 0.0), t.reach,
                 (1 if t.over_rail else 0) | (2 if t.id == self.interest.current else 0)] for t in tracks]
        return np.asarray(rows, np.float32).reshape(-1, 8)
