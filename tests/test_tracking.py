# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Clustering + tracker unit tests on synthetic world points (no rendering)."""
import os
import sys

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.segment import cluster_features, cluster_floor_grid, crop_world  # noqa: E402
from rt_depth_tracker.tracks import Tracker  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


rng = np.random.default_rng(0)
cfg = _scene.cfg()


def person_pts(x, y, h=1.7, n=600, r=0.18, torso=True):
    z = rng.uniform(0.1, h, n)
    if not torso:   # dark torso: no returns between 0.7 and 1.3 m
        z = z[(z < 0.7) | (z > 1.3)]
    a = rng.uniform(0, 2 * np.pi, len(z))
    return np.stack([x + r * np.cos(a), y + r * np.sin(a), z], 1).astype(np.float32)


print("crop")
P = np.array([[0, 2, 1.0], [0, 2, 0.02], [0, 2, 3.0], [5, 0, 1.0], [-1.0, 0, 1.0]], np.float32)
check("crop floor/ceiling/area/behind base plane", crop_world(P, cfg).tolist() == [True, False, False, False, False])
check("crop off keeps everything", crop_world(P, _scene.cfg(crop=False)).all())

print("clustering")
P = np.vstack([person_pts(0.3, 2.0), person_pts(-0.3, 2.0)])
labels, n = cluster_floor_grid(P, cfg)
cl = cluster_features(P, labels, None, cfg)
check("two people 0.6 m apart -> 2 clusters", len(cl) == 2, f"{len(cl)}")
P = person_pts(0.0, 2.0, torso=False)
labels, n = cluster_floor_grid(P, cfg)
cl = cluster_features(P, labels, None, cfg)
check("fragmented torso -> 1 cluster with full height", len(cl) == 1 and cl[0].height_p95 > 1.5, f"{len(cl)} {[c.height_p95 for c in cl]}")
P = np.vstack([person_pts(0.0, 2.0), rng.uniform(-4, 4, (40, 3)).astype(np.float32)])
labels, n = cluster_floor_grid(P, cfg)
cl = cluster_features(P, labels, None, cfg)
check("sparse noise ignored", len(cl) == 1 and cl[0].n >= 500)
child = person_pts(1.0, 2.5, h=1.0, n=200, r=0.12)
labels, n = cluster_floor_grid(child, cfg)
cl = cluster_features(child, labels, None, cfg)
check("child (1.0 m) kept", len(cl) == 1)
sign = person_pts(1.0, 2.5, h=0.5, n=200)
labels, n = cluster_floor_grid(sign, cfg)
check("0.5 m object dropped", len(cluster_features(sign, labels, None, cfg)) == 0)
body = person_pts(0.0, 1.55)
hand = np.array([[0.0, 1.0, 1.3]], np.float32) + rng.normal(0, 0.01, (30, 3)).astype(np.float32)
P = np.vstack([body, hand])
labels, n = cluster_floor_grid(P, cfg)
cl = cluster_features(P, labels, None, cfg)
check("reaching hand: over_rail + reach", len(cl) == 1 and cl[0].over_rail and cl[0].reach > 0.4 and abs(cl[0].reach_pt[1] - 1.0) < 0.05,
      f"{[(c.over_rail, round(c.reach,2)) for c in cl]}")
labels, n = cluster_floor_grid(body, cfg)
cl = cluster_features(body, labels, None, cfg)
check("leaning torso at rail is NOT over_rail", len(cl) == 1 and not cl[0].over_rail and cl[0].reach < 0.25)
cams = np.array([0] * 300 + [1] * 300, np.int8)
cl = cluster_features(body, np.zeros(600, np.int32), cams, cfg)
check("per-camera counts", cl[0].n_cam == {0: 300, 1: 300})
check("top_xy near head", np.linalg.norm(cl[0].top_xy - [0, 1.55]) < 0.1)
_st = rng.bit_generator.state
pillar = person_pts(0.5, 1.4, h=3.0, n=1500, r=0.25)             # p95 ~2.86: a set piece, not a visitor
labels, n = cluster_floor_grid(pillar, cfg)
check("3 m pillar dropped (max_height)", len(cluster_features(pillar, labels, None, cfg)) == 0)
body = person_pts(0.0, 2.0)
hand = np.array([[0.0, 1.4, 1.3]], np.float32) + rng.normal(0, 0.01, (30, 3)).astype(np.float32)   # 0.5 m from the pillar, 0.6 m from its body
P = np.vstack([body, hand, pillar])
labels, n = cluster_floor_grid(P, cfg)
cl = cluster_features(P, labels, None, cfg)
check("pillar beside a person: the person keeps its height and its hand", len(cl) == 1 and 1.5 < cl[0].height_p95 < 1.8 and cl[0].n == 630 and cl[0].reach > 0.4,
      f"{[(c.n, round(c.height_p95, 2), round(c.reach, 2)) for c in cl]}")
tall = person_pts(0.0, 2.0, h=1.9)
arm = np.stack([np.full(40, 0.15), np.full(40, 2.0), np.linspace(1.9, 2.5, 40)], 1).astype(np.float32)   # a hand up to 2.5 m
P = np.vstack([tall, arm])
labels, n = cluster_floor_grid(P, cfg)
cl = cluster_features(P, labels, None, cfg)
check("1.9 m person with a hand at 2.5 m is kept (p95, not max)", len(cl) == 1 and cl[0].height_p95 < cfg.max_height and P[:, 2].max() > cfg.max_height,
      f"{[(c.n, round(c.height_p95, 2)) for c in cl]}")
rng.bit_generator.state = _st

print("tracker")
tr = Tracker(cfg)
now, dt = 0.0, 1 / 30


def step(pts_list):
    global now
    P = np.vstack(pts_list) if pts_list else np.zeros((0, 3), np.float32)
    labels, _ = cluster_floor_grid(P, cfg)
    out = tr.update(cluster_features(P, labels, None, cfg), dt, now)
    now += dt
    return out


for i in range(10):
    tracks = step([person_pts(0.0, 3.0 - 0.02 * i)])
check("confirmed after 5 hits", len(tracks) == 1 and tracks[0].confirmed and tracks[0].id == 1)
check("velocity toward robot", tracks[0].v[1] < -0.2, str(tracks[0].v))
for i in range(10):
    tracks = step([])
check("10-frame dropout: still confirmed (coasting)", len(tracks) == 1 and tracks[0].misses == 10)
tracks = step([person_pts(0.0, 2.5)])
check("re-associated, same id, misses reset", len(tracks) == 1 and tracks[0].id == 1 and tracks[0].misses == 0)
for i in range(50):
    tracks = step([])
check("deleted after 45 misses", len(tracks) == 0 and len(tr.tracks) == 0)

tr = Tracker(cfg)
for i in range(60):
    x = -1.5 + 3.0 * i / 59
    tracks = step([person_pts(x, 2.0), person_pts(-x, 2.6)])
ids = sorted(t.id for t in tracks)
check("crossing: both tracks survive with distinct ids", len(tracks) == 2 and ids == [1, 2], str(ids))
a = [t for t in tracks if t.id == 1][0]
check("crossing: id 1 followed +x direction", a.xy[0] > 1.0 and abs(a.xy[1] - 2.0) < 0.3, str(a.xy))

tr = Tracker(cfg)
for i in range(8):
    tracks = step([person_pts(0.5, 2.0)])
tracks = step([person_pts(0.5 + 0.9, 2.0)])
check("0.9 m jump exceeds gate -> new tentative, old coasts", len(tr.tracks) == 2 and len(tracks) == 1 and tracks[0].id == 1)

tr = Tracker(cfg)
for i in range(90):
    ph = 0.4 * np.sin(2 * np.pi * 2 * i / 30)   # 2 Hz waving hand 0.4 m out
    hand = np.array([[0.5 + ph, 2.0 - 0.3, 1.4]], np.float32) + rng.normal(0, 0.01, (40, 3)).astype(np.float32)
    tracks = step([person_pts(0.5, 2.0), hand])
check("waving raises motion energy", tracks and tracks[0].energy > 0.3, f"{[round(t.energy,2) for t in tracks]}")
for i in range(60):
    tracks = step([person_pts(0.5, 2.0)])
check("standing still lowers energy", tracks and tracks[0].energy < 0.15, f"{[round(t.energy,2) for t in tracks]}")

print(f"test_tracking: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
