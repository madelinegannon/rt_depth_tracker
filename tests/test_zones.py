# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Zones end to end on a synthetic overhead scene (one camera above a table with two boxes on it, no robot): a
person standing at a zone occupies it, an arm over it sets the reach bit, the other zone stays untouched, the hold
suppresses one-frame gaps and the dwell clock resets after a real departure. Also the ZoneTracker rules on
hand-made tracks, and that nothing zone-shaped leaves the pipeline without zones in the config."""
import os
import sys

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.pipeline import Pipeline  # noqa: E402
from rt_depth_tracker.tracks import Track  # noqa: E402
from rt_depth_tracker.zones import ZoneTracker, cluster_zones, lean_rect, rect_distance, zones_from_cfg  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


TABLE = {"center_xy": [0.0, 0.0], "size_xy": [2.0, 1.2], "top_z": 0.9}
BOXES = [([-0.8, -0.4, 0.9], [-0.2, 0.4, 0.95]), ([0.2, -0.4, 0.9], [0.8, 0.4, 0.95])]
OVERHEAD = dict(min_pts_at_3m=0, min_pts=40, area_xy=3.5, x_min=-3.5, z_max=2.4, min_height=0.5)   # one camera 4 m up: same range everywhere
cfg = TrackerConfig(table=TABLE, zones=[{"name": "a", "rect": [-0.8, -0.4, -0.2, 0.4], "top_z": 0.95},
                                        {"name": "b", "rect": [0.2, -0.4, 0.8, 0.4], "top_z": 0.95}], **OVERHEAD)
zones = zones_from_cfg(cfg)

print("geometry helpers")
check("rect_distance inside is 0, outside is the gap", rect_distance(np.array([0.0, 0.0]), zones[0].lo, zones[0].hi) > 0
      and rect_distance(np.array([-0.5, 0.0]), zones[0].lo, zones[0].hi) == 0 and abs(rect_distance(np.array([-0.5, -0.8]), zones[0].lo, zones[0].hi) - 0.4) < 1e-9)
Q = np.array([[-0.5, -0.8, 1.0], [-0.5, 0.0, 1.1]])
check("cluster_zones: a point over zone a above its top sets bit 0", cluster_zones(Q, [-0.5, -0.8], zones, cfg)[::2] == (1, 0))
check("below the margin: no bit", cluster_zones(np.array([[-0.5, 0.0, 0.96]]), [-0.5, -0.8], zones, cfg)[0] == 0)
check("a hand above the hand band (top + 40 cm) is not a reach", cluster_zones(np.array([[-0.5, 0.0, 1.35]]), [-0.5, -0.8], zones, cfg)[0] == 0)
check("no zones -> nothing", cluster_zones(Q, [-0.5, -0.8], [], cfg) == (0, 0, -1, float("inf")))
lo, hi = lean_rect(zones[0], cfg)
check("lean rect: shifted along the zone's -Y (yaw 0 = world -y) and grown", abs(lo[1] + 0.67) < 1e-9 and abs(hi[1] - 0.43) < 1e-9 and abs(lo[0] + 0.95) < 1e-9, f"{lo} {hi}")
z90 = zones_from_cfg(TrackerConfig(zones=[{"name": "r", "rect": [-0.1, -0.1, 0.1, 0.1], "top_z": 0.9, "yaw_deg": 90}]))[0]
lo, hi = lean_rect(z90, cfg)
check("yaw 90: the zone's -Y is world +x, the rect shifts that way", abs(lo[0] - (-0.1 + 0.12 - 0.15)) < 1e-9 and abs(lo[1] + 0.25) < 1e-9, f"{lo} {hi}")
check("head in the head band over the near edge: lean bit, no reach bit", cluster_zones(np.array([[-0.5, -0.85, 1.0]]), [-0.5, -0.85], zones, cfg, [-0.5, -0.55], 1.3)[:2] == (0, 1))
check("a leaning person's low points over the zone do not reach (lean wins)", cluster_zones(np.array([[-0.5, -0.3, 1.0]]), [-0.5, -0.85], zones, cfg, [-0.5, -0.55], 1.3)[:2] == (0, 1))
check("head over the far edge: no lean bit", cluster_zones(np.array([[-0.5, -0.85, 1.0]]), [-0.5, -0.85], zones, cfg, [-0.5, 0.5], 1.3)[1] == 0)
check("a head above the head band is standing, not leaning", cluster_zones(np.array([[-0.5, -0.85, 1.0]]), [-0.5, -0.85], zones, cfg, [-0.5, -0.55], 1.7)[1] == 0)
check("a head below the head band (top + 10 cm) is not a lean either", cluster_zones(np.array([[-0.5, -0.85, 1.0]]), [-0.5, -0.85], zones, cfg, [-0.5, -0.55], 1.05)[1] == 0)
check("the default top_z comes from the table", zones_from_cfg(TrackerConfig(table=TABLE, zones=[{"name": "t", "rect": [0, 0, 1, 1]}]))[0].top_z == 0.9)

print("ZoneTracker rules")
zt = ZoneTracker(cfg)


def trk(id, zone, dist, mask, confirmed=True, lean=0):
    t = Track(id, np.zeros(2), np.zeros(2), 1.7, 0.0, 0.1, np.zeros(3), np.zeros(3), reach_mask=mask, nearest_zone=zone, zone_dist=dist, lean_mask=lean)
    t.confirmed = confirmed
    return t


st = zt.update([trk(1, 0, 0.3, 0)], zones, 1 / 30)
check("occupied by the nearest confirmed track", st[0].occupied and not st[0].reaching and st[0].track_id == 1 and not st[1].occupied)
st = zt.update([trk(1, 0, 0.3, 0), trk(2, 0, 0.1, 0b10)], zones, 1 / 30)
check("closest track occupies; b reached by track 2 but not occupied", st[0].track_id == 2 and st[1].reaching and st[1].track_id == 2 and not st[1].occupied, str(st))
st = zt.update([trk(3, 0, 0.6, 0)], zones, 1 / 30)
check("too far to occupy: held from the previous hit for zone_hold_s", st[0].occupied and st[0].track_id == 2, str(st))
for _ in range(12):
    st = zt.update([], zones, 1 / 30)
check("after the hold both zones clear and dwell resets", not st[0].occupied and st[0].dwell_s == 0.0 and st[0].track_id == -1)
for _ in range(30):
    st = zt.update([trk(4, 1, 0.0, 0)], zones, 1 / 30)
check("dwell accumulates while occupied (~1 s)", abs(st[1].dwell_s - 1.0) < 0.05, str(st[1].dwell_s))
st = zt.update([trk(4, 1, 0.0, 0, confirmed=False)], zones, 1 / 30)
check("tentative tracks do not count (hold keeps the state one frame)", st[1].occupied and st[1].track_id == 4)
st = zt.update([trk(5, 0, 0.2, 0, lean=0b01)], zones, 1 / 30)
check("a fresh lean is not reported yet (lean_min_s)", not st[0].leaning and st[0].occupied, str(st[0]))
for _ in range(31):
    st = zt.update([trk(5, 0, 0.2, 0, lean=0b01)], zones, 1 / 30)
check("after a second of leaning the zone reports it", st[0].leaning and not st[0].reaching and st[0].track_id == 5, str(st[0]))
st = zt.update([trk(5, 0, 0.2, 0)], zones, 1 / 30)
check("lean is held one frame after it stops", st[0].leaning)
for _ in range(12):
    st = zt.update([trk(5, 0, 0.2, 0)], zones, 1 / 30)
check("and clears after zone_hold_s", not st[0].leaning and st[0].occupied)

print("pipeline on the synthetic overhead scene (no robot)")
T = S.overhead_camera(4.0)
pipe = Pipeline(cfg)
check("no robot in the config: the pipeline has none", pipe.robot is None)
pipe.set_calib(0, T)
pipe.set_intrinsics(S.intrinsics_for(0))
rng = np.random.default_rng(0)
scene = S.Scene(table=TABLE, boxes=BOXES)
seq, t = 0, 0.0


def run(n, p=None):
    global seq, t
    p = p or pipe
    for _ in range(n):
        dec, dbg = p.step({0: S.packet(0, S.render(scene, T, rng=rng), seq, t)}, t)
        t += 1 / 30
        seq += 1
    return dbg.zones, dbg


pipe.learn_background(frames=30)
run(30)
zs, dbg = run(5)
check("empty table after learning: no tracks, no zones lit", len(dbg.tracks) == 0 and not any(z.occupied or z.reaching for z in zs))

standing = S.Person((-0.5, -0.85))
scene.people = [standing]
zs, dbg = run(20)
check("one confirmed track at the edge of zone a", len(dbg.tracks) == 1 and dbg.tracks[0].confirmed and np.linalg.norm(dbg.tracks[0].xy - [-0.5, -0.85]) < 0.2,
      str([(tr.id, tr.xy.round(2), tr.confirmed) for tr in dbg.tracks]))
check("a occupied, nobody reaching", zs[0].occupied and not zs[0].reaching and zs[0].track_id == dbg.tracks[0].id and not zs[1].occupied and not zs[1].reaching,
      str(zs))

scene.people = [standing.with_arm(90.0, -16.0)]     # arm toward +y over zone a, hand ~1.13 m up: in the hand band
zs, dbg = run(10)
check("arm over a: reach bit set, b untouched", zs[0].reaching and zs[0].occupied and not zs[1].reaching, str(zs))
check("the track reports reach_mask 1 and nearest zone 0", dbg.tracks and dbg.tracks[0].reach_mask == 1 and dbg.tracks[0].nearest_zone == 0,
      str([(tr.reach_mask, tr.nearest_zone) for tr in dbg.tracks]))
rows = pipe.tracks_rows(dbg.tracks)
check("tracks rows keep their 8-column shape", rows.shape == (1, 8))
scene.people = [standing]
zs, dbg = run(1)
check("one frame after the arm drops: reach held", zs[0].reaching)
zs, dbg = run(12)
check("after zone_hold_s the reach clears, occupancy stays", not zs[0].reaching and zs[0].occupied)
d0 = zs[0].dwell_s
zs, dbg = run(30)
check("dwell keeps counting while standing there", zs[0].dwell_s > d0 + 0.9, f"{d0:.2f} -> {zs[0].dwell_s:.2f}")

check("an arm over the zone is not a lean", not zs[0].leaning and dbg.tracks[0].lean_mask == 0)
hp = dbg.tracks[0].hand_pt
check("the hand point is the arm's end over the zone, nearest its centre", hp is not None and -0.75 < hp[1] < -0.35 and abs(hp[0] + 0.5) < 0.12 and hp[2] > 0.97, str(hp))

scene.people = [S.Person((-0.5, -0.85), height=1.1, extra=[(-0.5, -0.55, 1.3, 0.12)])]   # crouched torso, head 35 cm over a's near edge
zs, dbg = run(12)
check("head over the near edge: the track leans at once, the zone not before lean_min_s", not zs[0].leaning and dbg.tracks[0].lean_mask == 1,
      str(zs[0]) + str([(tr.top.round(2), tr.lean_mask) for tr in dbg.tracks]))
zs, dbg = run(30)
check("a second later the zone reports the lean, not a reach", zs[0].leaning and not zs[0].reaching and zs[0].occupied, str(zs[0]))
scene.people = [S.Person((-0.5, -0.85), height=1.7, extra=[(-0.5, -0.55, 1.75, 0.12)])]   # upright, head over the edge but high
zs, dbg = run(12)
check("an upright head over the edge is standing, not leaning", not zs[0].leaning and zs[0].occupied, str(zs[0]) + str([tr.top.round(2) for tr in dbg.tracks]))

scene.people = [S.Person((0.5, 0.85))]              # other side, at b
zs, dbg = run(25)
check("walks to b: b occupied with a new id, a clear with dwell reset", zs[1].occupied and not zs[0].occupied and zs[0].dwell_s == 0.0 and zs[1].track_id != -1, str(zs))

scene.people = [S.Person((0.5, 0.85)).with_arm(-90.0, -16.0), S.Person((-0.5, -0.85)).with_arm(90.0, -16.0)]
zs, dbg = run(25)
check("two people reaching over their own zones", zs[0].reaching and zs[1].reaching and zs[0].track_id != zs[1].track_id and len(dbg.tracks) == 2, str(zs))

print("without zones in the config nothing zone-shaped comes out")
pipe0 = Pipeline(TrackerConfig(table=TABLE, **OVERHEAD))
pipe0.set_calib(0, T)
pipe0.set_intrinsics(S.intrinsics_for(0))
scene.people = []
pipe0.learn_background(frames=30)
run(30, pipe0)
scene.people = [standing.with_arm(90.0, -16.0)]
zs0, dbg0 = run(20, pipe0)
check("a person is still tracked", len(dbg0.tracks) == 1 and dbg0.tracks[0].confirmed, str(dbg0.tracks))
check("zones empty, track carries no zone", zs0 == [] and dbg0.tracks[0].nearest_zone == -1 and dbg0.tracks[0].reach_mask == 0 and dbg0.tracks[0].hand_pt is None)

print(f"test_zones: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
