# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Viewer core without a GL context: the orbit camera maths, screen projection, boxes from labels, the
pipeline's kept-dropped points, track cluster labels, the clip step cache and trails on the synthetic scene."""
import os
import sys

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.pipeline import Pipeline  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel  # noqa: E402
from rt_depth_tracker import calib as C  # noqa: E402
from rt_depth_tracker.viewer import (ViewControl, apply_zone_edits, bg_cloud, box_edges, boxes_from_labels, frustum_lines,  # noqa: E402
                                     project, save_zone_edits, snapshot, sphere_wire, static_lines, track_style, trails,
                                     zone_edits, zone_lines)
from rt_depth_tracker.zones import zones_from_cfg  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


print("view control")
v = ViewControl()
V = v.view_matrix()
e = v.eye()
check("eye maps to the origin", np.allclose(V @ [*e, 1.0], [0, 0, 0, 1], atol=1e-6))
check("target maps to -z at the orbit distance", np.allclose(V @ [*v.target, 1.0], [0, 0, -v.distance, 1], atol=1e-6))
Pm = v.proj_matrix(16 / 9)
s, ok = project(v.target[None], V, Pm, 1600, 900)
check("target projects to the viewport centre", bool(ok[0]) and abs(s[0, 0] - 800) < 1e-6 and abs(s[0, 1] - 450) < 1e-6, str(s))
s2, ok2 = project((e + (e - v.target))[None], V, Pm, 1600, 900)
check("a point behind the eye is culled", not ok2[0])
up = v.target + [0, 0, 0.5]
s3, _ = project(up[None], V, Pm, 1600, 900)
check("world +z is up on screen", s3[0, 1] < 450)
v.zoom(1.0)
check("wheel tick zooms in by 0.3 m", abs(v.distance - 3.7) < 1e-9)
v.rotate(100.0, 0.0)
check("drag rotates 0.3 deg per pixel", abs(v.yaw - 120.0) < 1e-9)
t0 = v.target.copy()
v.pan(10.0, 0.0)
check("pan moves the target horizontally", abs(v.target[2] - t0[2]) < 1e-9 and np.linalg.norm(v.target[:2] - t0[:2]) > 0)

print("wireframes")
check("box has 12 edges", box_edges([0, 0, 0], [1, 2, 3]).shape == (24, 3))
sw = sphere_wire([1, 2, 3], 0.5, 16)
check("sphere wire lies on the sphere", sw.shape == (96, 3) and np.allclose(np.linalg.norm(sw - [1, 2, 3], axis=1), 0.5, atol=1e-9))
cams = S.two_camera_rig()
fr = frustum_lines(S.intrinsics_for(0), cams[0], 2)
check("frustum: 16 segments starting at the camera", fr.shape == (32, 3) and np.allclose(fr[16], cams[0][:3, 3], atol=1e-6))

print("pipeline keep_dropped + boxes + track labels")
cfg = _scene.cfg()
rm = RobotModel.from_config(cfg)
pipe = Pipeline(cfg, robot=rm, keep_dropped=True)
for c, T in cams.items():
    pipe.set_calib(c, T)
    pipe.set_intrinsics(S.intrinsics_for(c))
rng = np.random.default_rng(0)
scene = S.Scene(rail_radius=1.4, q=np.array([0, 0.26, -0.9, 0, 0.6, 0]), robot=rm)
seq, t = 0, 0.0
steps = []


def run(n):
    global seq, t
    for _ in range(n):
        pk = {c: S.packet(c, S.render(scene, cams[c], rng=rng), seq, t) for c in cams}
        pipe.set_q(scene.q, 0.0)
        dec, dbg = pipe.step(pk, t)
        steps.append(snapshot(pipe, dbg, dec, t, scene.q, 0.0))
        t += 1 / 30
        seq += 1
    return dec, dbg


pipe.learn_background(frames=40)
run(40)
steps.clear()
people = [S.Person((2.0, 0.6)), S.Person((2.2, -0.9))]
scene.people = people
dec, dbg = run(30)
st = steps[-1]
ok_counts = all(len(dbg.pix[c][0]) == len(dbg.dropped[c]["crop"]) + len(dbg.dropped[c]["robot"]) + int((dbg.cam_ids == c).sum())
                for c in dbg.pix)
check("fg = cropped + robot-masked + kept, per camera", ok_counts and set(dbg.pix) == {0, 1})
check("pix masks span the foreground", all(len(dbg.pix[c][1]) == len(dbg.pix[c][0]) and len(dbg.pix[c][2]) == int(dbg.pix[c][1].sum()) for c in dbg.pix))
check("robot points removed while the arm is in the background: none", all(len(d["robot"]) == 0 for d in dbg.dropped.values()))
boxes = st.boxes
scored = [cl for cl in st.clusters]
check("two scored clusters, each with a box", len(scored) == 2 and all(cl.label in boxes for cl in scored), f"{len(scored)} {sorted(boxes)}")
ok_box = True
for p in people:
    hits = [b for b in boxes.values() if np.linalg.norm((b[0][:2] + b[1][:2]) / 2 - p.xy) < 0.25 and abs(b[1][2] - 1.7) < 0.12]
    ok_box &= len(hits) == 1
check("box centres on the people, tops at 1.7 m", ok_box, str({k: (np.round(v[0], 2).tolist(), np.round(v[1], 2).tolist()) for k, v in boxes.items()}))
conf = [tr for tr in st.tracks if tr.confirmed]
check("both tracks confirmed after 30 frames", len(conf) == 2, str([(tr.id, tr.hits) for tr in st.tracks]))
ok_lab = all(tr.label in boxes and np.all(boxes[tr.label][0][:2] - 0.1 <= tr.xy) and np.all(tr.xy <= boxes[tr.label][1][:2] + 0.1) for tr in conf)
check("each track's label names the box it stands in", ok_lab, str([(tr.id, tr.label) for tr in conf]))
tr_ = trails(steps, len(steps) - 1, 3.0)
check("trails per confirmed track, bounded by the window", set(tr_) == {tr.id for tr in conf} and all(2 <= len(v) <= 30 for v in tr_.values()),
      str({k: len(v) for k, v in tr_.items()}))
cur = conf[0]
col, text = track_style(cur, 0.5, cur.id, "engage")
check("current track is green and labelled", col == (0.2, 0.9, 0.3) and text.startswith(f"#{cur.id} S=0.50"))
col2, _ = track_style(cur, 0.5, cur.id, "reach")
check("reach hint turns it red", col2 == (0.95, 0.25, 0.2))
tent = [tr for tr in st.tracks if not tr.confirmed]
check("tentative style is grey", all(track_style(tr, 0.0, None, "idle")[0] == (0.6, 0.6, 0.6) for tr in tent) if tent else True)

print("background cloud")
bgc = bg_cloud(pipe.cam(0).bg, S.intrinsics_for(0), cams[0], 2, n_max=5000)
floor = bgc[(bgc[:, 0] > 1.0) & (bgc[:, 0] < 3.0) & (np.abs(bgc[:, 1]) < 1.0) & (bgc[:, 2] < 0.3)]
check("subsampled to n_max", len(bgc) == 5000)
check("floor points sit at z ~ 0", len(floor) > 100 and abs(np.median(floor[:, 2])) < 0.03, f"{len(floor)} med z {np.median(floor[:, 2]) if len(floor) else None}")

print("crop flag")
pipe2 = Pipeline(_scene.cfg(crop=False), robot=rm, keep_dropped=True)
for c, T in cams.items():
    pipe2.set_calib(c, T)
    pipe2.set_intrinsics(S.intrinsics_for(c))
pipe2.learn_background(frames=10)
scene.people = []
for i in range(10):
    pipe2.set_q(scene.q, 0.0)
    pipe2.step({c: S.packet(c, S.render(scene, cams[c], rng=rng), 100 + i, 10 + i / 30) for c in cams}, 10 + i / 30)
scene.people = people
pipe2.set_q(scene.q, 0.0)
_, dbg2 = pipe2.step({c: S.packet(c, S.render(scene, cams[c], rng=rng), 200, 12.0) for c in cams}, 12.0)
check("crop off drops nothing", all(len(d["crop"]) == 0 for d in dbg2.dropped.values()) and len(dbg2.points) > 0)

print("static geometry follows the config: nothing drawn for what is not declared")
_plain = static_lines(TrackerConfig())
_full = static_lines(_scene.cfg(wall_x=-1.0, pedestal_radius=0.25, mount_z=1.0, marks={"M": [1.6, 0.0]}))
check("plain scene: the floor grid only; wall, rail, pedestal and marks add lines", len(_full) > len(_plain) and _plain[:, 2].max() == 0.0)
zx0, zc0 = zone_lines(TrackerConfig())
check("no zones, no table -> no zone lines", zx0.shape == (0, 3))
_zc = TrackerConfig(table={"center_xy": [0.0, 0.0], "size_xy": [2.0, 1.2], "top_z": 0.9},
                    zones=[{"name": "a", "rect": [-0.8, -0.4, -0.2, 0.4], "top_z": 0.95}, {"name": "b", "rect": [0.2, -0.4, 0.8, 0.4], "top_z": 0.95}])
zx, zc = zone_lines(_zc)
check("table box + 2 zone boxes", zx.shape == (72, 3) and zc.shape == (72, 3) and abs(zx[:, 2].max() - (0.95 + _zc.zone_margin_z)) < 1e-6 and abs(zx[:24, 0]).max() == 1.0, str(zx.shape))

print("zone edits: sliders apply live and save to the config file, keeping its other keys")
_ed = zone_edits(_zc)
check("edits mirror the config", len(_ed) == 2 and abs(_ed[0]["cx"] + 0.5) < 1e-9 and abs(_ed[0]["w"] - 0.6) < 1e-9 and _ed[0]["top"] == 0.95)
_ed[0]["cx"] += 0.5
_ed[0]["w"] = 0.3
apply_zone_edits(_zc, _ed)
_z0 = zones_from_cfg(_zc)[0]
check("apply: the config zone moved and resized", abs(_z0.lo[0] - (-0.15)) < 1e-6 and abs(_z0.hi[0] - 0.15) < 1e-6 and len(zones_from_cfg(_zc)) == 2, str((_z0.lo, _z0.hi)))
import json as _json  # noqa: E402
import tempfile as _tempfile  # noqa: E402
_tp = os.path.join(_tempfile.mkdtemp(prefix="rt_viewer_"), "tracker.json")
_json.dump({"keep": 1, "zones": [], "lean_bias_m": 0.0}, open(_tp, "w"))
_zc.lean_bias_m = 0.2
_msg = save_zone_edits(_tp, _zc, _ed)
_t2 = _json.load(open(_tp))
check("save: the file keeps its other keys and gets the zones + band knobs", _t2["keep"] == 1 and abs(_t2["zones"][0]["rect"][0] + 0.15) < 1e-6 and _t2["zones"][1]["name"] == "b"
      and _t2["lean_bias_m"] == 0.2 and "saved 2 zones" in _msg, str(_t2))
_T0 = cams[0].copy()
_Tn = C.yaw_T(90.0, 0.1, 0.0) @ _T0
check("the calibration nudge is a world yaw about the origin plus a shift", np.allclose(_Tn[:3, 3], [-_T0[1, 3] + 0.1, _T0[0, 3], _T0[2, 3]], atol=1e-9), str(_Tn[:3, 3]))

print("tracker mirror: TargetRx + snapshot carries what was sent")
import socket as _socket
import time as _time
from rt_depth_tracker import protocol as _P
from rt_depth_tracker.run import TargetRx
from rt_depth_tracker.viewer import cross_lines, diamond_lines, snapshot as _snapshot
_port = 7616
trx = TargetRx(_port)
_tx = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
_tx.sendto(_P.pack_target(_P.Target(1.5, 0.3, 1.6, "engage", 7, 0.8)), ("127.0.0.1", _port))
_rows = np.array([[7, 1.5, 0.3, 1.6, 1.7, 0.8, 0.1, 2], [8, 2.0, -1.0, 1.5, 1.6, 0.2, 0.0, 0]], np.float32)
_tx.sendto(_P.pack_tracks(_rows), ("127.0.0.1", _port))
_t0 = _time.monotonic()
while _time.monotonic() - _t0 < 2.0 and (trx.get()[0] is None or trx.get()[2] is None):
    _time.sleep(0.02)
_tgt, _age, _got, _rage = trx.get()
check("target received with age", _tgt is not None and _tgt.hint == "engage" and _tgt.track_id == 7 and _age is not None and _age < 1.0)
check("tracks rows received (n, 8)", _got is not None and _got.shape == (2, 8) and int(_got[0, 0]) == 7)
_tx.sendto(_P.pack_tracks(np.zeros((0, 8), np.float32)), ("127.0.0.1", _port))
_tx.sendto(_P.pack_zone(_P.ZoneMsg(0, "a", 1, 0, 7, 1.5)), ("127.0.0.1", _port))
_time.sleep(0.2)
check("empty tracks blob decodes to (0, 8)", trx.get()[2] is not None and trx.get()[2].shape == (0, 8))
_zones, _zage = trx.get_zones()
check("a zone message lands by index with its age", 0 in _zones and _zones[0].name == "a" and _zones[0].occupied == 1 and _zage is not None and _zage < 1.0, str(_zones))
trx.stop()
st2 = _snapshot(pipe, dbg, dec, t, scene.q, 0.0, sent=_tgt, sent_age=_age, sent_tracks=_rows, sent_zones=_zones)
check("snapshot carries sent target, rows and zones; local zones empty without a zone config", st2.sent is _tgt and st2.sent_tracks is _rows and st2.sent_age == _age
      and st2.sent_zones is _zones and st2.zones == [])
check("cross / diamond geometry", cross_lines([0, 0, 1], 0.1).shape == (6, 3) and cross_lines([0, 0, 1], 0.1, [0, 0, 0]).shape == (8, 3)
      and diamond_lines([1, 1, 1], 0.05).shape == (8, 3))

print(f"test_viewer: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
