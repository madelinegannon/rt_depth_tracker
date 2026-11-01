# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Field calibration steps on synthetic scenes: the floor plane fixes roll/pitch/height and a person standing on a
measured mark fixes yaw, both pivoting on the camera position, so a perturbed seed comes back to the true pose;
an overhead camera over a rotated, shifted table comes back to a world with the table centred and aligned (no
robot anywhere); the calib tool's file verbs (seed, plane, table, nudge, undo, validate) through its CLI."""
import argparse  # noqa: F401
import contextlib
import io
import json
import os
import socket
import sys
import tempfile

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import calib as C  # noqa: E402
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.background import BackgroundModel  # noqa: E402
from rt_depth_tracker.calib_tool import main as calib_main  # noqa: E402
from rt_depth_tracker.intrinsics import ray_table, unproject  # noqa: E402
from rt_depth_tracker.record import Recorder  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel  # noqa: E402
from rt_depth_tracker.segment import crop_world  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


def rot_axis(axis, deg):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rot_err_deg(Ta, Tb):
    R = Ta[:3, :3].T @ Tb[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


cfg = _scene.cfg()
rm = RobotModel.from_config(cfg)
rays = ray_table(S.FEMTO_NFOV_BINNED, 2)
rng = np.random.default_rng(3)
MARK = np.array([1.6, 0.0])

print("stable_runs")
cents = [None] * 3 + [(2.0, 0.0)] * 40 + [(2.05, 0.02)] * 40 + [None] + [(0.5, 1.0)] * 10 + [(1.2, 0.0)] * 90
times = [k / 30 for k in range(len(cents))]
runs = C.stable_runs(cents, times, min_s=2.0, jump=0.15)
check("two long runs found, short one dropped", len(runs) == 2 and runs[0][0] == 3 and runs[1][1] == len(cents) - 1, str(runs))
check("run means", np.allclose(runs[0][2], (2.025, 0.01), atol=0.01) and np.allclose(runs[1][2], (1.2, 0.0)), str([r[2] for r in runs]))

for cid, T_true in S.two_camera_rig().items():
    print(f"cam {cid}: perturbed seed -> plane -> yaw")
    cpos = T_true[:3, 3].copy()
    pitch, yaw = (12.0, 15.0) if cid == 0 else (-9.0, -20.0)
    T_bad = C.rotate_about(T_true, rot_axis(T_true[:3, 0], pitch), cpos)      # about the sensor's horizontal x axis
    T_bad = C.rotate_about(T_bad, rot_axis([0, 0, 1], yaw), cpos)              # about the vertical through the camera
    check("perturbation is rotations only", np.allclose(T_bad[:3, 3], cpos) and rot_err_deg(T_bad, T_true) > 10)

    empty = S.Scene(rail_radius=1.4, q=np.array([0, 0.26, -0.9, 0, 0.6, 0]), robot=rm)
    bg = BackgroundModel(cfg.bg_tol_mm, cfg.bg_tol_sig, cfg.bg_tol_frac, cfg.bg_valid_frac)
    bg.learn(np.stack([S.render(empty, T_true, rng=rng) for _ in range(25)]))

    P_s, _ = unproject(bg.bg.astype(np.float32), rays)
    T1, rep = C.floor_correction(C.apply(T_bad, P_s), T_bad)
    check("floor fit sees the pitch", T1 is not None and abs(rep["tilt_before_deg"] - abs(pitch)) < 1.0, str(rep))
    tilt, height = C.floor_residual(C.apply(T1, P_s))
    check("floor flat and at z=0 after plane", tilt < 0.5 and abs(height) < 0.02, f"tilt {tilt:.2f} deg height {height:.3f} m")
    check("camera xy untouched, z lifted by the floor residual only (<2 cm)", np.allclose(T1[:2, 3], cpos[:2]) and abs(T1[2, 3] - cpos[2]) < 0.02, str(T1[:3, 3] - cpos))

    scene = S.Scene(rail_radius=1.4, q=empty.q.copy(), robot=rm, people=[S.Person(tuple(MARK), height=1.7)])
    cents, times = [], []
    for k in range(20):
        d = S.render(scene, T_true, rng=rng)
        P, _ = unproject(d, rays, bg.foreground(d))
        Pw = C.apply(T1, P)
        keep = crop_world(Pw, cfg) & (np.hypot(Pw[:, 0], Pw[:, 1]) > 0.6) & (Pw[:, 2] > 0.25) & (Pw[:, 2] < 2.0)
        Q = Pw[keep]
        cents.append((Q[:, 0].mean(), Q[:, 1].mean(), np.percentile(Q[:, 2], 97)) if keep.sum() > 150 else None)
        times.append(k / 30)
    runs = C.stable_runs(cents, times, min_s=0.5)
    check("person forms one still run", len(runs) == 1, str(runs))
    T2, r = C.yaw_correction(runs[0][2], T1, MARK)
    check("range agrees with the mark (<5 %)", abs(r["range_ratio"] - 1) < 0.05, str(r))
    check("yaw recovered within 1 deg", abs(r["yaw_deg"] + yaw) < 1.0, f"{r['yaw_deg']} vs {-yaw}")
    seen = C.apply(T2, C.apply(C.invert(T1), np.array([[*runs[0][2], 1.0]])))[0]
    check("person lands on the mark (<0.2 m, surface bias)", np.linalg.norm(seen[:2] - MARK) < 0.2, str(seen.round(2)))
    check("full pose recovered (<1 deg, <2 cm)", rot_err_deg(T2, T_true) < 1.0 and np.linalg.norm(T2[:3, 3] - cpos) < 0.02, f"{rot_err_deg(T2, T_true):.2f} deg, {np.linalg.norm(T2[:3, 3] - cpos) * 100:.1f} cm")
    _, bad = C.yaw_correction(runs[0][2], T1, MARK + [0.6, 0.0])
    check("wrong mark shows up as a range mismatch", abs(bad["range_ratio"] - 1) > 0.1, str(bad))

print("overhead camera, no robot: perturbed seed -> plane -> table")
TABLE = {"center_xy": [0.0, 0.0], "size_xy": [2.0, 1.2], "top_z": 0.9}
BOXES = [([-0.8, -0.4, 0.9], [-0.2, 0.4, 0.95]), ([0.2, -0.4, 0.9], [0.8, 0.4, 0.95])]
YAW, OFF = 17.0, (0.3, -0.2)
scene = S.Scene(table=TABLE, boxes=BOXES, yaw_deg=YAW, offset_xy=OFF)
T_cam = S.overhead_camera(4.0)                                   # the real camera: above the world origin, which is NOT the table centre
cpos = T_cam[:3, 3].copy()
T_bad = C.rotate_about(T_cam, rot_axis(T_cam[:3, 0], 8.0), cpos)          # seed pitched 8 deg about the sensor x axis
T_bad = C.rotate_about(T_bad, rot_axis(T_cam[:3, 1], -5.0), cpos)         # and rolled 5 deg about the sensor y axis
T_bad[2, 3] += 0.25                                                        # and 25 cm too high
bg = BackgroundModel(cfg.bg_tol_mm, cfg.bg_tol_sig, cfg.bg_tol_frac, cfg.bg_valid_frac)
bg.learn(np.stack([S.render(scene, T_cam, rng=rng) for _ in range(20)]))
P_s, _ = unproject(bg.bg.astype(np.float32), rays)
check("straight-down look-at is a valid frame (image down = world -x, so image up = +x)", np.allclose(T_cam[:3, 2], [0, 0, -1]) and np.allclose(T_cam[:3, 1], [-1, 0, 0]), str(T_cam[:3, :3]))

T1, rep = C.floor_correction(C.apply(T_bad, P_s), T_bad)
check("floor fit sees the tilt (~9.4 deg)", T1 is not None and abs(rep["tilt_before_deg"] - 9.4) < 1.5, str(rep))
tilt, height = C.floor_residual(C.apply(T1, P_s))
check("floor flat and at z=0 after plane", tilt < 0.5 and abs(height) < 0.02, f"tilt {tilt:.2f} deg height {height:.3f} m")
check("camera height recovered (4.0 m)", abs(T1[2, 3] - 4.0) < 0.03, str(T1[:3, 3]))

P1 = C.apply(T1, P_s)
top = P1[np.abs(P1[:, 2] - 0.9) < 0.04]
check("table top visible at z = 0.9 under the plane-corrected pose", len(top) > 500, str(len(top)))
T2, rep2 = C.table_correction(P1, T1, TABLE["top_z"], TABLE["size_xy"])
check("table step applied", T2 is not None and rep2["applied"], str(rep2))
check("measured size within 10 %", all(abs(r - 1) < 0.10 for r in rep2["size_ratio"]), str(rep2["size_ratio"]))
P2 = C.apply(T2, P_s)
top2 = P2[np.abs(P2[:, 2] - 0.9) < 0.04]
yaw2, centre2, _ = C.rect_fit_xy(top2[:, :2])
check("table top centred on the origin (<3 cm) and aligned (<1 deg)", np.linalg.norm(centre2) < 0.03 and abs(yaw2) < 1.0, f"centre {centre2.round(3)} yaw {yaw2:.2f}")
check("applied yaw is -YAW", abs(rep2["yaw_applied_deg"] + YAW) < 1.0, str(rep2["yaw_applied_deg"]))
Rz = C.euler_xyz_deg_to_R(0, 0, -YAW)
cam_expect = Rz @ (cpos - np.array([OFF[0], OFF[1], 0.0]))
check("camera lands where the table frame has it (<3 cm)", np.linalg.norm(T2[:3, 3] - cam_expect) < 0.03, f"{T2[:3, 3].round(3)} vs {cam_expect.round(3)}")
m_pts = P2[(P2[:, 2] > 0.93) & (P2[:, 2] < 0.97)]
in_a = ((m_pts[:, 0] > -0.8) & (m_pts[:, 0] < -0.2) & (np.abs(m_pts[:, 1]) < 0.4)).sum()
in_b = ((m_pts[:, 0] > 0.2) & (m_pts[:, 0] < 0.8) & (np.abs(m_pts[:, 1]) < 0.4)).sum()
check("box tops fall inside their rectangles", len(m_pts) > 50 and (in_a + in_b) / len(m_pts) > 0.9, f"{in_a + in_b}/{len(m_pts)}")
check("180 deg ambiguity is the only freedom left: a 180 nudge keeps the fit", abs(C.rect_fit_xy(C.apply(C.yaw_T(180.0), top2)[:, :2])[0]) < 1.0)

print("plane refuses without floor points; table refuses the wrong size")
T_none, r_none = C.floor_correction(P1[P1[:, 2] > 0.5], T1)
check("no floor -> None", T_none is None and r_none["floor_candidates"] < 200, str(r_none))
T_wrong, r_wrong = C.table_correction(P1, T1, TABLE["top_z"], [2.0, 2.0])
check("square config vs the 2.0 x 1.2 table -> refused", T_wrong is None and "size" in r_wrong["why"], str(r_wrong))

print("calib_tool CLI on files: seed -> plane -> table solves a synthetic overhead clip without a robot")
tmp = tempfile.mkdtemp(prefix="rt_calib_")
cfg_path = os.path.join(tmp, "tracker.json")
calib_dir = os.path.join(tmp, "calib")
# the seed aims exactly straight down (pos and aim share x, y): a near-vertical look-at would let a tiny aim offset pick
# the image's up direction, and the table step can only resolve yaw within +-90 deg of the seed
json.dump({"table": TABLE, "cams": {"0": {"pos": [0.1, -0.1, 4.2], "aim": [0.1, -0.1, 0.0]}}}, open(cfg_path, "w"))
rec = Recorder(os.path.join(tmp, "rec"), chunk_s=1e9, tag="overhead")
rec.set_meta(intr={0: S.intrinsics_for(0)})
for k in range(25):
    rec.add(S.packet(0, S.render(scene, T_cam, rng=rng), k, 100.0 + k / 30), np.zeros(6), 100.0 + k / 30)
rec.close()
clip = rec.files[0]
common = ["--config", cfg_path, "--calib-dir", calib_dir, "--ctrl-port", str(free_port()), "--in", str(free_port()), "--q-port", str(free_port())]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_seed = calib_main(["seed", "--cam", "0", *common])
    rc_plane = calib_main(["plane", "--cams", "0", "--clip", clip, *common])
    rc_table = calib_main(["table", "--cams", "0", "--clip", clip, *common])
out = buf.getvalue()
T_sol, meta = C.load_sensor_calib(os.path.join(calib_dir, "sensor_a.json"))
check("seed, plane and table all ran", rc_seed == 0 and rc_plane == 0 and rc_table == 0, out[-600:])
check("method chain recorded", meta["meta"]["method"] == "seed+plane+table", str(meta["meta"].get("method")))
check("the solved camera lands where the table frame has it (<5 cm, <1 deg)",
      np.linalg.norm(T_sol[:3, 3] - cam_expect) < 0.05 and rot_err_deg(T_sol, C.yaw_T(-YAW) @ C.make_T(T_cam[:3, :3], T_cam[:3, 3] - [OFF[0], OFF[1], 0.0])) < 1.0,
      f"{T_sol[:3, 3].round(3)} vs {cam_expect.round(3)}")
check("a .prev.json backup exists for undo", os.path.exists(os.path.join(calib_dir, "sensor_a.prev.json")))

print("nudge applies on top, undo restores")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_nudge = calib_main(["nudge", "--cams", "0", "--yaw", "10", "--dx", "0.1", *common])
    T_nudged, meta_n = C.load_sensor_calib(os.path.join(calib_dir, "sensor_a.json"))
    rc_undo = calib_main(["undo", "--cams", "0", *common])
T_back, meta_b = C.load_sensor_calib(os.path.join(calib_dir, "sensor_a.json"))
check("nudge = yaw_T(10, 0.1, 0) @ T", rc_nudge == 0 and meta_n["meta"]["method"] == "seed+plane+table+nudge" and np.allclose(T_nudged, C.yaw_T(10.0, 0.1, 0.0) @ T_sol, atol=1e-6))
check("undo puts the table solution back", rc_undo == 0 and np.allclose(T_back, T_sol) and meta_b["meta"]["method"] == "seed+plane+table")

print("validate refuses without a /robot/q source, and without a robot")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = calib_main(["validate", *common])
out = buf.getvalue()
check("no robot -> leftover skipped, CHECK", rc == 1 and "no robot in the config" in out and "RESULT: CHECK" in out, out[-300:])
cfg_r = os.path.join(tmp, "tracker_robot.json")
json.dump({"robot": _scene.ROBOT}, open(cfg_r, "w"))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = calib_main(["validate", "--config", cfg_r, "--calib-dir", calib_dir, "--ctrl-port", str(free_port()), "--in", str(free_port()), "--q-port", str(free_port())])
out = buf.getvalue()
check("robot but no joint feed -> leftover skipped, CHECK", rc == 1 and "no /robot/q" in out and "RESULT: CHECK" in out, out[-300:])

print("== the calib tool's .prev.json backups are never loaded as the calibration ==")
_d = tempfile.mkdtemp(prefix="rt_prev_")
T_live = C.make_T(np.eye(3), np.array([0.0, 1.5, 3.0]))
T_prev = C.make_T(np.eye(3), np.array([0.0, 1.5, 2.5]))         # the camera before it was moved
C.save_sensor_calib(os.path.join(_d, "sensor_b.json"), 1, T_live, {"method": "sphere"})
C.save_sensor_calib(os.path.join(_d, "sensor_b.prev.json"), 1, T_prev, {"method": "sphere"})
C.save_sensor_calib(os.path.join(_d, "sensor_a.json"), 0, T_live, {"method": "sphere"})
_files = [os.path.basename(f) for f in C.sensor_calib_files(_d)]
check("only the live files, in order", _files == ["sensor_a.json", "sensor_b.json"], str(_files))
_loaded = {}
for _f in C.sensor_calib_files(_d):
    _T, _meta = C.load_sensor_calib(_f)
    _loaded[int(_meta["cam_id"])] = _T
check("cam 1 is the live calibration, not the backup that sorts after it", np.allclose(_loaded[1], T_live), str(_loaded[1][:3, 3]))

print(f"test_calib_steps: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
