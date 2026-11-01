# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Aiming and the swapped-camera check: aim_search finds pan/tilt pairs that put the coverage zone in both
cameras, pan_tilt_of round-trips, and a camera configured on the wrong side of the robot is caught by the
range check of the yaw step on a lateral mark (an on-axis mark proves nothing, by symmetry)."""
import os
import sys

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import calib as C  # noqa: E402
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.background import BackgroundModel  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.intrinsics import ray_table, unproject  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


intr = S.FEMTO_NFOV_BINNED
cfg = _scene.cfg()
rm = RobotModel.from_config(cfg)
rays = ray_table(intr, 2)
rng = np.random.default_rng(5)

print("pan/tilt round trip")
T = C.look_at_pan_tilt([-0.915, 1.55, 3.0], -20.0, 40.0)
pan, tilt, fl = C.pan_tilt_of(T)
check("pan/tilt recovered", abs(pan + 20) < 1e-6 and abs(tilt - 40) < 1e-6, f"{pan} {tilt}")
check("floor point on the axis line, in front", fl is not None and abs(fl[2]) < 1e-9 and fl[0] > 1.0, str(fl))

print("aim search, cameras 3.0 m up")
pos = {0: np.array([-0.915, 1.55, 3.0]), 1: np.array([-0.915, -1.55, 3.0])}
res, both = C.aim_search(pos, intr, 2)
check("both cameras cover >= 90 % of the zone", both >= 0.9, f"{both:.3f}")
check("symmetric solution", abs(res[0]["pan_deg"] + res[1]["pan_deg"]) <= 2 and abs(res[0]["tilt_deg"] - res[1]["tilt_deg"]) <= 2,
      f"{res[0]['pan_deg']}/{res[1]['pan_deg']} {res[0]['tilt_deg']}/{res[1]['tilt_deg']}")
check("each camera points toward the axis", res[0]["pan_deg"] < 0 < res[1]["pan_deg"], f"{res[0]['pan_deg']} {res[1]['pan_deg']}")
check("floor points in front of the robot", all(r["floor_point"] is not None and r["floor_point"][0] > 0.5 for r in res.values()),
      str([r["floor_point"] for r in res.values()]))
print(f"     cam 0 pan {res[0]['pan_deg']} tilt {res[0]['tilt_deg']} F {res[0]['floor_point']} | cam 1 pan {res[1]['pan_deg']} tilt {res[1]['tilt_deg']} F {res[1]['floor_point']} | both {both:.3f}")

print("aim search, cameras 2.5 m up")
pos25 = {0: np.array([-0.915, 1.55, 2.5]), 1: np.array([-0.915, -1.55, 2.5])}
res25, both25 = C.aim_search(pos25, intr, 2)
check("2.5 m: both cameras cover >= 80 % of the zone", both25 >= 0.8, f"{both25:.3f}")

print("swapped camera: invisible on the axis mark, caught by the range check on a lateral mark")
T_true = C.look_at_T([-0.915, 1.55, 3.0], [1.5, 0.0, 0.9])      # cam 0 really on the visitor right (+y)
T_cfg = C.look_at_T([-0.915, -1.55, 3.0], [1.5, 0.0, 0.9])      # config believes it is on the left
for name, mark, must_catch in (("A1 (on axis)", (1.6, 0.0), False), ("Rt (lateral)", (1.6, 1.0), True), ("L (lateral)", (1.6, -1.0), True)):
    seen_true = np.array(mark) + 0.0                              # what the real camera measures (ideal centroid)
    # express that observation through the mirrored config: same range from the (wrong) pivot as from the true one
    r_true = np.hypot(*(np.array(mark) - T_true[:3, 3][:2]))
    b_cfg = np.arctan2(mark[1] - T_cfg[1, 3], mark[0] - T_cfg[0, 3])
    seen_cfg = T_cfg[:3, 3][:2] + r_true * np.array([np.cos(b_cfg), np.sin(b_cfg)])
    _, rep = C.yaw_correction(seen_cfg, T_cfg, mark)
    caught = abs(rep["range_ratio"] - 1) > 0.10
    check(f"{name}: range ratio {rep['range_ratio']} -> {'caught' if caught else 'not caught'}", caught == must_catch, str(rep))

print(f"test_aim_side: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
