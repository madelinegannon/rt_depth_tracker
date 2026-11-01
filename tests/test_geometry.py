# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""intrinsics (ray table / distortion round-trip), robot model (FK + masks), calib math, background model."""
import os
import sys
import tempfile

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import calib as C  # noqa: E402
from rt_depth_tracker.background import BackgroundModel  # noqa: E402
from rt_depth_tracker.intrinsics import distort, project, ray_table, undistort, unproject  # noqa: E402
from rt_depth_tracker.protocol import Intrinsics  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel, rpy_to_R  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


rng = np.random.default_rng(0)

print("intrinsics")
k = Intrinsics(0, 320, 288, 251.5, 251.5, 160.0, 144.0, k1=0.08, k2=-0.05, k3=0.01, p1=0.0005, p2=-0.0003)
x, y = rng.uniform(-0.7, 0.7, 500), rng.uniform(-0.6, 0.6, 500)
xd, yd = distort(x, y, k)
xu, yu = undistort(xd, yd, k)
check("undistort inverts distort", np.abs(xu - x).max() < 1e-6 and np.abs(yu - y).max() < 1e-6, f"{np.abs(xu-x).max():.2e}")
rays = ray_table(k, 2)
check("ray table shape", rays.shape == (144, 160, 3) and np.allclose(rays[..., 2], 1))
depth = np.full((144, 160), 3000, np.uint16)
P, idx = unproject(depth, rays)
u, v = project(P, k, 2)
U, V = np.meshgrid(np.arange(160), np.arange(144))
check("project(unproject) returns pixel centres", np.abs(u - U.ravel()).max() < 1e-3 and np.abs(v - V.ravel()).max() < 1e-3,
      f"{np.abs(u-U.ravel()).max():.2e}")
check("depth 3 m -> z 3 m", np.allclose(P[:, 2], 3.0))
depth[10:20, :] = 0
P2, idx2 = unproject(depth, rays)
check("zero depth dropped", len(P2) == len(P) - 10 * 160)
k0 = Intrinsics(0, 320, 288, 251.5, 251.5, 160.0, 144.0)
check("no distortion -> identity", np.array_equal(*undistort(x, y, k0)[:1], x))

print("robot model")
rm = RobotModel.from_config(_scene.cfg())
bz = rm.T_world_base[2, 3]
check("6 joints, 7 links", len(rm.joints) == 6 and len(rm.links) == 7, str(rm.links))
check("rpy identity", np.allclose(rpy_to_R(0, 0, 0), np.eye(3)))
check("rpy x 90", np.allclose(rpy_to_R(np.pi / 2, 0, 0) @ [0, 1, 0], [0, 0, 1]))
ee0 = rm.ee_position(np.zeros(6))
# UF850 zero pose: upper arm up (0.364+0.39), forearm folded down (0.426+0.09) -> EE ~0.24 m above base, 0.15 m out
check("q=0: folded pose, EE ~0.24 m above base", abs(ee0[2] - (bz + 0.364 + 0.39 - 0.426 - 0.09)) < 0.02 and abs(ee0[0] - 0.15) < 0.02,
      str(ee0.round(3)))
qd = np.array([0, 0.26, -0.9, 0, 0.6, 0])
eed = rm.ee_position(qd)
check("default_q EE forward + above mount", eed[2] > bz + 0.39 and 0.2 < eed[0] < 0.8 and abs(eed[1]) < 0.01, str(eed.round(3)))
e1 = rm.ee_position(np.array([np.pi / 2, 0.26, -0.9, 0, 0.6, 0]))
check("joint1 rotates EE about z", abs(np.hypot(e1[0], e1[1]) - np.hypot(eed[0], eed[1])) < 1e-6 and abs(e1[2] - eed[2]) < 1e-6
      and abs(e1[1] - eed[0]) < 1e-6, str(e1.round(3)))
ey = _scene.robot(base_yaw_deg=90.0).ee_position(qd)
check("base yaw rotates FK about world z", abs(ey[0] + eed[1]) < 1e-6 and abs(ey[1] - eed[0]) < 1e-6 and abs(ey[2] - eed[2]) < 1e-6, str(ey.round(3)))
cs, rs = rm.sphere_world(qd)
check("spheres above the floor and within reach", cs[:, 2].min() > bz - 0.01 and np.hypot(cs[:, 0], cs[:, 1]).max() < 0.95)
surf = rm.surface_samples(qd, 30)
m = rm.inside_mask(surf, qd, margin=0.08, pedestal_radius=0.0)
check("all sphere-surface points masked at 0.08", m.all(), f"{(~m).sum()} escaped")
far = cs + np.array([[0.0, 0.4, 0.0]])          # sideways: along x a folded arm puts shifted centres inside its other links
m2 = rm.inside_mask(far, qd, margin=0.08, pedestal_radius=0.0)
check("points 0.4 m to the side of the arm survive", not m2.any(), f"{m2.sum()} masked")
rm_ped = _scene.robot(mount_z=1.0)                                   # a base on a 1 m column
ped = np.array([[0.1, 0.1, 0.5], [0.1, 0.1, 1.1], [0.5, 0.0, 0.5]])
mp = rm_ped.inside_mask(ped, qd, margin=0.0, pedestal_radius=0.25)
check("pedestal cylinder mask (inside the column, above it, beside it)", mp.tolist() == [True, False, False], str(mp))
check("no pedestal by default: a point under a floor-standing base is not masked", not rm.inside_mask(np.array([[0.1, 0.1, 0.02]]), qd, margin=0.0)[0])
rm_bare = _scene.robot(mount_z=_scene.cfg().mount_z)
check("the YAML is the whole arm: the config adds no spheres", rm.n_spheres == rm_bare.n_spheres and not _scene.cfg().robot_spheres_extra,
      f"{rm.n_spheres} vs {rm_bare.n_spheres}")
T3 = rm.link_transforms(qd)["link3"]
elbow = T3 @ np.array([0.15, 0.0, 0.0, 1.0])
check("elbow corner (0.15, 0, 0) is inside the mask from the YAML alone", rm_bare.inside_mask(elbow[None, :3], qd, 0.0, 0.0)[0])
diag = T3 @ np.array([0.075, 0.213, 0.0, 1.0])
check("the elbow->wrist diagonal is air: no sphere on its midpoint", not rm_bare.inside_mask(diag[None, :3], qd, 0.0, 0.0)[0])
rm2 = RobotModel.from_config(_scene.cfg(robot_sphere_scale=2.0))
check("sphere scale doubles the radii", np.allclose(rm2.sphere_world(qd)[1], 2 * rm_bare.sphere_world(qd)[1]))
ball = {"link6": [{"center": [0, 0, 0.18], "radius": 0.075}]}
rm3 = RobotModel.from_config(_scene.cfg(robot_spheres_extra=ball))
check("an extra rides on top of the YAML", rm3.n_spheres == rm_bare.n_spheres + 1, f"{rm3.n_spheres} vs {rm_bare.n_spheres}")
c3 = _scene.cfg()
c3.apply({"robot_spheres_extra": {"link3": [{"center": [0, 0, 0.1], "radius": 0.05}]}})
c3.apply({"robot_spheres_extra": ball})
check("overlay adds a link's spheres without dropping the others", set(c3.robot_spheres_extra) == {"link3", "link6"})

print("calib math")
R_true = C.euler_xyz_deg_to_R(12, -35, 140)
t_true = np.array([1.2, -0.7, 3.0])
A = rng.uniform(-0.5, 0.5, (16, 3)) + [0, 1.5, 1.2]
B = A @ R_true.T + t_true + rng.normal(0, 0.010, A.shape)
T, rms = C.kabsch(A, B)
dR = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ R_true) - 1) / 2, -1, 1)))
# 16 pts, sigma 10 mm, 0.5 m spread: expect ~0.5 deg / ~15 mm (rotation error x 1.9 m centroid offset)
check("kabsch recovers T from 16 noisy pts", np.linalg.norm(T[:3, 3] - t_true) < 0.025 and dR < 1.0, f"dt={np.linalg.norm(T[:3,3]-t_true):.4f} dR={dR:.3f}deg rms={rms:.4f}")
check("invert", np.allclose(C.invert(T) @ T, np.eye(4)))
c_true = np.array([0.3, 1.5, 1.4])
dirs = rng.normal(size=(400, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
dirs = dirs[dirs[:, 1] < -0.1]          # front hemisphere only (camera sees one side)
Ps = c_true + 0.075 * dirs + rng.normal(0, 0.004, (len(dirs), 3))
c_fit, rms, n_in = C.fit_sphere_fixed_radius(Ps, 0.075, c0=c_true + [0.05, -0.05, 0.03])
check("sphere fit from front hemisphere", np.linalg.norm(c_fit - c_true) < 0.005, f"err={np.linalg.norm(c_fit-c_true)*1000:.1f} mm inl={n_in}")
n_true = np.array([0.02, -0.01, 1.0])
n_true /= np.linalg.norm(n_true)
plane = rng.uniform(-2, 2, (600, 3))
plane[:, 2] = (-n_true[0] * plane[:, 0] - n_true[1] * plane[:, 1]) / n_true[2] + rng.normal(0, 0.005, 600)
outl = rng.uniform(-2, 2, (250, 3)) + [0, 0, 1.0]
n_fit, d_fit, inl = C.fit_plane_ransac(np.vstack([plane, outl]), thresh=0.02, rng=rng)
check("plane ransac with 30% outliers", np.degrees(np.arccos(abs(n_fit @ n_true))) < 0.2 and inl[:600].mean() > 0.95 and inl[600:].mean() < 0.15,
      f"ang={np.degrees(np.arccos(abs(n_fit@n_true))):.3f} inl={inl[:600].mean():.2f}/{inl[600:].mean():.2f}")
Ra = C.R_align(np.array([0, 0.1, 1.0]), np.array([0, 0, 1.0]))
check("R_align", np.allclose(Ra @ (np.array([0, 0.1, 1.0]) / np.linalg.norm([0, 0.1, 1.0])), [0, 0, 1], atol=1e-9))
Tl = C.look_at_T([0, -1, 3], [0, 1.6, 0.9])
check("look_at: z axis toward target, x horizontal", np.allclose(Tl[:3, 2], (np.array([0, 2.6, -2.1]) / np.linalg.norm([0, 2.6, -2.1]))) and abs(Tl[2, 0]) < 1e-9)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "sensor_a.json")
    sha = C.save_sensor_calib(p, 0, T, {"method": "test"})
    T2, meta = C.load_sensor_calib(p)
    check("calib file roundtrip", np.allclose(T2, T) and meta["sha"] == sha and meta["cam_id"] == 0)

print("background")
H, W = 144, 160
base = np.full((H, W), 3000.0)
base[:60, :] = 4000.0
frames = np.stack([base + rng.normal(0, 5, (H, W)) for _ in range(60)]).astype(np.uint16)
frames[:, 100:110, :] = 0                      # never-valid band
frames[:31, 20:30, :] = 0                      # valid 29/60 of the time (<50%)
bg = BackgroundModel()
bg.learn(frames)
check("bg learned", bg.ready and abs(bg.bg[80, 80] - 3000) < 5 and abs(bg.bg[10, 80] - 4000) < 5)
check("invalid where <50% valid", bg.invalid[105, 5] and bg.invalid[25, 25] and not bg.invalid[80, 80])
check("tol floor 40 mm", bg.tol[80, 80] >= 40.0)
frame = frames[0].copy()
frame[70:90, 70:90] = 2500                     # box 0.5 m closer
frame[40:45, 40:45] = 4300                     # farther than bg (multipath) -> ignore
fg = bg.foreground(frame)
check("closer box is fg", fg[70:90, 70:90].all())
check("farther noise is not fg", not fg[40:45, 40:45].any())
check("background is not fg", not fg[120:140, 120:140].any())
check("invalid band with a return is fg", fg[105, 5] == (frame[105, 5] > 0))
near, far = frame.copy(), frame.copy()
near[105, 5], far[105, 5] = 2000, 5200
check("never-learned pixel: return inside range is fg, beyond the range gate is not",
      bg.foreground(near)[105, 5] and not bg.foreground(far)[105, 5])
still = frames[1].copy()
still[70:90, 70:90] = 2500
fgs = bg.foreground(still)
for i in range(3000):                          # 100 s at 30 Hz with a 'person' present -> no adaptation
    bg.slow_update(still, fgs, scene_empty=False, now=i / 30)
check("no absorption while occupied", abs(bg.bg[80, 80] - 3000) < 5)
for i in range(3000, 3000 + 30 * 130):         # 130 s empty
    bg.slow_update(still, np.zeros_like(fgs), scene_empty=True, now=i / 30)
check("adapts only after empty_s", bg.bg[80, 80] < 2990, str(bg.bg[80, 80]))
bg4 = BackgroundModel(absorb_s=600.0)
bg4.learn(frames)
intr = frames[0].copy()
intr[70:90, 70:90] = 2500                      # a chair moved 0.5 m closer
intr[105, :] = 3500                            # a steady return where nothing was learned
for _ in range(60):
    bg4.absorb(intr, 1 / 30)
check("2 s intruder is not absorbed", bg4.foreground(intr)[80, 80] and bg4.foreground(intr)[105, 5] and abs(bg4.bg[80, 80] - 3000) < 5)
n = sum(bg4.absorb(intr, 1.0) for _ in range(601))
check("601 s intruder is healed into the session overlay, the ground truth untouched",
      n >= 400 + 160 and abs(bg4.heal[80, 80] - 2500) < 1 and abs(bg4.bg[80, 80] - 3000) < 5 and not bg4.foreground(intr)[80, 80], str(n))
check("healed never-learned pixel is excused, and stays never-learned", bg4.invalid[105, 5] and abs(bg4.heal[105, 5] - 3500) < 1 and not bg4.foreground(intr)[105, 5])
gone = frames[0].copy()
check("after it leaves nothing is fg (farther never is)", not bg4.foreground(gone)[70:90, 70:90].any())
for _ in range(3):
    bg4.absorb(gone, 1.0)
check("un-healed within unheal_s of the learned surface showing again", np.isnan(bg4.heal[80, 80]) and np.isnan(bg4.heal[105, 5]) and not bg4.foreground(gone)[80, 80],
      str(bg4.heal[80, 80]))
back = intr.copy()
back[70:90, 70:90] = 2800                     # someone else stands where the chair was, farther than it, closer than the wall
check("the old spot is not blind: a closer return there is foreground at once", bg4.foreground(back)[70:90, 70:90].all())
bg5 = BackgroundModel(absorb_s=600.0)
bg5.learn(frames)
blank = intr.copy()
blank[70:90, 70:90] = 0                       # the dark chair returns nothing every other frame
n = sum(bg5.absorb(f, 1.0) for f in [intr, blank] * 601)
check("flickery object is healed on accumulated disagreement", n >= 400 and abs(bg5.heal[80, 80] - 2500) < 1, str(n))
# seen in the field: a wall edge 85 mm off (tol 80) flickered across the tolerance and never healed in two hours
bg6 = BackgroundModel(tol_mm=80.0, absorb_s=600.0)
bg6.learn(frames)
rng_h = np.random.default_rng(7)
n6 = 0
for _ in range(601):
    f = frames[0].astype(np.float32)
    f[115:135, 30:50] -= 85 + rng_h.normal(0, 8, (20, 20))   # nudged onto the tolerance: readings 60..110 mm closer
    f[115:135, 100:120] += rng_h.normal(0, 5, (20, 20))      # the learned surface with its own noise
    f[70:80, 120:140] -= 60                                  # inside the tolerance: never foreground
    f = np.clip(f, 0, 65535).astype(np.uint16)
    n6 += bg6.absorb(f, 1.0)
check("surface hovering at the tolerance is absorbed after absorb_s and quiet afterwards",
      n6 >= 400 + 200 and not bg6.foreground(f)[115:135, 30:50].any() and abs(bg6.heal[125, 40] - (int(frames[0][125, 40]) - 85)) < 40, str(n6))
check("a learned surface with its own noise never ages", bg6.cand_age[115:135, 100:120].max() < 2.0, str(bg6.cand_age[115:135, 100:120].max()))
check("a shift inside the tolerance is healed as the new reference, so a later nudge cannot stack onto it",
      abs(bg6.heal[75, 130] - (int(frames[0][75, 130]) - 60)) < 1 and not bg6.foreground(f)[70:80, 120:140].any(), str(bg6.heal[75, 130]))
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "bg.npz")
    bg.save(p)
    bg2 = BackgroundModel()
    check("save/load", bg2.load(p) and np.allclose(bg2.bg, bg.bg) and bg2.age_s() is not None)
bg3 = BackgroundModel()
bg3.begin_learning(5)
for i in range(5):
    done = bg3.feed_learning(frames[i])
check("incremental learning completes", done and bg3.ready and not bg3.learning)

# ---- a robot pose nobody has confirmed is not a pose ----
# The tracker masks the arm out of the depth clouds. If the joint feed dies, holding the last pose means
# masking where the arm WAS; the arm keeps moving and its points become a cluster inside the rail.
print("== stale robot pose widens the mask ==")
from rt_depth_tracker.pipeline import Pipeline  # noqa: E402

_cfg = _scene.cfg()
_pipe = Pipeline(_cfg, robot=RobotModel.from_config(_cfg))
check("q_age starts unknown, not fresh", _pipe.q_age is None or _pipe.q_age == 0.0, str(_pipe.q_age))
_pipe.set_q(np.zeros(6), None)
check("set_q keeps a None age", _pipe.q_age is None, str(_pipe.q_age))
_pipe.set_q(np.zeros(6), 4.2)
check("set_q keeps a real age", abs(_pipe.q_age - 4.2) < 1e-9, str(_pipe.q_age))

# points strung along +x at the flange height, i.e. right where the arm lives
_home = np.array([0.0, 1.012291, -0.785398, 0.0, -0.226893, 0.0])
_base_z = _cfg.mount_z
_P = np.array([[x, 0.0, _base_z + 0.15] for x in np.linspace(0.05, 0.85, 40)], np.float32)
_rm = RobotModel.from_config(_cfg)
_tight = _rm.inside_mask(_P, _home, _cfg.robot_margin, _cfg.pedestal_radius)
_wide = _rm.inside_mask(_P, _home, _cfg.robot_stale_margin, _cfg.pedestal_radius)
check("the normal margin leaves points near the arm", not _tight.all(), f"{int(_tight.sum())}/{len(_P)}")
check("the stale margin covers every one of them", _wide.all(), f"{int(_wide.sum())}/{len(_P)}")
check("stale margin is the wider of the two", _cfg.robot_stale_margin > _cfg.robot_margin)
# and it must still be inside the rail, or it would swallow visitors
check("stale envelope stays inside the rail", _cfg.robot_stale_margin < _cfg.rail_radius,
      f"{_cfg.robot_stale_margin} vs rail {_cfg.rail_radius}")

print("== the robot is optional, and the config resolves its files ==")
check("no robot block -> None", RobotModel.from_config(_scene.cfg(robot={})) is None)
check("root link and dof read from the URDF", rm.root == "link_base" and rm.n_dof == 6 and rm.links[-1] == "link6")
import json as _json  # noqa: E402
_tmp = tempfile.mkdtemp(prefix="rt_cfg_")
_cp = os.path.join(_tmp, "tracker.json")
import shutil as _shutil  # noqa: E402
for _k in ("urdf", "robot_description"):
    _shutil.copy(_scene.ROBOT[_k], _tmp)
_json.dump({"robot": {"urdf": os.path.basename(_scene.ROBOT["urdf"]), "robot_description": os.path.basename(_scene.ROBOT["robot_description"])}}, open(_cp, "w"))
_lc = TrackerConfig.load(_cp)
check("relative robot paths resolve against the config file's directory", _lc.config_dir == _tmp and RobotModel.from_config(_lc) is not None)
check("config_dir is not written back", "config_dir" not in _lc.to_dict())
_c4 = TrackerConfig()
_c4.apply({"zones": [{"name": "a", "rect": [0, 0, 1, 1]}], "wall_x": -1.2})
_c4.apply({"zones": [], "wall_x": None})
check("lists are replaced whole and a null clears an optional value", _c4.zones == [] and _c4.wall_x is None)

print("== min_pts: range-scaled beside the base, flat under an overhead camera ==")
from rt_depth_tracker.segment import min_pts_at  # noqa: E402
_scaled = _scene.cfg()
check("beside the base: 40 at 3 m, 360 at 1 m", min_pts_at(_scaled, 3.0) == 40 and abs(min_pts_at(_scaled, 1.0) - 360) < 1e-9)
_flat = _scene.cfg(min_pts_at_3m=0, min_pts=40)
check("overhead: 40 everywhere", min_pts_at(_flat, 0.3) == 40 and min_pts_at(_flat, 3.0) == 40)
check("both set: the larger wins", min_pts_at(_scene.cfg(min_pts=100), 3.0) == 100 and min_pts_at(_scene.cfg(min_pts=10), 1.0) == 360)

print(f"test_geometry: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
