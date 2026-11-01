# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""RGB-assisted ball detection on the synthetic scene: orange blob -> masked depth -> sphere fit."""
import os
import sys

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import calib as C  # noqa: E402
from rt_depth_tracker import rgb  # noqa: E402
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.intrinsics import ray_table  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


rm = RobotModel.from_config(_scene.cfg())
cams = S.two_camera_rig()
intr = S.intrinsics_for(0)
rays = ray_table(intr, 2)
rng = np.random.default_rng(3)

print("blob detection")
ball = (1.2, 0.35, 1.7, 0.075)
scene = S.Scene(rail_radius=1.4, q=np.array([0, 0.26, -0.9, 0, 0.6, 0]), robot=rm, balls=[ball])
depth, bgr, truth_px = S.render_rgb(scene, cams[0], rng=rng)
blob = rgb.find_blob(bgr)
check("orange blob found", blob is not None and blob.area >= 20, str(blob and blob.area))
check("blob matches rendered ball pixels", blob is not None and (blob.mask & truth_px).sum() / max(1, truth_px.sum()) > 0.8
      and (blob.mask & ~truth_px).sum() < 0.3 * truth_px.sum(), f"{blob and (blob.mask & truth_px).sum()}/{truth_px.sum()}")
grey = np.full_like(bgr, 140)
check("no blob on a grey scene", rgb.find_blob(grey) is None)

print("sphere fit from the blob")
c_s, rms, n_in, _ = rgb.fit_ball(depth, bgr, intr, 2, 0.075, rays=rays)
check("fit returns", c_s is not None, str(rms))
c_w = C.apply(cams[0], c_s[None])[0]
err = np.linalg.norm(c_w - np.array(ball[:3]))
check("centre within 15 mm of truth (world)", err < 0.015, f"err={err*1000:.1f} mm rms={rms*1000:.1f} n={n_in}")

print("multi-pose: blob-fit centres reproduce the camera pose via Kabsch")
A, B = [], []
for k in range(12):
    az = np.radians(-90 + 180 * k / 11)       # toward the visitors (+x)
    r = 0.45 + 0.35 * (k % 3) / 2
    b = (r * np.cos(az), r * np.sin(az), rm.T_world_base[2, 3] + 0.15 + 0.6 * ((k * 7) % 12) / 11, 0.075)
    sc = S.Scene(rail_radius=1.4, q=np.zeros(6), robot=rm, balls=[b])
    d, im, _ = S.render_rgb(sc, cams[1], rng=rng)
    c_s, rms, n_in, bl = rgb.fit_ball(d, im, S.intrinsics_for(1), 2, 0.075, rays=rays)
    if c_s is not None:
        A.append(c_s)
        B.append(b[:3])
check("balls found at >= 10 of 12 poses", len(A) >= 10, str(len(A)))
T, rms = C.kabsch(np.array(A), np.array(B))
dt = np.linalg.norm(T[:3, 3] - cams[1][:3, 3])
dR = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ cams[1][:3, :3]) - 1) / 2, -1, 1)))
check("recovered pose within 20 mm / 0.5 deg", dt < 0.02 and dR < 0.5, f"dt={dt*1000:.1f} mm dR={dR:.2f} deg rms={rms*1000:.1f} mm")

print("depth gate rejects background bleeding into the mask")
blob = rgb.find_blob(bgr)
dil = blob.mask.copy()
import cv2  # noqa: E402
dil = cv2.dilate(dil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
fat = rgb.Blob(dil, int(dil.sum()), blob.centroid, blob.bbox)
P = rgb.blob_points(depth, fat, rays)
z = P[:, 2]
check("gated points span < 0.3 m in depth", z.max() - z.min() < 0.3, f"{z.max()-z.min():.2f}")

print("a bigger orange object in view (a cardboard box, a warm-lit door frame)")
box = (2.6, -0.6, 0.35, 0.30)                      # four times the ball's radius, further away, still more pixels
sc = S.Scene(rail_radius=1.4, q=np.array([0, 0.26, -0.9, 0, 0.6, 0]), robot=rm, balls=[ball, box])
d2, im2, _ = S.render_rgb(sc, cams[0], rng=rng)
blobs = rgb.find_blobs(im2)
check("both orange things are found, largest first", len(blobs) >= 2 and blobs[0].area >= blobs[1].area, str([b.area for b in blobs]))
c_big, _, _, _ = rgb.fit_ball(d2, im2, intr, 2, 0.075, rays=rays)
far = c_big is None or np.linalg.norm(C.apply(cams[0], c_big[None])[0] - np.array(ball[:3])) > 0.5
check("without an expected position the bigger object wins", far)
rough = C.apply(C.invert(cams[0]), (np.array(ball[:3]) + [0.35, -0.3, 0.25])[None])[0]    # a seed calibration ~0.5 m out
c_s, rms, n_in, _ = rgb.fit_ball(d2, im2, intr, 2, 0.075, rays=rays, near=rough, near_r=1.2)
err = np.inf if c_s is None else np.linalg.norm(C.apply(cams[0], c_s[None])[0] - np.array(ball[:3]))
check("with it, the ball is found within 15 mm", err < 0.015, f"err={err*1000:.1f} mm")
nowhere = C.apply(C.invert(cams[0]), np.array([[0.5, 2.5, 2.0]]))[0]
check("nothing orange near the expected position: not seen, rather than the wrong thing",
      rgb.fit_ball(d2, im2, intr, 2, 0.075, rays=rays, near=nowhere, near_r=0.5)[0] is None)

print("color decode helpers")
img = np.zeros((4, 6, 3), np.uint8)
img[..., 2] = 200
check("hue wrap range accepted", rgb.find_blob(img, (170, 50, 50), (10, 255, 255), min_px=1) is not None)

print(f"test_rgb: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
