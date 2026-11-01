# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Pinhole + Brown-Conrady (k1..k6, p1, p2) camera model. Ray tables are computed once per
(camera, stride) so per-frame unprojection is a single multiply."""
from __future__ import annotations

import numpy as np

from rt_depth_tracker.protocol import Intrinsics


def distort(x: np.ndarray, y: np.ndarray, k: Intrinsics):
    """Normalized undistorted -> distorted (forward model)."""
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    radial = (1 + k.k1 * r2 + k.k2 * r4 + k.k3 * r6) / (1 + k.k4 * r2 + k.k5 * r4 + k.k6 * r6)
    xd = x * radial + 2 * k.p1 * x * y + k.p2 * (r2 + 2 * x * x)
    yd = y * radial + k.p1 * (r2 + 2 * y * y) + 2 * k.p2 * x * y
    return xd, yd


def undistort(xd: np.ndarray, yd: np.ndarray, k: Intrinsics, iters: int = 10):
    """Distorted -> undistorted normalized coords by fixed-point iteration (exact enough for ToF FOVs)."""
    if not any((k.k1, k.k2, k.k3, k.k4, k.k5, k.k6, k.p1, k.p2)):
        return xd.copy(), yd.copy()
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        xdd, ydd = distort(x, y, k)
        x = x - (xdd - xd)
        y = y - (ydd - yd)
    return x, y


def ray_table(k_full: Intrinsics, stride: int = 1) -> np.ndarray:
    """(H', W', 3) float32 rays [x/z, y/z, 1] for the strided image (pixel (u,v) of the strided
    image is full-res pixel (u*stride, v*stride); pixel centres at integer coords)."""
    h, w = k_full.height // stride, k_full.width // stride
    us = (np.arange(w, dtype=np.float64) * stride)
    vs = (np.arange(h, dtype=np.float64) * stride)
    U, V = np.meshgrid(us, vs)
    xd = (U - k_full.cx) / k_full.fx
    yd = (V - k_full.cy) / k_full.fy
    x, y = undistort(xd, yd, k_full)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
    return rays


def unproject(depth_mm: np.ndarray, rays: np.ndarray, mask: np.ndarray | None = None):
    """Depth image (H',W') uint16 mm + rays -> (N,3) float32 metres in the SENSOR frame
    (x right, y down, z forward) for valid (nonzero, masked) pixels. Also returns the flat
    pixel indices so callers can go back to image space."""
    valid = depth_mm > 0
    if mask is not None:
        valid &= mask
    idx = np.flatnonzero(valid)
    z = depth_mm.reshape(-1)[idx].astype(np.float32) * 0.001
    r = rays.reshape(-1, 3)[idx]
    return r * z[:, None], idx


def project(P_sensor: np.ndarray, k: Intrinsics, stride: int = 1):
    """(N,3) sensor-frame metres -> (u,v) pixel coords in the strided image (with distortion)."""
    z = np.maximum(P_sensor[:, 2], 1e-6)
    x, y = P_sensor[:, 0] / z, P_sensor[:, 1] / z
    xd, yd = distort(x, y, k)
    u = (xd * k.fx + k.cx) / stride
    v = (yd * k.fy + k.cy) / stride
    return u, v
