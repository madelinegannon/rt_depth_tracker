# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""RGB-assisted ball detection for the sphere calibration: colour stream aligned to the depth image, HSV blob,
depth pixels under the blob -> sensor points -> fixed-radius sphere fit. Pure functions are hardware-free;
RgbDepthCam wraps pyorbbecsdk2 (used only by calib_tool, never by the live tracker)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker.intrinsics import ray_table, unproject
from rt_depth_tracker.protocol import Intrinsics

ORANGE_LO = (5, 120, 80)      # OpenCV HSV: H 0-180
ORANGE_HI = (25, 255, 255)


@dataclass
class Blob:
    mask: np.ndarray          # (H,W) bool at depth resolution
    area: int
    centroid: tuple           # (u, v) pixels
    bbox: tuple               # (u0, v0, u1, v1)


def decode_color(frame) -> np.ndarray | None:
    """Orbbec color frame -> BGR uint8 (H,W,3). Handles RGB/BGR/MJPG/YUYV/UYVY/NV12."""
    import cv2
    from pyorbbecsdk import OBFormat
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    return None


def find_blobs(bgr: np.ndarray, hsv_lo=ORANGE_LO, hsv_hi=ORANGE_HI, min_px: int = 20) -> list[Blob]:
    """Every connected component of the HSV range with at least min_px pixels, largest first. Hue wraps if
    lo > hi (for red)."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
    if lo[0] <= hi[0]:
        m = cv2.inRange(hsv, lo, hi)
    else:
        m = cv2.inRange(hsv, lo, np.array([180, hi[1], hi[2]], np.uint8)) | cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA])):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_px:
            break
        x, y, w, h = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        out.append(Blob(lab == k, area, (float(cents[k][0]), float(cents[k][1])), (x, y, x + w, y + h)))
    return out


def find_blob(bgr: np.ndarray, hsv_lo=ORANGE_LO, hsv_hi=ORANGE_HI, min_px: int = 20) -> Blob | None:
    """Largest connected component of the HSV range."""
    blobs = find_blobs(bgr, hsv_lo, hsv_hi, min_px)
    return blobs[0] if blobs else None


def blob_points(depth_mm: np.ndarray, blob: Blob, rays: np.ndarray, depth_gate_m: float = 0.15) -> np.ndarray:
    """Sensor-frame points under the blob, gated to ±depth_gate_m around the blob's median depth so
    background pixels bleeding into the mask edge never join the fit."""
    P, idx = unproject(depth_mm, rays, blob.mask)
    if len(P) == 0:
        return P
    z_med = float(np.median(P[:, 2]))
    return P[np.abs(P[:, 2] - z_med) < depth_gate_m]


def fit_ball(depth_mm: np.ndarray, bgr: np.ndarray, intr: Intrinsics, stride: int, radius_m: float,
             hsv_lo=ORANGE_LO, hsv_hi=ORANGE_HI, min_px: int = 20, rays: np.ndarray | None = None,
             near: np.ndarray | None = None, near_r: float = 1.2):
    """One frame -> (centre_sensor_m, rms_m, n_inliers, blob) or (None, ..., blob/None).
    near: where the ball is expected, sensor frame (m). With it, the largest blob within near_r of that point
    is the ball; without it, the largest blob in the picture is (a cardboard box or a warm-lit door frame
    bigger than the ball wins)."""
    rays = ray_table(intr, stride) if rays is None else rays
    if near is None:
        blob = find_blob(bgr, hsv_lo, hsv_hi, min_px)
        if blob is None:
            return None, float("inf"), 0, None
        P = blob_points(depth_mm, blob, rays)
    else:
        blob, P = None, np.zeros((0, 3))
        for b in find_blobs(bgr, hsv_lo, hsv_hi, min_px):
            Pb = blob_points(depth_mm, b, rays)
            if len(Pb) >= min_px and np.linalg.norm(Pb.mean(0) - np.asarray(near, float)) <= near_r:
                blob, P = b, Pb
                break
        if blob is None:
            return None, float("inf"), 0, None
    if len(P) < min_px:
        return None, float("inf"), int(len(P)), blob
    # the visible cap is the near side: seed the centre one radius behind the median surface point
    c0 = P.mean(0) + np.array([0, 0, radius_m])
    c, rms, n_in = C.fit_sphere_fixed_radius(P, radius_m, c0=c0)
    return c, rms, n_in, blob


class RgbDepthCam:
    """An Orbbec camera over the network: binned depth + colour, colour software-aligned TO the depth image (so
    every colour pixel maps to a depth pixel). Open only while the capture process for it is stopped."""

    def __init__(self, ip: str, port: int = 8090, mode: str = "nfov 2x2 binned", res=(320, 288), fps: int = 30, stride: int = 1):
        from pyorbbecsdk import AlignFilter, Config, Context, OBFormat, OBSensorType, OBStreamType, Pipeline
        from rt_depth_tracker.capture import match_work_mode, Capture
        self.stride = stride
        self.ctx = Context()
        self.dev = self.ctx.create_net_device(ip, port)
        names = Capture._work_mode_names(self.dev)
        pick = match_work_mode(names, mode) if names else None
        if pick:
            cur = getattr(self.dev.get_depth_work_mode(), "name", "")
            if pick.replace("_", "").upper() != str(cur).replace("_", "").upper():
                self.dev.set_depth_work_mode(pick)
        self.pipe = Pipeline(self.dev)
        cfg = Config()
        dl = self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            dp = dl.get_video_stream_profile(res[0], res[1], OBFormat.Y16, fps)
        except Exception:  # noqa: BLE001
            dp = dl.get_default_video_stream_profile()
        cfg.enable_stream(dp)
        cl = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        cp = None
        for fmt in (OBFormat.RGB, OBFormat.MJPG, OBFormat.YUYV):
            try:
                cp = cl.get_video_stream_profile(1280, 720, fmt, fps)
                break
            except Exception:  # noqa: BLE001
                continue
        if cp is None:
            cp = cl.get_default_video_stream_profile()
        cfg.enable_stream(cp)
        self.pipe.enable_frame_sync()
        self.pipe.start(cfg)
        self.align = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
        k, d = dp.get_intrinsic(), dp.get_distortion()
        self.intr = Intrinsics(0, int(k.width), int(k.height), float(k.fx), float(k.fy), float(k.cx), float(k.cy),
                               float(d.k1), float(d.k2), float(d.k3), float(d.k4), float(d.k5), float(d.k6), float(d.p1), float(d.p2))
        self.rays = ray_table(self.intr, stride)
        self.depth_profile, self.color_profile = dp, cp

    def read(self, timeout_ms: int = 1000):
        """(depth_mm uint16 (h,w), bgr (h,w,3), device_ts_s) or None."""
        fs = self.pipe.wait_for_frames(timeout_ms)
        if fs is None:
            return None
        out = self.align.process(fs)
        if out is None:
            return None
        fs2 = out.as_frame_set()
        df, cf = fs2.get_depth_frame(), fs2.get_color_frame()
        if df is None or cf is None:
            return None
        h, w = df.get_height(), df.get_width()
        raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(h, w)
        scale = float(df.get_depth_scale())
        depth = raw if abs(scale - 1.0) < 1e-6 else np.clip(raw.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
        bgr = decode_color(cf)
        if bgr is None or bgr.shape[:2] != (h, w):
            return None
        if self.stride > 1:
            depth, bgr = depth[::self.stride, ::self.stride], bgr[::self.stride, ::self.stride]
        return depth, bgr, df.get_timestamp_us() * 1e-6

    def read_n(self, n: int, timeout_s: float = 5.0):
        frames, t0 = [], time.monotonic()
        while len(frames) < n and time.monotonic() - t0 < timeout_s:
            r = self.read(500)
            if r is not None:
                frames.append(r)
        return frames

    def close(self):
        try:
            self.pipe.stop()
        except Exception:  # noqa: BLE001
            pass
