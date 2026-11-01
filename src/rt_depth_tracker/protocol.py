# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""OSC wire protocol: sensor process -> tracker -> consumer (a robot behaviour app, a show controller, the viewer).

All transports are loopback UDP; one OSC message per datagram (no bundles).
Depth frames are sent as the organized uint16 image (2D stride applied) so the
tracker can do per-pixel background subtraction; at stride 2 a 320x288 NFOV-binned
frame is 46 080 B and fits one datagram (IPv4 UDP payload cap 65 507 B).

Pure Python + numpy + python-osc: importable on the Mac for replay and tests.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass

import numpy as np
from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

# ---- ports (loopback) --------------------------------------------------------
PORT_DEPTH = 7010        # capture (one process per camera) -> tracker
PORT_TRACKER_OUT = 7001  # tracker -> consumer
PORT_TRACKER_MIRROR = 7016  # tracker -> viewer --live (the target, tracks and zones the consumer receives)
PORT_ROBOT_Q = 7011      # robot joint feed -> tracker (only with a robot configured)
PORT_TRACKER_CTRL = 7012  # calib tool / ctl / a supervisor -> tracker
PORT_CAPTURE_CTRL = 7017  # a supervisor -> capture (+ cam_id: 7017, 7018); 7014/7015 are the viewer's q mirror and calib validate
PORT_HB = 9000           # every process -> a supervisor, if there is one

# ---- addresses ---------------------------------------------------------------
ADDR_DEPTH = "/depth"
ADDR_INTRINSICS = "/intrinsics"
ADDR_HB = "/hb"
ADDR_TARGET = "/tracker/target"
ADDR_TRACKS = "/tracker/tracks"
ADDR_ROBOT_Q = "/robot/q"
ADDR_ZONE = "/tracker/zone"       # one per configured zone per frame; never sent without zones
ADDR_LEARN_BG = "/tracker/learn_bg"
ADDR_RECORD = "/tracker/record"
ADDR_RELOAD_CALIB = "/tracker/reload_calib"
ADDR_SET = "/tracker/set"          # <key:s> <value:f>: a slider somewhere; applied at once, saved once it settles
TUNABLE_KEYS = ("near_pref", "kid_pref", "kid_h", "adult_h")
ADDR_TRACKER_QUIT = "/tracker/quit"    # clean stop, exit 0 (a CTRL_BREAK never reaches a child of pythonw)
ADDR_CAPTURE_QUIT = "/capture/quit"    # stop the stream, release the camera, exit 0

HINTS = ("idle", "watch", "engage", "reach", "sleep")
UDP_MAX_PAYLOAD = 65507


@dataclass
class DepthPacket:
    cam_id: int
    seq: int
    width: int      # full-resolution width before stride
    height: int
    stride: int
    device_ts: float  # seconds, camera clock
    depth: np.ndarray  # (height//stride, width//stride) uint16, mm


@dataclass
class Intrinsics:
    cam_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def scaled(self, stride: int) -> "Intrinsics":
        """Intrinsics for the strided image. Pixel centre convention: sub-sampling every
        `stride`-th pixel starting at 0 maps full pixel u to u/stride."""
        s = 1.0 / stride
        return Intrinsics(self.cam_id, self.width // stride, self.height // stride,
                          self.fx * s, self.fy * s, self.cx * s, self.cy * s,
                          self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.p1, self.p2)


@dataclass
class Target:
    x: float
    y: float
    z: float
    hint: str
    track_id: int
    score: float


@dataclass
class Heartbeat:
    name: str
    metrics: dict


@dataclass
class RobotQ:
    q: np.ndarray  # (n_joints,) rad


@dataclass
class ZoneMsg:
    idx: int
    name: str
    occupied: int
    reaching: int
    track_id: int
    dwell_s: float
    leaning: int = 0


# ---- builders ----------------------------------------------------------------
def _msg(address: str, *typed_args) -> bytes:
    b = OscMessageBuilder(address=address)
    for value, tag in typed_args:
        b.add_arg(value, tag)
    return b.build().dgram


I, F, D, S, B = (OscMessageBuilder.ARG_TYPE_INT, OscMessageBuilder.ARG_TYPE_FLOAT,
                 OscMessageBuilder.ARG_TYPE_DOUBLE, OscMessageBuilder.ARG_TYPE_STRING,
                 OscMessageBuilder.ARG_TYPE_BLOB)


def stride_depth(depth_full: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return np.ascontiguousarray(depth_full, dtype=np.uint16)
    return np.ascontiguousarray(depth_full[::stride, ::stride], dtype=np.uint16)


CHUNK_BYTES = 8000   # macOS caps datagrams at ~9 KB; Windows/Linux allow 64 KB. 8 KB works everywhere.


def pack_depth(cam_id: int, seq: int, depth_full: np.ndarray, stride: int, device_ts: float,
               chunk_bytes: int = CHUNK_BYTES) -> list[bytes]:
    """One strided depth frame -> list of /depth datagrams (chunked). Header per chunk:
    cam_id, seq, w, h, stride, device_ts, chunk_idx, n_chunks, blob."""
    h, w = depth_full.shape
    sub = stride_depth(depth_full, stride)
    blob = sub.astype("<u2", copy=False).tobytes()
    n = max(1, -(-len(blob) // chunk_bytes))
    out = []
    for i in range(n):
        part = blob[i * chunk_bytes:(i + 1) * chunk_bytes]
        out.append(_msg(ADDR_DEPTH, (cam_id, I), (seq, I), (w, I), (h, I), (stride, I),
                        (float(device_ts), D), (i, I), (n, I), (part, B)))
    return out


def pack_depth_packet(pkt: DepthPacket, chunk_bytes: int = CHUNK_BYTES) -> list[bytes]:
    """Chunk an existing (already strided) DepthPacket without re-striding it."""
    blob = np.ascontiguousarray(pkt.depth, dtype="<u2").tobytes()
    n = max(1, -(-len(blob) // chunk_bytes))
    return [_msg(ADDR_DEPTH, (pkt.cam_id, I), (pkt.seq, I), (pkt.width, I), (pkt.height, I), (pkt.stride, I),
                 (float(pkt.device_ts), D), (i, I), (n, I), (blob[i * chunk_bytes:(i + 1) * chunk_bytes], B))
            for i in range(n)]


@dataclass
class DepthChunk:
    cam_id: int
    seq: int
    width: int
    height: int
    stride: int
    device_ts: float
    idx: int
    n: int
    blob: bytes


class DepthAssembler:
    """Reassembles chunks into DepthPackets per camera. A newer seq discards an incomplete older one; a lower seq
    is a late chunk, unless the pending frame is older than stale_s: then it is a restarted sender counting from 1."""

    def __init__(self, clock=time.monotonic, stale_s: float = 1.0):
        self._clock, self._stale_s = clock, float(stale_s)
        self._parts: dict[int, tuple[int, list, float]] = {}   # cam -> (seq, [chunk|None]*n, t_first)

    def feed(self, c: DepthChunk):
        cur = self._parts.get(c.cam_id)
        if cur is None or cur[0] != c.seq:
            if cur is not None and c.seq < cur[0] and self._clock() - cur[2] <= self._stale_s:
                return None          # late chunk of an older frame
            cur = (c.seq, [None] * c.n, self._clock())
            self._parts[c.cam_id] = cur
        parts = cur[1]
        if c.idx < len(parts):
            parts[c.idx] = c
        if any(p is None for p in parts):
            return None
        blob = b"".join(p.blob for p in parts)
        hs, ws = c.height // c.stride, c.width // c.stride
        if len(blob) < hs * ws * 2:
            return None
        depth = np.frombuffer(blob, dtype="<u2", count=hs * ws).reshape(hs, ws).copy()
        del self._parts[c.cam_id]
        return DepthPacket(c.cam_id, c.seq, c.width, c.height, c.stride, c.device_ts, depth)


def unpack_depth_all(dgrams: list[bytes]) -> "DepthPacket | None":
    """Convenience for tests: chunks -> packet."""
    assembler = DepthAssembler()
    pk = None
    for d in dgrams:
        obj = unpack(d)
        if isinstance(obj, DepthChunk):
            pk = assembler.feed(obj) or pk
    return pk


def pack_intrinsics(k: Intrinsics) -> bytes:
    return _msg(ADDR_INTRINSICS, (k.cam_id, I), (k.width, I), (k.height, I),
                *[(float(v), F) for v in (k.fx, k.fy, k.cx, k.cy, k.k1, k.k2, k.k3,
                                          k.k4, k.k5, k.k6, k.p1, k.p2)])


def pack_hb(name: str, metrics: dict) -> bytes:
    return _msg(ADDR_HB, (name, S), (json.dumps(metrics, separators=(",", ":")), S))


def pack_target(t: Target) -> bytes:
    if t.hint not in HINTS:
        raise ValueError(f"bad hint {t.hint!r}")
    return _msg(ADDR_TARGET, (float(t.x), F), (float(t.y), F), (float(t.z), F),
                (t.hint, S), (int(t.track_id), I), (float(t.score), F))


def pack_robot_q(q) -> bytes:
    q = [float(v) for v in q]
    if not q:
        raise ValueError("q must have at least one joint")
    return _msg(ADDR_ROBOT_Q, *[(v, F) for v in q])


def pack_zone(z) -> bytes:
    """z: anything with idx, name, occupied, reaching, track_id, dwell_s, leaning (a zones.ZoneState or a ZoneMsg)."""
    return _msg(ADDR_ZONE, (int(z.idx), I), (str(z.name), S), (int(bool(z.occupied)), I), (int(bool(z.reaching)), I),
                (int(z.track_id), I), (float(z.dwell_s), F), (int(bool(getattr(z, "leaning", 0))), I))


def pack_tracks(rows: np.ndarray) -> bytes:
    """rows: (n, 8) float32 [id x y z height score reach flags]."""
    rows = np.ascontiguousarray(rows, dtype=np.float32).reshape(-1, 8)
    blob = rows.tobytes() or b"\x00"      # python-osc refuses empty blobs; count carries the truth
    return _msg(ADDR_TRACKS, (int(len(rows)), I), (blob, B))


# ---- parser ------------------------------------------------------------------
def unpack(dgram: bytes):
    """Parse one datagram. Returns a dataclass, a (address, params) tuple for control
    messages, or None for anything unknown / malformed."""
    try:
        msg = OscMessage(dgram)
    except Exception:
        return None
    a, p = msg.address, msg.params
    try:
        if a == ADDR_DEPTH:
            cam_id, seq, w, h, stride, ts, idx, n, blob = p[:9]
            return DepthChunk(int(cam_id), int(seq), int(w), int(h), int(stride), float(ts), int(idx), int(n), bytes(blob))
        if a == ADDR_INTRINSICS:
            return Intrinsics(int(p[0]), int(p[1]), int(p[2]), *[float(v) for v in p[3:15]])
        if a == ADDR_HB:
            return Heartbeat(str(p[0]), json.loads(p[1]))
        if a == ADDR_TARGET:
            return Target(float(p[0]), float(p[1]), float(p[2]), str(p[3]), int(p[4]), float(p[5]))
        if a == ADDR_ROBOT_Q:
            return RobotQ(np.asarray(p, dtype=np.float64))
        if a == ADDR_ZONE:
            return ZoneMsg(int(p[0]), str(p[1]), int(p[2]), int(p[3]), int(p[4]), float(p[5]), int(p[6]) if len(p) > 6 else 0)
        if a == ADDR_TRACKS:
            n = int(p[0])
            return np.frombuffer(p[1], dtype=np.float32, count=n * 8).reshape(n, 8).copy()
        return (a, list(p))
    except Exception:
        return None


def depth_datagram_size(width: int, height: int, stride: int) -> int:
    """Approximate datagram size for a given frame geometry (for choosing stride)."""
    return 64 + 2 * (width // stride) * (height // stride)


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("struct",)]
