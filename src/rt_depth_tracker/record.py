# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Recorder / Reader for depth streams: 30 s .npz chunks with per-camera depth, timestamps, robot q,
intrinsics and the calibration in force. Replayable on any machine."""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np

from rt_depth_tracker.pacing import Pacer
from rt_depth_tracker.protocol import DepthPacket, Intrinsics


class Recorder:
    def __init__(self, out_dir: str, chunk_s: float = 30.0, tag: str = "rec"):
        self.out_dir, self.chunk_s = out_dir, chunk_s
        self.tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag) or "rec"
        os.makedirs(out_dir, exist_ok=True)
        self._t0 = None
        self._n = 0
        self._buf: dict[int, list] = {}
        self._q: list = []
        self.meta: dict = {"intrinsics": {}, "calib": {}}
        self.files: list[str] = []
        self._writer: threading.Thread | None = None
        self._last_seq: dict[int, int] = {}

    def set_meta(self, intr: dict[int, Intrinsics] | None = None, calib: dict[int, np.ndarray] | None = None):
        if intr:
            self.meta["intrinsics"] = {str(k): v.__dict__ for k, v in intr.items()}
        if calib:
            self.meta["calib"] = {str(k): np.asarray(v).tolist() for k, v in calib.items()}

    def add(self, pkt: DepthPacket, q, t: float):
        if self._last_seq.get(pkt.cam_id) == pkt.seq:      # the same frame offered twice is recorded once
            return
        self._last_seq[pkt.cam_id] = pkt.seq
        if self._t0 is None:
            self._t0 = t
        self._buf.setdefault(pkt.cam_id, []).append((t, pkt.device_ts, pkt.seq, pkt.stride, pkt.width, pkt.height, pkt.depth.copy()))
        self._q.append((t, *[float(v) for v in q]))
        if t - self._t0 >= self.chunk_s:
            self.flush()

    def flush(self):
        """Hand the chunk to a writer thread: compressing 30 s of depth takes seconds and must not stall the tracker."""
        if not self._buf:
            return
        buf, q, n = self._buf, self._q, self._n
        self._buf, self._q, self._t0 = {}, [], None
        self._n += 1
        path = os.path.join(self.out_dir, f"{self.tag}_{time.strftime('%Y%m%d_%H%M%S')}_{n:04d}.npz")
        self.files.append(path)
        self.join()
        self._writer = threading.Thread(target=self._write, args=(path, buf, q, json.dumps(self.meta)), daemon=True, name="recorder-write")
        self._writer.start()

    @staticmethod
    def _write(path: str, buf: dict, q: list, meta: str):
        arrays = {"q": np.asarray(q, np.float64), "meta": np.frombuffer(meta.encode(), np.uint8)}
        for cid, rows in buf.items():
            arrays[f"t_{cid}"] = np.array([r[0] for r in rows], np.float64)
            arrays[f"dts_{cid}"] = np.array([r[1] for r in rows], np.float64)
            arrays[f"seq_{cid}"] = np.array([r[2] for r in rows], np.int64)
            arrays[f"geom_{cid}"] = np.array([rows[0][3], rows[0][4], rows[0][5]], np.int64)
            arrays[f"depth_{cid}"] = np.stack([r[6] for r in rows])
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)      # readers only ever see complete files

    def join(self):
        if self._writer is not None:
            self._writer.join()
            self._writer = None

    def close(self):
        self.flush()
        self.join()

class Reader:
    """Iterates chunk files in time order and yields one step per new frame set, paced like the live tracker:
    (t, {cam_id: latest DepthPacket}, q). Repeated frames in older clips are stepped once."""

    def __init__(self, paths: list[str]):
        self.paths = sorted(paths)
        first = np.load(self.paths[0])
        self.meta = json.loads(bytes(first["meta"]).decode()) if "meta" in first else {}
        self.intrinsics = {int(k): Intrinsics(**v) for k, v in self.meta.get("intrinsics", {}).items()}
        self.calib = {int(k): np.array(v) for k, v in self.meta.get("calib", {}).items()}

    def __iter__(self):
        latest: dict[int, DepthPacket] = {}
        pacer = Pacer()
        for path in self.paths:
            d = np.load(path)
            cams = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("depth_")})
            q = d["q"]
            events = []
            for cid in cams:
                t, dts, seq, geom, depth = d[f"t_{cid}"], d[f"dts_{cid}"], d[f"seq_{cid}"], d[f"geom_{cid}"], d[f"depth_{cid}"]
                for i in range(len(t)):
                    events.append((float(t[i]), cid, DepthPacket(cid, int(seq[i]), int(geom[1]), int(geom[2]), int(geom[0]), float(dts[i]), depth[i])))
            events.sort(key=lambda e: e[0])
            qi = 0
            for t, cid, pkt in events:
                latest[cid] = pkt
                if pacer.offer(latest, t) is None:
                    continue
                while qi + 1 < len(q) and q[qi + 1, 0] <= t:
                    qi += 1
                yield t, dict(latest), q[qi, 1:] if len(q) else np.zeros(6)
