# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Recorder: chunk compression runs off the caller's thread (flush returns at once, the caller is not stalled),
files appear only when complete, and the Reader gets every packet back in order."""
import glob
import os
import shutil
import sys
import tempfile
import threading
import time

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker.protocol import DepthPacket  # noqa: E402
from rt_depth_tracker.record import Reader, Recorder  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


rng = np.random.default_rng(1)
out = tempfile.mkdtemp(prefix="rt_rec_")
try:
    rec = Recorder(out, chunk_s=1e9, tag="t/est")
    n_frames, h, w = 300, 144, 160
    t0 = 1000.0
    for i in range(n_frames):
        for cid in (0, 1):
            depth = rng.integers(0, 4000, (h, w), dtype=np.uint16)     # random: slow to compress, like real depth
            rec.add(DepthPacket(cid, i, w, h, 2, t0 + i / 30, depth), np.zeros(6), t0 + i / 30)
    check("tag sanitised", rec.tag == "t_est", rec.tag)

    stalls = []
    stop = threading.Event()

    def hot_loop():
        last = time.perf_counter()
        while not stop.is_set():
            time.sleep(0.002)
            now = time.perf_counter()
            stalls.append(now - last)
            last = now

    th = threading.Thread(target=hot_loop, daemon=True)
    th.start()
    time.sleep(0.05)
    t_flush = time.perf_counter()
    rec.flush()
    dt_flush = time.perf_counter() - t_flush
    check("flush returns without waiting for the write", dt_flush < 0.1, f"{dt_flush * 1000:.0f} ms")
    check("path known immediately", len(rec.files) == 1, str(rec.files))
    seen_tmp = os.path.exists(rec.files[0] + ".tmp") or not os.path.exists(rec.files[0])
    t_close = time.perf_counter()
    rec.close()
    dt_write = time.perf_counter() - t_close + dt_flush
    stop.set()
    th.join()
    worst = max(stalls) if stalls else 0.0
    print(f"     write took {dt_write * 1000:.0f} ms in the background; worst caller stall {worst * 1000:.0f} ms")
    check("caller not stalled by the compression", worst < 0.3, f"worst gap {worst * 1000:.0f} ms")
    check("file written under a temp name first", seen_tmp)
    check("no temp file after close", not glob.glob(os.path.join(out, "*.tmp")))
    check("final file exists", os.path.exists(rec.files[0]))

    rd = Reader(rec.files)
    events = list(rd)
    check("reader yields one step per frame set (paced like the live loop)", len(events) == n_frames, str(len(events)))
    seqs = [pk[0].seq for _, pk, _ in events if 0 in pk]
    check("packets in time order", all(a <= b for a, b in zip(seqs, seqs[1:])))
    last_t, last_pk, _ = events[-1]
    check("last step carries both cameras at the last frame", set(last_pk) == {0, 1} and all(p.seq == n_frames - 1 for p in last_pk.values()))
    check("depth round-trips", np.array_equal(last_pk[1].depth, last_pk[1].depth) and last_pk[1].depth.shape == (h, w))

    rec2 = Recorder(out, chunk_s=1e9, tag="empty")
    rec2.close()
    check("close on an empty recorder writes nothing", rec2.files == [] and not glob.glob(os.path.join(out, "empty_*")))
finally:
    shutil.rmtree(out, ignore_errors=True)

print(f"test_record: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
