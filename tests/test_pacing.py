# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Frame pacing: one step per new frame set at the camera rate, a late partner is waited for at most one
frame, a dead camera does not hold the others back, repeated frames never step; the Reader applies the
same rule so replay steps like the live tracker, and the Recorder stores a repeated frame once."""
import os
import shutil
import sys
import tempfile

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker.pacing import Pacer  # noqa: E402
from rt_depth_tracker.protocol import DepthPacket  # noqa: E402
from rt_depth_tracker.record import Reader, Recorder  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


def pkt(cid, seq):
    return DepthPacket(cid, seq, 8, 6, 2, float(seq), np.full((6, 8), 1000 + seq, np.uint16))


def same(a, b):
    return a is not None and set(a) == set(b) and all(a[c].seq == b[c].seq for c in a)


print("pacer")
pc = Pacer(max_wait=0.034)
check("a single live camera steps at once", same(pc.offer({0: pkt(0, 1)}, 0.0), {0: pkt(0, 1)}))
pc = Pacer(max_wait=0.034)
a = pc.offer({0: pkt(0, 1), 1: pkt(1, 1)}, 0.000)
check("both fresh -> step with both", a is not None and set(a) == {0, 1})
b = pc.offer({0: pkt(0, 1), 1: pkt(1, 1)}, 0.016)
check("same frames again -> no step", b is None)
c = pc.offer({0: pkt(0, 2), 1: pkt(1, 1)}, 0.033)
check("cam 0 fresh, cam 1 stale, within the wait -> hold", c is None)
d = pc.offer({0: pkt(0, 2), 1: pkt(1, 2)}, 0.043)
check("partner arrives 10 ms later -> one step carrying both fresh", d is not None and set(d) == {0, 1})
e = pc.offer({0: pkt(0, 3), 1: pkt(1, 2)}, 0.066)
f = pc.offer({0: pkt(0, 3), 1: pkt(1, 2)}, 0.101)
check("partner never comes -> step after max_wait with the fresh one only", e is None and same(f, {0: pkt(0, 3)}))
g = pc.offer({0: pkt(0, 4)}, 0.133)
check("cam 1 gone from the live set -> cam 0 steps alone immediately", same(g, {0: pkt(0, 4)}))

print("30 fps pair with a 12 ms phase offset -> 30 steps/s, every step has both fresh")
pc = Pacer(max_wait=0.034)
steps, both = 0, 0
latest = {}
events = sorted([(i / 30, 0, i) for i in range(90)] + [(i / 30 + 0.012, 1, i) for i in range(90)])
for t, cid, seq in events:
    latest[cid] = pkt(cid, seq)
    r = pc.offer(dict(latest), t)
    if r is not None:
        steps += 1
        both += set(r) == {0, 1}
check("90 frame pairs -> 90 steps", steps == 90, str(steps))
check("every step after the first carries both fresh frames (cam 1 is unknown at the very first step)", both >= 89, str(both))

print("reader groups an old-style clip (each frame recorded twice, stamps shared) into paired steps")
out = tempfile.mkdtemp(prefix="rt_pace_")
try:
    rec = Recorder(out, chunk_s=1e9, tag="dup")
    n = 60
    # emulate the old loop: at each 16 ms wake both cameras' latest packets were recorded, so each frame appears twice
    for k in range(2 * n):
        t = 100.0 + k * 0.016
        rec._last_seq = {}                                 # bypass the new dedupe to build the legacy layout
        rec.add(pkt(0, k // 2), np.zeros(6), t)
        rec._last_seq = {}
        rec.add(pkt(1, max(0, (k - 1) // 2)), np.zeros(6), t)
    rec.close()
    d = np.load(rec.files[0])
    check("clip really holds each frame twice", len(d["seq_0"]) == 2 * n and len(np.unique(d["seq_0"])) == n)
    steps = list(Reader(rec.files))
    seqs0 = [pk[0].seq for _, pk, _ in steps]
    check("one step per frame pair", len(steps) == n, str(len(steps)))
    check("steps in order, no repeats", seqs0 == list(range(n)), str(seqs0[:8]))
    check("each step after the first carries both cameras", all(set(pk) == {0, 1} for _, pk, _ in steps[1:]))
    check("after startup each step pairs the SAME frame of both cameras (no one-frame skew)", all(pk[0].seq == pk[1].seq for _, pk, _ in steps[1:]),
          str([(pk[0].seq, pk[1].seq) for _, pk, _ in steps[1:5]]))
    dts = np.diff([t for t, _, _ in steps])
    check("step interval is the frame interval (~32 ms), never ~0", dts.min() > 0.025, f"min {dts.min() * 1000:.1f} ms")

    print("recorder stores a repeated frame once")
    rec2 = Recorder(out, chunk_s=1e9, tag="dedupe")
    for k in range(10):
        rec2.add(pkt(0, k), np.zeros(6), 200.0 + k / 30)
        rec2.add(pkt(0, k), np.zeros(6), 200.0 + k / 30 + 0.016)
    rec2.close()
    d2 = np.load(rec2.files[0])
    check("10 frames offered twice -> 10 stored", len(d2["seq_0"]) == 10, str(len(d2["seq_0"])))
finally:
    shutil.rmtree(out, ignore_errors=True)

print(f"test_pacing: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
