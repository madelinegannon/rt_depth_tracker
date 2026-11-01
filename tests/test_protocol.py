# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Round-trip tests for rt_depth_tracker.protocol. Plain Python, no hardware."""
import os
import sys

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import protocol as P  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")
    if not cond:
        _fail += 1


def main():
    print("test_protocol")
    rng = np.random.default_rng(0)
    full = rng.integers(0, 6000, size=(288, 320), dtype=np.uint16)

    ds = P.pack_depth(3, 42, full, 2, 12.5)
    check("depth chunked into 8 KB datagrams", len(ds) == 6 and all(len(d) <= 8200 for d in ds), f"{len(ds)} x {max(map(len, ds))} B")
    check("chunk parses", isinstance(P.unpack(ds[0]), P.DepthChunk) and P.unpack(ds[0]).n == 6)
    pk = P.unpack_depth_all(ds)
    check("depth type", isinstance(pk, P.DepthPacket))
    assembler = P.DepthAssembler()
    check("incomplete -> None", all(assembler.feed(P.unpack(d)) is None for d in ds[:-1]))
    check("newer seq discards older partial", assembler.feed(P.unpack(P.pack_depth(3, 43, full, 2, 13.0)[0])) is None and assembler._parts[3][0] == 43)
    check("late chunk of old seq ignored", assembler.feed(P.unpack(ds[-1])) is None)
    # a restarted capture counts seq from 1 again; a half-frame the old one left behind must not block it for
    # as long as the old process had been running
    clk = [100.0]
    assembler2 = P.DepthAssembler(clock=lambda: clk[0])
    for d in ds[:-1]:
        assembler2.feed(P.unpack(d))                                   # seq 42 pending, incomplete
    low = P.pack_depth(3, 1, full, 2, 14.0)
    clk[0] += 0.5
    check("a lower seq within 1 s of a pending frame is a late chunk: dropped", assembler2.feed(P.unpack(low[0])) is None and assembler2._parts[3][0] == 42)
    clk[0] += 1.0
    pk3 = None
    for d in low:
        pk3 = assembler2.feed(P.unpack(d)) or pk3
    check("a lower seq once the pending frame is 1 s old is a restarted capture: accepted and completes",
          isinstance(pk3, P.DepthPacket) and pk3.seq == 1 and 3 not in assembler2._parts, str(pk3 and pk3.seq))
    check("single datagram when small", len(P.pack_depth(0, 1, full[:32, :32], 1, 0.0)) == 1)
    pk2 = P.unpack_depth_all(P.pack_depth_packet(pk))
    check("pack_depth_packet keeps geometry", pk2.depth.shape == (144, 160) and np.array_equal(pk2.depth, pk.depth) and pk2.width == 320)
    check("depth header", (pk.cam_id, pk.seq, pk.width, pk.height, pk.stride) == (3, 42, 320, 288, 2))
    check("depth ts", abs(pk.device_ts - 12.5) < 1e-9)
    check("depth shape", pk.depth.shape == (144, 160))
    check("depth values", np.array_equal(pk.depth, full[::2, ::2]))

    big = P.pack_depth(0, 0, np.zeros((576, 640), np.uint16), 1, 0.0)
    check("full-res unbinned frame = 93 chunks", len(big) == -(-576 * 640 * 2 // P.CHUNK_BYTES))

    k = P.Intrinsics(1, 320, 288, 250.1, 250.2, 160.3, 144.4, 0.1, -0.2, 0.03, 0, 0, 0, 0.001, -0.002)
    k2 = P.unpack(P.pack_intrinsics(k))
    check("intrinsics roundtrip", isinstance(k2, P.Intrinsics) and abs(k2.fx - 250.1) < 1e-4
          and abs(k2.p2 + 0.002) < 1e-6 and k2.cam_id == 1)
    ks = k.scaled(2)
    check("intrinsics scaled", ks.width == 160 and abs(ks.fx - 125.05) < 1e-6 and abs(ks.cx - 80.15) < 1e-6)

    hb = P.unpack(P.pack_hb("capture_a", {"fps": 29.9, "valid_px": 1234}))
    check("hb roundtrip", isinstance(hb, P.Heartbeat) and hb.name == "capture_a" and hb.metrics["valid_px"] == 1234)

    t = P.unpack(P.pack_target(P.Target(0.5, 1.4, 1.6, "engage", 7, 0.83)))
    check("target roundtrip", isinstance(t, P.Target) and t.hint == "engage" and t.track_id == 7
          and abs(t.y - 1.4) < 1e-6)
    t = P.unpack(P.pack_target(P.Target(1.8, 0.0, 1.6, "sleep", -1, 0.0)))
    check("sleep hint roundtrip", isinstance(t, P.Target) and t.hint == "sleep" and t.track_id == -1)
    try:
        P.pack_target(P.Target(0, 0, 0, "bogus", 0, 0))
        check("bad hint raises", False)
    except ValueError:
        check("bad hint raises", True)

    q = P.unpack(P.pack_robot_q([0, 0.26, -0.9, 0, 0.6, 0]))
    check("robot q roundtrip", isinstance(q, P.RobotQ) and abs(q.q[2] + 0.9) < 1e-6)

    rows = np.arange(16, dtype=np.float32).reshape(2, 8)
    r2 = P.unpack(P.pack_tracks(rows))
    check("tracks roundtrip", isinstance(r2, np.ndarray) and np.array_equal(r2, rows))
    r0 = P.unpack(P.pack_tracks(np.zeros((0, 8), np.float32)))
    check("empty tracks message", isinstance(r0, np.ndarray) and r0.shape == (0, 8))

    ctrl = P.unpack(P.pack_hb.__globals__["_msg"](P.ADDR_LEARN_BG))
    check("control passthrough", ctrl == (P.ADDR_LEARN_BG, []))
    check("garbage -> None", P.unpack(b"not osc") is None)

    print(f"test_protocol: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
