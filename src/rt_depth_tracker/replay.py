# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Replay recordings through the identical Pipeline (any OS). Prints a per-second summary; can emit
/tracker/target over OSC so a consumer (or a mock) can be driven from a recording.

  python -m rt_depth_tracker.replay recordings/rec_*.npz --calib-dir config/calib [--config config/tracker.json]
                          [--realtime] [--emit 127.0.0.1:7001] [--csv out.csv] [--loop [N]]

--loop plays the file list again when it ends (N passes, or forever without N) with a continuous
clock, so a playlist of short clips behaves like a live scene.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import socket
import sys
import time

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker import protocol as P
from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.pipeline import Pipeline
from rt_depth_tracker.record import Reader


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("--calib-dir", default=None, help="override the calibration stored in the recording")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--emit", default=None, help="host:port to send /tracker/target")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--learn-bg", type=int, default=0, help="learn background from the first N frames (else use calib-dir bg)")
    ap.add_argument("--q-mode", choices=["real", "frozen", "zeros", "offset"], default="real",
                    help="what robot pose to mask with: the recorded measured joints, the first frame held "
                         "(feed died), all zeros (feed never arrived), or the truth off by --q-offset-deg")
    ap.add_argument("--q-offset-deg", type=float, default=0.0, help="--q-mode offset: degrees added to every joint")
    ap.add_argument("--q-age", type=float, default=None, help="age in seconds to report with each pose (default: fresh)")
    ap.add_argument("--loop", nargs="?", const=0, type=int, default=None, metavar="N", help="repeat the playlist N times (no N = forever)")
    ap.add_argument("--envelope-r", type=float, default=None, help="robot: radius around the base inside which an unmasked point is a phantom (default 0.95)")
    return ap.parse_args(argv)


ENVELOPE_R = 0.95        # the arm's reach plus slack: nothing but the robot can be this close to its base (--envelope-r)


def q_for(q, q_first, a):
    """The pose to mask with, per --q-mode. 'real' is the recording's own measured joints."""
    if a.q_mode == "frozen":
        return q_first
    if a.q_mode == "zeros":
        return np.zeros(6)
    if a.q_mode == "offset":
        return np.asarray(q, np.float64) + np.radians(a.q_offset_deg)
    return q


def stream(rd, passes: int | None):
    """Reader events with a clock that keeps running across passes (None = one pass, 0 = forever)."""
    shift, last, n = 0.0, None, 0
    while True:
        t0 = None
        for t, packets, q in rd:
            if t0 is None:
                t0 = t
                if last is not None:
                    shift = last - t + 1 / 30
            last = t + shift
            yield last, packets, q
        n += 1
        if passes is None or t0 is None or (passes and n >= passes):
            return


def run(a) -> dict:
    files = sorted(sum((glob.glob(f) for f in a.files), []))
    if not files:
        raise SystemExit("no files")
    rd = Reader(files)
    cfg = TrackerConfig.load(a.config)
    pipe = Pipeline(cfg)
    for cid, k in rd.intrinsics.items():
        pipe.set_intrinsics(k)
    for cid, T in rd.calib.items():
        pipe.set_calib(cid, T)
    if a.calib_dir:
        for f in C.sensor_calib_files(a.calib_dir):
            T, meta = C.load_sensor_calib(f)
            pipe.set_calib(int(meta["cam_id"]), T)
        for f in sorted(glob.glob(os.path.join(a.calib_dir, "bg_*.npz"))):
            pipe.cam(int(os.path.basename(f)[3:-4])).bg.load(f)
    if a.learn_bg:
        pipe.learn_background(frames=a.learn_bg)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if a.emit else None
    emit_addr = (a.emit.split(":")[0], int(a.emit.split(":")[1])) if a.emit else None
    writer = None
    if a.csv:
        fh = open(a.csv, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["t", "hint", "track_id", "score", "x", "y", "z", "n_tracks", "n_points", "step_ms"])

    t_first = None
    wall0 = time.monotonic()
    last_print = None
    ms = []
    hints = {}
    t_prev, monotonic = None, True
    q_first = None
    robot = pipe.robot is not None
    base_xyz = np.asarray(pipe.robot.T_world_base[:3, 3] if robot else [cfg.base_xy[0], cfg.base_xy[1], cfg.mount_z], np.float64)
    env_r = a.envelope_r if a.envelope_r is not None else ENVELOPE_R
    phantom_pts, phantom_frames, phantom_ids = 0, 0, set()
    for t, packets, q in stream(rd, a.loop):
        if t_first is None:
            t_first = t
        monotonic &= t_prev is None or t >= t_prev      # both cameras may share a step time
        t_prev = t
        if a.realtime:
            lag = (t - t_first) - (time.monotonic() - wall0)
            if lag > 0:
                time.sleep(lag)
        if q_first is None:
            q_first = np.asarray(q, np.float64).copy()
        pipe.set_q(q_for(q, q_first, a), 0.0 if a.q_age is None else a.q_age)
        dec, dbg = pipe.step(packets, t)
        ms.append(dbg.step_ms)
        hints[dec.hint] = hints.get(dec.hint, 0) + 1
        if robot:
            # anything the mask failed to remove inside the arm's own reach: a phantom visitor made of robot
            n_in = 0
            if len(dbg.points):
                d = np.asarray(dbg.points, np.float64)[:, :3] - base_xyz
                n_in = int((np.einsum("ij,ij->i", d, d) < env_r ** 2).sum())
            phantom_pts = max(phantom_pts, n_in)
            for tk in dbg.tracks:
                dxy = np.asarray(tk.reach_pt, np.float64) - base_xyz
                if float(dxy @ dxy) < env_r ** 2:
                    phantom_frames += 1
                    phantom_ids.add(int(tk.id))
                    break
        if tx is not None:
            tx.sendto(P.pack_target(P.Target(*dec.target, dec.hint, dec.track_id, dec.score)), emit_addr)
        if writer is not None:
            writer.writerow([round(t - t_first, 3), dec.hint, dec.track_id, round(dec.score, 3), *np.round(dec.target, 3),
                             len(dbg.tracks), len(dbg.points), round(dbg.step_ms, 2)])
        if last_print is None or t - last_print >= 1.0:
            last_print = t
            print(f"t={t - t_first:7.1f}s hint={dec.hint:6s} id={dec.track_id:3d} S={dec.score:.2f} tgt={np.round(dec.target, 2)} "
                  f"tracks={len(dbg.tracks)} pts={len(dbg.points)} ms={dbg.step_ms:.1f}")
    if writer is not None:
        fh.close()
    return {"frames": len(ms), "p50_ms": float(np.percentile(ms, 50)) if ms else None,
            "p95_ms": float(np.percentile(ms, 95)) if ms else None, "hints": hints,
            "t_span": (t_prev - t_first) if t_prev is not None else 0.0, "monotonic": monotonic, "robot": robot,
            "phantom_pts": phantom_pts, "phantom_frames": phantom_frames, "phantom_ids": sorted(phantom_ids)}


def main(argv=None):
    st = run(parse(argv))
    print(f"\nframes={st['frames']} step p50={st['p50_ms']:.1f} p95={st['p95_ms']:.1f} ms  span={st['t_span']:.1f}s  hints={st['hints']}")
    if st["robot"]:
        print(f"inside the robot envelope: worst {st['phantom_pts']} unmasked points, "
              f"{st['phantom_frames']}/{st['frames']} frames with a track there, ids {st['phantom_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
