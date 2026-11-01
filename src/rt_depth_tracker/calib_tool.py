# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Sensor calibration tool (sensor -> WORLD; world = Z-up, metres, origin on the floor).

Three ways to place a camera, each chained on whatever calibration is on disk (docs/calibration.md):
  floor + mark   seed -> learn-bg -> plane -> yaw       a tape measure, a floor mark, a person standing on it
  table          seed -> learn-bg -> plane -> table     a rectangle of known size and height in view (overhead rigs)
  sphere         seed -> learn-bg -> sphere             a ball on a robot flange; needs a robot in the config and a
                 [-> refine, validate]                  /robot/q joint feed. The tool never commands the arm: you move it.

Commands (seed, plane, table, nudge, yaw, aim, undo, validate read files; learn-bg, refine and sphere --no-rgb read /depth on --in):
  seed      --cam 0 [--pos x,y,z --aim x,y,z]                look-at seed; default = cams[cam] in tracker.json
  learn-bg  [--seconds 10]                                   empty scene (robot parked): learn + save the backgrounds
  plane     --cams 0,1 [--clip clip.npz] [--z-band lo,hi]    roll/pitch/height from the floor in the background
  aim       --cams 0,1 [--height 3.0] [--write]              pan/tilt per camera so both cover the zone; floor point to aim at
  yaw       --cams 0,1 --clip clip.npz --ref 1.6,0 [--dry-run] yaw from a person standing still on a measured mark
  table     --cams 0 [--clip clip.npz] [--band 0.06]         yaw + xy from the table top (table.size_xy / top_z in tracker.json)
  nudge     --cams 0 --yaw DEG [--dx M --dy M]               manual world-frame correction on top of the current file
  sphere    --cams 0,1 --poses 12 --cam-ips a,b              ball on the flange: you move the arm to each pose, the tool
            [--ball-r 0.075 --offset 0,0,0.180] [--no-rgb]   reads /robot/q, finds the ball (colour; depth with --no-rgb),
                                                             fits it, solves the pose by Kabsch. --append adds to the last sweep
  probe-color --cam 0 --cam-ips a,b                          is the ball detectable? blob stats + mask PNG
  refine    --cam 0                                          ICP of the robot's own blob vs its sphere model (needs /robot/q)
  validate  [--q-port 7015]                                  leave-one-out on the last sweep + robot-mask leftover (needs /robot/q)
  undo      --cams 0,1                                       put back the calibration files from before the last step
Each step backs up the previous file to sensor_<x>.prev.json and tells a running tracker to reload.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time

import numpy as np
from pythonosc.osc_message_builder import OscMessageBuilder
from scipy.spatial import cKDTree

from rt_depth_tracker import calib as C
from rt_depth_tracker import protocol as P
from rt_depth_tracker import rgb as RGB
from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.intrinsics import unproject
from rt_depth_tracker.pipeline import Pipeline
from rt_depth_tracker.record import Reader
from rt_depth_tracker.run import DepthRx, RobotRx
from rt_depth_tracker.segment import crop_world


def osc(address, *args):
    b = OscMessageBuilder(address=address)
    for v in args:
        if isinstance(v, int):
            b.add_arg(v, OscMessageBuilder.ARG_TYPE_INT)
        elif isinstance(v, float):
            b.add_arg(v, OscMessageBuilder.ARG_TYPE_FLOAT)
        else:
            b.add_arg(str(v), OscMessageBuilder.ARG_TYPE_STRING)
    return b.build().dgram


class Session:
    def __init__(self, a):
        self.a = a
        self.cfg = TrackerConfig.load(a.config)
        self.pipe = Pipeline(self.cfg)
        self.robot = self.pipe.robot          # None without a robot in the config: sphere / refine / validate refuse
        self._rx = self._qrx = None        # bound lazily: file-only commands never touch the live ports
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        os.makedirs(a.calib_dir, exist_ok=True)
        for f in C.sensor_calib_files(a.calib_dir):
            T, meta = C.load_sensor_calib(f)
            self.pipe.set_calib(int(meta["cam_id"]), T)
        for f in sorted(glob.glob(os.path.join(a.calib_dir, "bg_*.npz"))):
            self.pipe.cam(int(os.path.basename(f)[3:-4])).bg.load(f)
        for f in sorted(glob.glob(os.path.join(a.calib_dir, "intrinsics_*.json"))):   # written by the running tracker
            with open(f, "r", encoding="utf-8") as fh:
                self.pipe.set_intrinsics(P.Intrinsics(**json.load(fh)))

    @property
    def rx(self):
        if self._rx is None:
            self._rx = DepthRx(self.a.in_port)
        return self._rx

    @property
    def qrx(self):
        if self._qrx is None:
            self._qrx = RobotRx(self.a.q_port)
        return self._qrx

    def stop(self):
        for r in (self._rx, self._qrx):
            if r is not None:
                r.stop()

    def _need_robot(self, what: str):
        if self.robot is None:
            raise SystemExit(f"{what} needs a robot: set robot.urdf and robot.robot_description in the config, "
                             "and publish its measured joint angles as /robot/q")

    def _cams(self) -> list[int]:
        return [int(c) for c in self.a.cams.split(",")]

    def _prev_method(self, cam) -> str:
        try:
            return C.load_sensor_calib(self.calib_path(cam))[1].get("meta", {}).get("method", "?")
        except (OSError, KeyError, ValueError):
            return "?"

    def _clip(self) -> dict[int, list]:
        """{cam: [(t, DepthPacket), ...]} from --clip files; the clip's intrinsics feed the pipeline."""
        paths = sorted(set(sum((glob.glob(p) for p in (self.a.clip or [])), [])))
        if not paths:
            raise SystemExit("--clip recordings/<tag>_*.npz required (record one with: python -m rt_depth_tracker.ctl record <tag>)")
        rd = Reader(paths)
        for k in rd.intrinsics.values():
            self.pipe.set_intrinsics(k)
        out: dict[int, list] = {}
        last_seq: dict[int, int] = {}
        for t, pk, _q in rd:
            for c, p in pk.items():
                if last_seq.get(c) != p.seq:
                    last_seq[c] = p.seq
                    out.setdefault(c, []).append((t, p))
        return out

    def _bg_cloud(self, c: int, frames: dict):
        """Sensor-frame points of the learned background (or a clip's median frame) for camera c."""
        cs = self.pipe.cam(c)
        if cs.T_ws is None:
            raise SystemExit(f"cam {c}: seed first")
        if cs.bg.ready:
            depth, stride = cs.bg.bg, self.cfg.stride
        elif frames.get(c):
            st = np.stack([p.depth for _, p in frames[c]]).astype(np.float32)
            depth, stride = np.nanmedian(np.where(st > 0, st, np.nan), axis=0), frames[c][0][1].stride
        else:
            raise SystemExit(f"cam {c}: learn-bg first, or pass --clip")
        if cs.intr is None:
            raise SystemExit(f"cam {c}: no intrinsics; start the tracker once (it writes intrinsics_*.json) or pass --clip")
        rays = self.pipe._rays_for(cs, stride)
        P_s, _ = unproject(np.nan_to_num(depth, nan=0.0).astype(np.float32), rays)
        return cs, P_s

    def _commit(self, c: int, T_new, method_suffix: str, rep: dict) -> dict:
        self._backup(self.calib_path(c))
        sha = C.save_sensor_calib(self.calib_path(c), c, T_new, {"method": self._prev_method(c) + method_suffix, **rep})
        self.pipe.set_calib(c, T_new)
        return {**rep, "sha": sha}

    def _reload(self):
        self.tx.sendto(osc(P.ADDR_RELOAD_CALIB), ("127.0.0.1", self.a.ctrl_port))

    def _backup(self, path: str):
        """Keep the previous calibration file as <name>.prev.json so `undo` can put it back."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                prev = f.read()
            with open(path[:-5] + ".prev.json", "w", encoding="utf-8") as f:
                f.write(prev)

    def calib_path(self, cam):
        return os.path.join(self.a.calib_dir, f"sensor_{'abcdefgh'[cam]}.json")

    def wait_frames(self, seconds: float, cams: list[int] | None = None) -> dict:
        """Collect {cam: [depth,...]} for `seconds`, also absorbing intrinsics."""
        out: dict[int, list] = {}
        seq_seen: dict[int, int] = {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self.rx.event.wait(0.2)
            self.rx.event.clear()
            pk, intr = self.rx.take(0.5)
            for k in intr.values():
                self.pipe.set_intrinsics(k)
            for c, p in pk.items():
                if cams is not None and c not in cams:
                    continue
                if seq_seen.get(c) != p.seq:
                    seq_seen[c] = p.seq
                    out.setdefault(c, []).append(p)
        return out

    # ---- file-only steps -------------------------------------------------------------------------------
    def seed(self):
        pos, aim = self.cfg.cam_pose(self.a.cam)
        if self.a.pos:
            pos = [float(v) for v in self.a.pos.split(",")]
        if self.a.aim:
            aim = [float(v) for v in self.a.aim.split(",")]
        T = C.look_at_T(pos, aim)
        sha = C.save_sensor_calib(self.calib_path(self.a.cam), self.a.cam, T, {"method": "seed", "pos": pos, "aim": aim})
        self.pipe.set_calib(self.a.cam, T)
        print(json.dumps({"cmd": "seed", "cam": self.a.cam, "file": self.calib_path(self.a.cam), "sha": sha}))
        self._reload()

    def learn_bg(self):
        print(f"collecting {self.a.seconds}s of empty-scene frames...")
        frames = self.wait_frames(self.a.seconds)
        if not frames:
            raise SystemExit(f"no /depth frames on port {self.a.in_port}: is a capture running with --out ...:{self.a.in_port}?")
        rep = {}
        for c, lst in frames.items():
            cs = self.pipe.cam(c)
            cs.bg.learn(np.stack([p.depth for p in lst]))
            path = os.path.join(self.a.calib_dir, f"bg_{c}.npz")
            cs.bg.save(path)
            rep[c] = {"frames": len(lst), "invalid_frac": round(float(cs.bg.invalid.mean()), 3), "file": path}
        print(json.dumps({"cmd": "learn-bg", "cams": rep}))
        self._reload()

    def plane(self):
        """Roll/pitch/height per camera from the floor: uses the learned background (or a clip's median frame)."""
        frames = self._clip() if self.a.clip else {}
        z_band = tuple(float(v) for v in self.a.z_band.split(","))
        rep = {"cmd": "plane", "cams": {}}
        for c in self._cams():
            cs, P_s = self._bg_cloud(c, frames)
            T_new, r = C.floor_correction(C.apply(cs.T_ws, P_s), cs.T_ws, z_band=z_band)
            if T_new is None:
                rep["cams"][c] = {**r, "applied": False, "why": "too few floor points: check the seed pose and that the floor is in view"}
                continue
            tilt, height = C.floor_residual(C.apply(T_new, P_s))
            r.update(tilt_after_deg=round(tilt, 2), height_after_m=round(height, 3), applied=True)
            rep["cams"][c] = self._commit(c, T_new, "+plane", r)
        self._reload()
        print(json.dumps(rep, indent=1))
        return all(v.get("applied") for v in rep["cams"].values())

    def table(self):
        """Yaw + xy per camera from the table top (tracker.json table.size_xy / top_z) in the background cloud."""
        tb = self.cfg.table
        if not tb or "size_xy" not in tb or "top_z" not in tb:
            raise SystemExit('table needs "table": {"size_xy": [long, short], "top_z": height} in the config')
        frames = self._clip() if self.a.clip else {}
        rep = {"cmd": "table", "size_xy": tb["size_xy"], "top_z": tb["top_z"], "cams": {}}
        for c in self._cams():
            cs, P_s = self._bg_cloud(c, frames)
            T_new, r = C.table_correction(C.apply(cs.T_ws, P_s), cs.T_ws, float(tb["top_z"]), tb["size_xy"], band=self.a.band)
            if T_new is None:
                rep["cams"][c] = r
                continue
            rep["cams"][c] = self._commit(c, T_new, "+table", r)
        self._reload()
        print(json.dumps(rep, indent=1))
        return all(v.get("applied") for v in rep["cams"].values())

    def nudge(self):
        """Manual world-frame correction: rotate about the origin by --yaw, then shift by --dx/--dy (the viewer's sliders do the same)."""
        rep = {"cmd": "nudge", "yaw_deg": self.a.yaw, "dx": self.a.dx, "dy": self.a.dy, "cams": {}}
        for c in self._cams():
            cs = self.pipe.cam(c)
            if cs.T_ws is None:
                raise SystemExit(f"cam {c}: seed first")
            rep["cams"][c] = self._commit(c, C.yaw_T(self.a.yaw, self.a.dx, self.a.dy) @ cs.T_ws, "+nudge", {"applied": True})
        self._reload()
        print(json.dumps(rep, indent=1))
        return True

    def undo(self):
        """Restore each camera's previous calibration file (written before the last step)."""
        rep = {"cmd": "undo", "cams": {}}
        for c in self._cams():
            path = self.calib_path(c)
            prev = path[:-5] + ".prev.json"
            if not os.path.exists(prev):
                rep["cams"][c] = {"restored": False, "why": "no .prev.json"}
                continue
            with open(prev, "r", encoding="utf-8") as f:
                data = f.read()
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            os.remove(prev)
            T, meta = C.load_sensor_calib(path)
            self.pipe.set_calib(c, T)
            rep["cams"][c] = {"restored": True, "method": meta.get("meta", {}).get("method"), "sha": meta.get("sha")}
        self._reload()
        print(json.dumps(rep, indent=1))
        return all(v.get("restored") for v in rep["cams"].values())

    def yaw(self):
        """Yaw per camera from a person (or pole) standing still on the measured mark --ref, recorded in a clip.
        Candidate runs must match the camera->mark range within --max-range-err; when both cameras have
        candidates, the pair that overlaps most in time is the mark (a bystander seen by one camera at a
        similar range cannot win). --dry-run only reports."""
        cams = self._cams()
        ref = [float(v) for v in self.a.ref.split(",")]
        frames = self._clip()
        rep = {"cmd": "yaw", "ref": ref, "dry_run": bool(self.a.dry_run), "cams": {}}
        bx, by = self.cfg.base_xy
        per: dict[int, dict] = {}
        for c in cams:
            cs = self.pipe.cam(c)
            if cs.T_ws is None or cs.intr is None:
                raise SystemExit(f"cam {c}: seed first, and the clip must carry intrinsics")
            if not cs.bg.ready:
                raise SystemExit(f"cam {c}: learn-bg first (the person must stand out from the background)")
            if not frames.get(c):
                raise SystemExit(f"cam {c}: no frames for this camera in the clip")
            stride = frames[c][0][1].stride
            rays = self.pipe._rays_for(cs, stride)
            cents, times = [], []
            for t, p in frames[c]:
                times.append(t)
                P_s, _ = unproject(p.depth, rays, cs.bg.foreground(p.depth))
                if len(P_s) < 150:
                    cents.append(None)
                    continue
                P_w = C.apply(cs.T_ws, P_s)
                keep = crop_world(P_w, self.cfg) & (np.hypot(P_w[:, 0] - bx, P_w[:, 1] - by) > 0.6) & (P_w[:, 2] > 0.25) & (P_w[:, 2] < 2.0)
                if keep.sum() < 150:
                    cents.append(None)
                    continue
                Q = P_w[keep]
                cents.append((float(Q[:, 0].mean()), float(Q[:, 1].mean()), float(np.percentile(Q[:, 2], 97))))
            cands = []
            for i0, i1, xy in C.stable_runs(cents, times, self.a.min_run):
                _, r = C.yaw_correction(xy, cs.T_ws, ref)
                top = float(np.median([cents[i][2] for i in range(i0, i1 + 1) if cents[i] is not None]))
                cands.append({"t0": times[i0], "t1": times[i1], "xy": xy, "rep": {**r, "top_z_m": round(top, 2)}, "err": abs(r["range_ratio"] - 1.0)})
            per[c] = {"cs": cs, "cands": cands, "t_base": times[0] if times else 0.0}
        within = {c: [k for k in per[c]["cands"] if k["err"] <= self.a.max_range_err] for c in cams}
        chosen: dict[int, dict] = {}
        if len(cams) >= 2 and within[cams[0]] and within[cams[1]]:
            a, b = cams[0], cams[1]
            best, best_ov = None, 0.0
            for ka in within[a]:
                for kb in within[b]:
                    ov = min(ka["t1"], kb["t1"]) - max(ka["t0"], kb["t0"])
                    if ov > best_ov:
                        best_ov, best = ov, (ka, kb)
            if best is not None:
                chosen = {a: best[0], b: best[1]}
        for c in cams:
            if c not in chosen and within[c]:
                chosen[c] = min(within[c], key=lambda k: k["err"])
        ok = True
        for c in cams:
            tb = per[c]["t_base"]
            info = {"runs": [{"t_s": [round(k["t0"] - tb, 1), round(k["t1"] - tb, 1)], "seen": [round(float(k["xy"][0]), 2), round(float(k["xy"][1]), 2)], **k["rep"]}
                             for k in per[c]["cands"]]}
            k = chosen.get(c)
            if k is None:
                why = (f"no run of a still person >= {self.a.min_run} s" if not per[c]["cands"] else
                       f"camera->person range off by more than {self.a.max_range_err * 100:.0f} % on every run: wrong --ref, the person was not on the mark, "
                       "or the camera is not where the config says (a camera on the wrong side shows ~25-30 % on a lateral mark while an on-axis mark is fine)")
                rep["cams"][c] = {**info, "applied": False, "why": why}
                ok = False
                continue
            cs = per[c]["cs"]
            T_new, r = C.yaw_correction(k["xy"], cs.T_ws, ref)
            entry = {**info, "chosen_run": {"t_s": [round(k["t0"] - tb, 1), round(k["t1"] - tb, 1)], "seen": [round(float(k["xy"][0]), 2), round(float(k["xy"][1]), 2)]},
                     "yaw_deg": r["yaw_deg"], "range_ratio": r["range_ratio"], "top_z_m": k["rep"]["top_z_m"], "applied": not self.a.dry_run}
            if not self.a.dry_run:
                entry = self._commit(c, T_new, "+yaw", {**entry, "ref": ref})
            rep["cams"][c] = entry
        if not self.a.dry_run:
            self._reload()
        print(json.dumps(rep, indent=1))
        return ok

    # ---- aiming ----------------------------------------------------------------------------------------
    def aim(self):
        """Pan/tilt per camera so both cover the zone; prints the floor point F each lens must point at."""
        cams = self._cams()
        intr = next((self.pipe.cam(c).intr for c in cams if self.pipe.cam(c).intr is not None), None)
        if intr is None and self.a.clip:
            self._clip()
            intr = next((self.pipe.cam(c).intr for c in cams if self.pipe.cam(c).intr is not None), None)
        note = "intrinsics from the clip"
        if intr is None:
            from rt_depth_tracker.synth import FEMTO_NFOV_BINNED as intr
            note = "nominal 320x288 ToF intrinsics (pass --clip for the real ones)"
        x0, x1, yh = [float(v) for v in self.a.zone.split(",")]
        pos = {}
        for c in cams:
            p, _ = self.cfg.cam_pose(c)
            if self.a.height is not None:
                p[2] = float(self.a.height)
            pos[c] = np.array(p, float)
        res, both = C.aim_search(pos, intr, self.cfg.stride, C.zone_samples(x0, x1, yh))
        rep = {"cmd": "aim", "note": note, "zone": {"x": [x0, x1], "y_half": yh, "heights_m": [0.2, 1.0]},
               "both_covered_frac": round(both, 3), "cams": {}}
        for c in cams:
            r = res[c]
            fl = r["floor_point"]
            rep["cams"][c] = {"pos": pos[c].round(3).tolist(), "pan_deg_from_x_toward_y": r["pan_deg"],
                              "tilt_deg_below_horizontal": r["tilt_deg"], "own_covered_frac": r["own_frac"],
                              "floor_point_F": None if fl is None else fl.tolist()}
        if self.a.write:
            paths = [p for p in TrackerConfig.paths(self.a.config) if os.path.exists(p)]
            target = paths[-1]
            with open(target, "r", encoding="utf-8") as f:
                d = json.load(f)
            d.setdefault("cams", {})
            for c in cams:
                fl = res[c]["floor_point"]
                if fl is not None:
                    d["cams"].setdefault(str(c), {})["aim"] = [round(float(v), 3) for v in fl]
                    d["cams"][str(c)].setdefault("pos", pos[c].round(3).tolist())
            with open(target, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
                f.write("\n")
            rep["written"] = target
        print(json.dumps(rep, indent=1))
        return True

    # ---- the ball on the flange ----------------------------------------------------------------------
    def _ball_points(self, cam: int, frames: list, q, predicted, stem_from=None):
        """Foreground points (sensor frame, world if calibrated) near the predicted ball position.
        stem_from: world point of the flange face; points within stem_r of the flange->ball segment are
        dropped so the stem never leaks into the sphere fit (black filament helps, geometry guarantees it)."""
        cs = self.pipe.cam(cam)
        if cs.intr is None:
            raise RuntimeError(f"cam {cam}: no intrinsics yet")
        rays = self.pipe._rays_for(cs, frames[0].stride)
        pts = []
        for p in frames:
            fg = cs.bg.foreground(p.depth) if cs.bg.ready else (p.depth > 0)
            P_s, _ = unproject(p.depth, rays, fg)
            pts.append(P_s)
        P_s = np.vstack(pts)
        if cs.T_ws is None:
            return P_s, None
        P_w = C.apply(cs.T_ws, P_s)
        keep = np.linalg.norm(P_w - predicted, axis=1) < self.a.search_r
        rob = self.robot.inside_mask(P_w[keep], q, margin=0.05, pedestal_radius=0.0)
        if stem_from is not None:
            a, b = np.asarray(stem_from, float), np.asarray(predicted, float)
            ab = b - a
            t = np.clip(((P_w[keep] - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)
            d = np.linalg.norm(P_w[keep] - (a + t[:, None] * ab), axis=1)
            seg_len = np.linalg.norm(ab)
            near_stem = (d < self.a.stem_r) & (t * seg_len < seg_len - self.a.ball_r)   # stem only, not the ball itself
            rob = rob | near_stem
        return P_s[keep][~rob], P_w[keep][~rob]

    def _cam_ip(self, cam: int) -> str:
        if not self.a.cam_ips:
            raise SystemExit("--cam-ips <ip0>,<ip1> is required to open the cameras for colour (or use --no-rgb with the /depth stream)")
        ips = [v.strip() for v in self.a.cam_ips.split(",")]
        return ips[cam] if cam < len(ips) else ips[-1]

    def _open_rgb(self, cams: list[int]) -> dict:
        out = {}
        for c in cams:
            print(f"opening cam {c} at {self._cam_ip(c)} (depth + colour aligned to depth)...")
            cam = RGB.RgbDepthCam(self._cam_ip(c), stride=self.cfg.stride)
            cam.intr.cam_id = c
            self.pipe.set_intrinsics(cam.intr)
            out[c] = cam
        return out

    def _hsv(self):
        lo = tuple(int(v) for v in self.a.hsv_lo.split(","))
        hi = tuple(int(v) for v in self.a.hsv_hi.split(","))
        return lo, hi

    def _expected_in_sensor(self, cam: int, ball_w):
        """Where the current calibration puts the ball in this camera's frame; None without one or with --near 0."""
        T = self.pipe.cam(cam).T_ws
        if T is None or self.a.near <= 0:
            return None
        return C.apply(C.invert(T), np.asarray(ball_w, float)[None])[0]

    def _rgb_ball(self, cam, n_frames: int, radius: float, near=None):
        """Median ball centre (sensor frame) over n frames via the orange blob; None if not seen.
        near: expected ball centre (sensor frame); only blobs within --near of it count."""
        lo, hi = self._hsv()
        cs, areas, rmss, nins = [], [], [], []
        for depth, bgr, _ in cam.read_n(n_frames):
            c, rms, n_in, blob = RGB.fit_ball(depth, bgr, cam.intr, cam.stride, radius, lo, hi, self.a.min_px, rays=cam.rays,
                                              near=near, near_r=self.a.near)
            if c is not None:
                cs.append(c)
                areas.append(blob.area)
                rmss.append(rms)
                nins.append(n_in)
        if len(cs) < max(3, n_frames // 4):
            return None, {"frames_with_ball": len(cs)}
        cs = np.array(cs)
        return np.median(cs, 0), {"frames_with_ball": len(cs), "area_px": int(np.median(areas)), "rms_mm": round(float(np.median(rmss)) * 1000, 1),
                                  "inliers": int(np.median(nins)), "spread_mm": round(float(np.std(cs, 0).max()) * 1000, 1)}

    def probe_color(self):
        cam = self._open_rgb([self.a.cam])[self.a.cam]
        try:
            frames = cam.read_n(30)
            if not frames:
                print(json.dumps({"cmd": "probe-color", "ok": False, "reason": "no frames"}))
                return False
            depth, bgr, _ = frames[-1]
            lo, hi = self._hsv()
            blob = RGB.find_blob(bgr, lo, hi, self.a.min_px)
            c, rms, n_in, _ = RGB.fit_ball(depth, bgr, cam.intr, cam.stride, self.a.ball_r, lo, hi, self.a.min_px, rays=cam.rays)
            import cv2
            vis = bgr.copy()
            if blob is not None:
                vis[blob.mask] = (0.5 * vis[blob.mask] + (0, 255, 0)).clip(0, 255).astype(np.uint8)
                x0, y0, x1, y1 = blob.bbox
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 1)
            png = os.path.join(self.a.calib_dir, f"probe_color_{self.a.cam}.png")
            cv2.imwrite(png, cv2.resize(vis, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            cs = self.pipe.cam(self.a.cam)
            rep = {"cmd": "probe-color", "cam": self.a.cam, "ok": c is not None, "blob_area_px": blob.area if blob else 0,
                   "centroid_px": blob.centroid if blob else None, "centre_sensor_m": (None if c is None else np.round(c, 4).tolist()),
                   "centre_world_m": (None if c is None or cs.T_ws is None else np.round(C.apply(cs.T_ws, c[None])[0], 4).tolist()),
                   "rms_mm": (None if c is None else round(rms * 1000, 1)), "inliers": n_in, "png": png,
                   "hint": "adjust --hsv-lo/--hsv-hi if area is 0; the ball should be a solid patch in the PNG"}
            print(json.dumps(rep, indent=1))
            return bool(rep["ok"])
        finally:
            cam.close()

    def _wait_settle(self, timeout: float):
        """The /robot/q pose once it has been still for 1 s; None when no feed arrives in time."""
        t0 = time.monotonic()
        last, still_since = None, None
        while time.monotonic() - t0 < timeout:
            q, age = self.qrx.get()
            if age is None:
                time.sleep(0.1)
                continue
            if last is not None and np.abs(q - last).max() < 1e-3:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since > 1.0:
                    return q
            else:
                still_since = None
            last = q
            time.sleep(0.05)
        return None

    def sphere(self):
        """A ball on the flange seen at n poses. You move the arm to each pose by whatever means you have (a teach
        pendant, a jog UI, your own program); the tool reads the pose from /robot/q, predicts the ball by FK, finds
        it in the cameras (colour via the cameras directly, or depth from the /depth stream with --no-rgb) and solves
        each camera's pose by Kabsch. The arm is never commanded from here."""
        self._need_robot("sphere")
        cams = self._cams()
        offset = np.array([float(v) for v in self.a.offset.split(",")])
        n = self.a.poses
        rgb_cams = self._open_rgb(cams) if self.a.rgb else {}
        rng = np.random.default_rng(0)
        az = np.radians(np.linspace(-70, 70, n))
        rs = rng.uniform(0.45, 0.75, n)
        zs = np.linspace(0.35, 0.9, n)[rng.permutation(n)]
        targets = np.stack([rs * np.cos(az), rs * np.sin(az), zs], 1)
        print("Suggested BALL positions in the robot base frame (mm; z above the base plate), spread over the cameras' view.")
        print(f"The ball sits {offset[2] * 1000:.0f} mm along the flange z axis; keep the arm out of its line of sight to each camera.")
        for i, p in enumerate(targets):
            print(f"  pose {i + 1:2d}: ball x {p[0] * 1000:7.0f}  y {p[1] * 1000:7.0f}  z {p[2] * 1000:7.0f}")
        self.qrx            # bind now so the feed is flowing by the first prompt
        obs: dict[int, list] = {c: [] for c in cams}
        truth: list = []
        try:
            for i in range(n):
                ans = input(f"pose {i + 1}/{n}: move the arm there, then press Enter when it is still (s = skip, q = finish) > ")
                if ans.strip().lower().startswith("q"):
                    break
                if ans.strip().lower().startswith("s"):
                    continue
                q = self._wait_settle(20.0)
                if q is None:
                    print(f"  no still /robot/q on port {self.a.q_port} within 20 s: is the joint feed running? pose skipped")
                    continue
                ball_w = self.robot.ee_position(q, offset=offset)
                flange_w = self.robot.ee_position(q)
                truth.append(ball_w)
                print(f"  q (deg) {np.round(np.degrees(q), 1).tolist()} -> ball (world, m) {np.round(ball_w, 3).tolist()}")
                if self.a.rgb:
                    for c in cams:
                        centre_s, info = self._rgb_ball(rgb_cams[c], int(self.a.dwell * 30), self.a.ball_r,
                                                        near=self._expected_in_sensor(c, ball_w))
                        if centre_s is None:
                            print(f"  cam {c}: ball not seen ({info})")
                            continue
                        T = self.pipe.cam(c).T_ws
                        if T is not None:
                            gap = float(np.linalg.norm(C.apply(T, centre_s[None])[0] - ball_w))
                            if gap > self.a.search_r:
                                print(f"  cam {c}: blob {gap * 100:.0f} cm from where the current calibration expects the ball; kept but flagged")
                        obs[c].append((centre_s, ball_w, info["rms_mm"] / 1000, info["inliers"]))
                        print(f"  cam {c}: ball (sensor) {np.round(centre_s, 3).tolist()} {info}")
                    continue
                frames = self.wait_frames(self.a.dwell, cams)
                for c in cams:
                    if c not in frames:
                        print(f"  cam {c}: no frames")
                        continue
                    P_s, P_w = self._ball_points(c, frames[c], q, ball_w, stem_from=flange_w)
                    if len(P_s) < 30:
                        print(f"  cam {c}: only {len(P_s)} candidate points near the ball")
                        continue
                    c0 = None if P_w is None else C.apply(C.invert(self.pipe.cam(c).T_ws), ball_w[None])[0]
                    centre_s, rms, n_in = C.fit_sphere_fixed_radius(P_s, self.a.ball_r, c0=c0)
                    obs[c].append((centre_s, ball_w, rms, n_in))
                    print(f"  cam {c}: ball centre (sensor) {np.round(centre_s, 3).tolist()} rms {rms * 1000:.1f} mm inliers {n_in}")
        finally:
            for cam in rgb_cams.values():
                cam.close()
        return self._finish_sweep(obs, cams, offset, len(truth), "sphere")

    def _finish_sweep(self, obs: dict, cams: list, offset, n_poses: int, method: str) -> bool:
        report = {"cmd": method, "poses": n_poses, "cams": {}}
        for c in cams:
            if len(obs[c]) < 4 and not (self.a.append and os.path.exists(os.path.join(self.a.calib_dir, f"sweep_{c}.npz")) and obs[c]):
                report["cams"][c] = {"ok": False, "reason": f"only {len(obs[c])} observations"}
                continue
            A = np.array([o[0] for o in obs[c]])
            B = np.array([o[1] for o in obs[c]])
            prev_sweep = os.path.join(self.a.calib_dir, f"sweep_{c}.npz")
            if self.a.append and os.path.exists(prev_sweep):
                d_prev = np.load(prev_sweep)
                A, B = np.vstack([d_prev["A"], A]), np.vstack([d_prev["B"], B])
                print(f"cam {c}: appended to the previous sweep -> {len(A)} observations")
            T, rms = C.kabsch(A, B)
            loo = []
            for k in range(len(A)):
                m = np.arange(len(A)) != k
                Tk, _ = C.kabsch(A[m], B[m])
                loo.append(np.linalg.norm(C.apply(Tk, A[k:k + 1])[0] - B[k]))
            np.savez(os.path.join(self.a.calib_dir, f"sweep_{c}.npz"), A=A, B=B)
            loo_rms = float(np.sqrt(np.mean(np.square(loo))))
            self._backup(self.calib_path(c))
            sha = C.save_sensor_calib(self.calib_path(c), c, T, {"method": method, "n": len(A), "rms_m": rms, "loo_rms_m": loo_rms,
                                                                  "ball_r": self.a.ball_r, "offset": np.asarray(offset).tolist()})
            self.pipe.set_calib(c, T)
            pan, tilt, fl = C.pan_tilt_of(T)
            report["cams"][c] = {"ok": True, "n": len(A), "rms_mm": round(rms * 1000, 1), "loo_rms_mm": round(loo_rms * 1000, 1),
                                 "cam_pos": np.round(T[:3, 3], 3).tolist(), "pan_deg": round(pan, 1), "tilt_deg": round(tilt, 1), "sha": sha}
        self._reload()
        print(json.dumps(report, indent=1))
        return all(v.get("ok") for v in report["cams"].values()) and bool(report["cams"])

    def refine(self):
        """ICP: the robot's own foreground blob (points the mask would remove) against sphere surfaces."""
        self._need_robot("refine")
        cam = self.a.cam
        cs = self.pipe.cam(cam)
        if cs.T_ws is None:
            raise SystemExit("seed or sphere first")
        q, age = self.qrx.get()
        if age is None:
            time.sleep(0.5)
            q, age = self.qrx.get()
        if age is None:
            raise SystemExit(f"no /robot/q on port {self.a.q_port}: refine needs the arm's measured pose")
        frames = self.wait_frames(2.0, [cam])[cam]
        rays = self.pipe._rays_for(cs, frames[0].stride)
        pts = []
        for p in frames:
            fg = cs.bg.foreground(p.depth) if cs.bg.ready else (p.depth > 0)
            P_s, _ = unproject(p.depth, rays, fg)
            pts.append(P_s)
        P_s = np.vstack(pts)
        model = self.robot.surface_samples(q, 60)
        tree = cKDTree(model)
        T = cs.T_ws.copy()
        for it in range(self.a.iters):
            P_w = C.apply(T, P_s)
            near = np.linalg.norm(P_w - self.robot.T_world_base[:3, 3], axis=1) < 1.3
            d, idx = tree.query(P_w[near])
            keep = d < 0.06 + 0.02 * max(0, 5 - it)
            if keep.sum() < 50:
                print(f"iter {it}: only {keep.sum()} correspondences; stop")
                break
            dT, rms = C.kabsch(P_w[near][keep], model[idx[keep]])
            T = dT @ T
            print(f"iter {it}: n={keep.sum()} rms={rms*1000:.1f} mm")
        rep = self._commit(cam, T, "+refine-icp", {"q": q.tolist()})
        self._reload()
        print(json.dumps({"cmd": "refine", "cam": cam, "sha": rep["sha"]}))

    def validate(self):
        rep = {"cmd": "validate", "cams": {}}
        for f in sorted(glob.glob(os.path.join(self.a.calib_dir, "sweep_*.npz"))):
            c = int(os.path.basename(f)[6:-4])
            d = np.load(f)
            A, B = d["A"], d["B"]
            loo = []
            for k in range(len(A)):
                m = np.arange(len(A)) != k
                Tk, _ = C.kabsch(A[m], B[m])
                loo.append(np.linalg.norm(C.apply(Tk, A[k:k + 1])[0] - B[k]))
            rep["cams"][c] = {"loo_rms_mm": round(float(np.sqrt(np.mean(np.square(loo)))) * 1000, 1), "n": len(A)}
        # live: robot-mask leftover at the current pose; with no q source the mask would sit on a phantom pose
        age = None
        if self.robot is not None:
            self.qrx                                   # bind first so a running publisher has time to deliver
            time.sleep(0.3)
            q, age = self.qrx.get()
            rep["q_deg"] = None if age is None else np.round(np.degrees(q), 1).tolist()
            rep["q_age_s"] = None if age is None else round(age, 3)
        if self.robot is None:
            print("no robot in the config: robot leftover not computed")
            frames = {}
        elif age is None:
            print(f"no /robot/q on port {self.a.q_port} (publish the arm's measured joint angles there): robot leftover not computed")
            frames = {}
        else:
            frames = self.wait_frames(1.0)
        for c, lst in frames.items():
            cs = self.pipe.cam(c)
            if not self.pipe.ready(c):
                continue
            self.pipe.set_q(q)
            Pk, _ = self.pipe.points_for(lst[-1])
            near = np.linalg.norm(Pk - self.robot.T_world_base[:3, 3], axis=1) < 1.2 if len(Pk) else np.zeros(0, bool)
            rep["cams"].setdefault(c, {})["robot_leftover_pts"] = int(near.sum())
            rep["cams"][c]["bg_ready"] = cs.bg.ready
        print(json.dumps(rep, indent=1))
        ok = bool(rep["cams"]) and age is not None and all(v.get("loo_rms_mm", 99) < 15 and v.get("robot_leftover_pts", 0) < 30 for v in rep["cams"].values())
        print("RESULT:", "OK" if ok else "CHECK")
        return ok


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["seed", "learn-bg", "plane", "yaw", "table", "nudge", "aim", "sphere", "probe-color", "refine", "validate", "undo"])
    ap.add_argument("--config", default="config/tracker.json", help="tracker.json (+ tracker.local.json overlay)")
    ap.add_argument("--calib-dir", default="config/calib")
    ap.add_argument("--in", dest="in_port", type=int, default=P.PORT_DEPTH)
    ap.add_argument("--q-port", type=int, default=P.PORT_ROBOT_Q, help="/robot/q feed for sphere / refine / validate")
    ap.add_argument("--ctrl-port", type=int, default=P.PORT_TRACKER_CTRL)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--cams", default="0,1")
    ap.add_argument("--pos", default=None, help="seed: camera position x,y,z (m); default from config cams[cam]")
    ap.add_argument("--aim", default=None, help="seed: look-at point x,y,z (m); default from config cams[cam]")
    ap.add_argument("--seconds", type=float, default=10.0, help="learn-bg: how long to collect")
    ap.add_argument("--clip", nargs="*", default=None, help="recorded clip(s) recordings/<tag>_*.npz for plane/table/yaw")
    ap.add_argument("--z-band", default="-1.5,0.6", help="plane: world z band of the floor candidates under the current pose")
    ap.add_argument("--band", type=float, default=0.06, help="table: |z - top_z| band of the table-top candidates (m)")
    ap.add_argument("--yaw", type=float, default=0.0, help="nudge: rotation about the world z axis (deg, +ccw seen from above)")
    ap.add_argument("--dx", type=float, default=0.0, help="nudge: world x shift (m)")
    ap.add_argument("--dy", type=float, default=0.0, help="nudge: world y shift (m)")
    ap.add_argument("--ref", default="1.6,0", help="yaw: world x,y (m) of the mark the person stands on")
    ap.add_argument("--min-run", type=float, default=5.0, help="yaw: seconds the person must stand still")
    ap.add_argument("--max-range-err", type=float, default=0.10, help="yaw: reject if camera->person range differs from the mark by more")
    ap.add_argument("--dry-run", action="store_true", help="yaw: report only (verification on a second mark)")
    ap.add_argument("--height", type=float, default=None, help="aim: override the camera height (m)")
    ap.add_argument("--zone", default="1.2,4.0,2.0", help="aim: coverage zone x_min,x_max,y_half (m)")
    ap.add_argument("--write", action="store_true", help="aim: write cams[].aim into the config overlay")
    ap.add_argument("--poses", type=int, default=12, help="sphere: how many poses to prompt for")
    ap.add_argument("--dwell", type=float, default=1.0, help="sphere: seconds of frames per pose")
    ap.add_argument("--ball-r", type=float, default=0.075, help="sphere: ball radius (m)")
    ap.add_argument("--offset", default="0,0,0.180", help="sphere: ball centre in the flange frame (m); 0.180 = the printed target in calib_target/")
    ap.add_argument("--search-r", type=float, default=0.3, help="sphere: accept depth points within this of the predicted ball (m)")
    ap.add_argument("--near", type=float, default=1.2, help="sphere (rgb): the ball is the largest orange blob within this distance (m) of where the "
                                                           "robot and the current calibration put it; 0 = largest blob in the whole picture")
    ap.add_argument("--stem-r", type=float, default=0.03, help="sphere: exclusion radius around the flange->ball stem (m)")
    ap.add_argument("--rgb", dest="rgb", action="store_true", default=True, help="sphere: find the ball by colour (default)")
    ap.add_argument("--no-rgb", dest="rgb", action="store_false", help="sphere: depth-only fallback via the /depth stream")
    ap.add_argument("--cam-ips", default=None, help="camera IPs by cam index, e.g. 10.0.0.21,10.0.0.22 (rgb mode opens them directly; required)")
    ap.add_argument("--hsv-lo", default="5,120,80", help="OpenCV HSV lower bound for the ball colour")
    ap.add_argument("--hsv-hi", default="25,255,255")
    ap.add_argument("--min-px", type=int, default=20)
    ap.add_argument("--iters", type=int, default=15, help="refine: ICP iterations")
    ap.add_argument("--append", action="store_true", help="sphere: add this run to the saved sweep_*.npz before solving (more poses, same solve)")
    ap.add_argument("--log", default=None, help="also write everything printed to this file")
    a = ap.parse_args(argv)
    if a.log:
        sys.stdout = _Tee(sys.stdout, open(a.log, "w", encoding="utf-8"))
    s = Session(a)
    try:
        return 0 if getattr(s, a.cmd.replace("-", "_"))() is not False else 1
    finally:
        s.stop()


if __name__ == "__main__":
    sys.exit(main() or 0)
