# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Tracker process: /depth (+/intrinsics) from the capture processes in, /tracker/target out at frame rate,
/tracker/tracks at 10 Hz, /tracker/zone per configured zone at frame rate, /hb at 1 Hz. With a robot in the
config it also listens for /robot/q (measured joint angles) to mask the arm. Config live-reloads on file change.

  python -m rt_depth_tracker.run --config config/tracker.json --calib-dir config/calib
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import threading
import time

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker import protocol as P
from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.ctrl_rx import CtrlRx, install_quit_signals
from rt_depth_tracker.pipeline import Pipeline
from rt_depth_tracker.pacing import Pacer
from rt_depth_tracker.record import Recorder


class DepthRx:
    def __init__(self, port: int):
        self._lock = threading.Lock()
        self.latest: dict[int, tuple[P.DepthPacket, float]] = {}
        self.intr: dict[int, P.Intrinsics] = {}
        self._counts: dict[int, list] = {}
        self.event = threading.Event()
        self._assembler = P.DepthAssembler()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.2)
        self._running = True
        threading.Thread(target=self._loop, name="DepthRx", daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            obj = P.unpack(data)
            now = time.monotonic()
            if isinstance(obj, P.DepthChunk):
                pkt = self._assembler.feed(obj)
                if pkt is None:
                    continue
                with self._lock:
                    self.latest[pkt.cam_id] = (pkt, now)
                    self._counts.setdefault(pkt.cam_id, []).append(now)
                self.event.set()
            elif isinstance(obj, P.Intrinsics):
                with self._lock:
                    self.intr[obj.cam_id] = obj

    def take(self, max_age: float):
        """{cam_id: packet} for packets younger than max_age, plus {cam_id: intrinsics}."""
        now = time.monotonic()
        with self._lock:
            pk = {c: p for c, (p, t) in self.latest.items() if now - t <= max_age}
            intr = dict(self.intr)
        return pk, intr

    def rates(self) -> dict:
        now = time.monotonic()
        out = {}
        with self._lock:
            for c, ts in self._counts.items():
                ts[:] = [t for t in ts if now - t < 2.0]
                out[c] = round(len(ts) / 2.0, 1)
        return out

    def stop(self):
        self._running = False
        self._sock.close()


class RobotRx:
    def __init__(self, port: int):
        self._lock = threading.Lock()
        self.q = np.zeros(6)
        self.stamp = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.2)
        self._running = True
        threading.Thread(target=self._loop, name="RobotRx", daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            obj = P.unpack(data)
            if isinstance(obj, P.RobotQ):
                with self._lock:
                    self.q, self.stamp = obj.q, time.monotonic()

    def get(self):
        with self._lock:
            return self.q.copy(), (None if self.stamp is None else time.monotonic() - self.stamp)

    def stop(self):
        self._running = False
        self._sock.close()


class TargetRx:
    """Mirror of the tracker's own output (viewer): latest /tracker/target, /tracker/tracks and /tracker/zone set with their age."""

    def __init__(self, port: int):
        self._lock = threading.Lock()
        self.target = None
        self.stamp = None
        self.tracks = None
        self.tracks_stamp = None
        self.zones: dict = {}
        self.zones_stamp = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.2)
        self._running = True
        threading.Thread(target=self._loop, name="TargetRx", daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            obj = P.unpack(data)
            if isinstance(obj, P.Target):
                with self._lock:
                    self.target, self.stamp = obj, time.monotonic()
            elif isinstance(obj, np.ndarray):
                with self._lock:
                    self.tracks, self.tracks_stamp = obj, time.monotonic()
            elif isinstance(obj, P.ZoneMsg):
                with self._lock:
                    self.zones[obj.idx] = obj
                    self.zones_stamp = time.monotonic()

    def get(self):
        """(target, age_s, tracks rows, tracks age_s); ages are None until the first message."""
        with self._lock:
            now = time.monotonic()
            return (self.target, None if self.stamp is None else now - self.stamp,
                    self.tracks, None if self.tracks_stamp is None else now - self.tracks_stamp)

    def get_zones(self):
        """({idx: ZoneMsg}, age_s of the newest); empty and None until the first zone message."""
        with self._lock:
            return dict(self.zones), None if self.zones_stamp is None else time.monotonic() - self.zones_stamp

    def stop(self):
        self._running = False
        self._sock.close()


class TrackerRuntime:
    def _persist_intrinsics(self, k):
        """Keep the last intrinsics next to the calibration files so the offline tools can unproject without a clip."""
        d = dict(k.__dict__)
        if self._saved_intr.get(k.cam_id) == d:
            return
        path = os.path.join(self.calib_dir, f"intrinsics_{k.cam_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            self._saved_intr[k.cam_id] = d
        except OSError:
            pass

    def __init__(self, config_path: str | None, calib_dir: str, in_port: int = P.PORT_DEPTH,
                 out=("127.0.0.1", P.PORT_TRACKER_OUT), q_port: int = P.PORT_ROBOT_Q, ctrl_port: int = P.PORT_TRACKER_CTRL,
                 hb=("127.0.0.1", P.PORT_HB), record_dir: str | None = None, name: str = "tracker", log=print):
        self.config_path, self.calib_dir, self.name, self.log = config_path, calib_dir, name, log
        self._saved_intr: dict = {}
        self.pacer = Pacer()
        self.cfg = TrackerConfig.load(config_path)
        self._cfg_mtime = self._mtime(config_path)
        self.pipe = Pipeline(self.cfg)
        self.rx = DepthRx(in_port)
        self.robot = RobotRx(q_port) if self.pipe.robot is not None else None     # no robot: no joint listener, no mask
        self.ctrl = CtrlRx(ctrl_port)
        outs = out if isinstance(out, list) else [out]
        self.outs = [tuple(o) for o in outs]           # the consumer first, then any mirror (the viewer)
        self.out, self.hb_addr = self.outs[0], tuple(hb)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recorder: Recorder | None = None
        self.record_dir = record_dir or os.path.join(os.path.dirname(calib_dir.rstrip("/\\")), "..", "recordings")
        self._running = True
        self._last_hb = self._last_tracks = self._last_cfg_check = 0.0
        self._step_ms: list[float] = []
        self._last_decision = None
        self._last_zones: list = []
        self._frames = 0
        self._step_errors = 0          # in the heartbeat: a step that never succeeds is invisible otherwise
        self._last_err_t = 0.0
        self._last_hint_log, self._hint_skipped = 0.0, 0   # one 'hint a -> b' line per change, at most one per second
        self._bg_learning_started = False
        self._last_heal_log = 0.0
        self._tune_pending: dict = {}      # /tracker/set values not yet written to the config file
        self._tune_dirty_at: float | None = None
        os.makedirs(calib_dir, exist_ok=True)
        self._calib_mtime_seen = self._calib_mtime(calib_dir)
        self.load_calib()
        self.load_bg()
        self.log(f"[{self.name}] robot self-mask {'on, /robot/q on ' + str(q_port) if self.robot is not None else 'off (no robot in the config)'}")

    @staticmethod
    def _mtime(p):
        tot = 0.0
        for f in TrackerConfig.paths(p):
            try:
                tot += os.path.getmtime(f)
            except OSError:
                pass
        return tot

    # ---- persistence ----------------------------------------------------------------------------
    def load_calib(self) -> int:
        n = 0
        for f in C.sensor_calib_files(self.calib_dir):
            try:
                T, meta = C.load_sensor_calib(f)
                self.pipe.set_calib(int(meta["cam_id"]), T)
                n += 1
            except Exception as e:  # noqa: BLE001
                self.log(f"[{self.name}] bad calib {f}: {e}")
        if n == 0:
            self.log(f"[{self.name}] WARNING no calibration in {self.calib_dir}: "
                     f"the cameras cannot be placed in the world, so nobody will ever be seen")
        else:
            self.log(f"[{self.name}] calibration files loaded: {n}")
        return n

    def load_bg(self):
        for f in sorted(glob.glob(os.path.join(self.calib_dir, "bg_*.npz"))):
            try:
                cam_id = int(os.path.basename(f)[3:-4])
                cs = self.pipe.cam(cam_id)
                want = None if cs.intr is None else (cs.intr.height, cs.intr.width)
                if cs.bg.load(f, shape=want):
                    self.log(f"[{self.name}] background for cam {cam_id} loaded ({f})")
                else:
                    self.log(f"[{self.name}] WARNING background for cam {cam_id} rejected ({f}): "
                             f"wrong shape for this depth mode, or unreadable")
            except Exception as e:  # noqa: BLE001
                self.log(f"[{self.name}] bad bg {f}: {e}")

    def save_bg(self):
        for cid, cs in self.pipe.cams.items():
            if cs.bg.ready:
                cs.bg.save(os.path.join(self.calib_dir, f"bg_{cid}.npz"))
        self.log(f"[{self.name}] backgrounds saved")

    @staticmethod
    def _calib_mtime(calib_dir):
        """Only sensor_*.json: intrinsics_*.json are rewritten by this process, so watching them
        would make the reload retrigger itself forever."""
        tot = 0.0
        for f in C.sensor_calib_files(calib_dir):
            try:
                tot += os.path.getmtime(f)
            except OSError:
                pass
        return tot

    def maybe_reload_calib(self, now):
        """Calibration dropped in while we are running (a remote recalibration, or a restore after
        it went missing) takes effect on the next frame, with no restart."""
        m = self._calib_mtime(self.calib_dir)
        if m != self._calib_mtime_seen:
            self._calib_mtime_seen = m
            self.load_calib()
            self.load_bg()
            self.log(f"[{self.name}] calibration changed on disk: reloaded")

    def maybe_reload_config(self, now):
        if now - self._last_cfg_check < 1.0:
            return
        self._last_cfg_check = now
        self.maybe_reload_calib(now)
        m = self._mtime(self.config_path)
        if m != self._cfg_mtime:
            self._cfg_mtime = m
            try:
                new = TrackerConfig.load(self.config_path)
                self.cfg.__dict__.update(new.__dict__)
                self.log(f"[{self.name}] config reloaded")
            except Exception as e:  # noqa: BLE001
                self.log(f"[{self.name}] config reload failed: {e}")

    def maybe_save_tune(self, now, settle_s: float = 2.0):
        """Slider values go to the config file once the drag has settled, so a restart keeps them."""
        if self._tune_dirty_at is None or now - self._tune_dirty_at < settle_s:
            return
        pending, self._tune_pending, self._tune_dirty_at = self._tune_pending, {}, None
        paths = [p for p in TrackerConfig.paths(self.config_path) if os.path.exists(p)]
        if not paths:
            return
        target = paths[-1]     # the overlay wins on reload, so it is the file to edit (calib_tool aim --write does the same)
        try:
            with open(target, "r", encoding="utf-8") as f:
                d = json.load(f)
            d.update(pending)
            tmp = target + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
                f.write("\n")
            os.replace(tmp, target)
            self._cfg_mtime = self._mtime(self.config_path)      # our own write is not a reload
            self.log(f"[{self.name}] tune saved to {os.path.basename(target)}: " + ", ".join(f"{k}={v}" for k, v in pending.items()))
        except Exception as e:  # noqa: BLE001
            self.log(f"[{self.name}] tune save failed ({target}): {e}")

    # ---- control --------------------------------------------------------------------------------
    def handle_ctrl(self):
        while not self.ctrl.queue.empty():
            addr, params = self.ctrl.queue.get_nowait()
            if addr == P.ADDR_LEARN_BG:
                self.pipe.learn_background(frames=int(params[0]) if params else None)
                self._bg_learning_started = True
                self.log(f"[{self.name}] learning background")
            elif addr == P.ADDR_RECORD:
                # /tracker/record <tag> starts a tagged clip; 0 / "off" stops; no arg toggles
                arg = params[0] if params else None
                tag = arg if isinstance(arg, str) and arg.lower() not in ("0", "off", "stop") else None
                on = tag is not None or (arg is None and self.recorder is None) or (isinstance(arg, (int, float)) and int(arg) != 0)
                if on and self.recorder is None:
                    self.recorder = Recorder(self.record_dir, tag=tag or "rec")
                    self.recorder.set_meta(intr={c: cs.intr for c, cs in self.pipe.cams.items() if cs.intr},
                                           calib={c: cs.T_ws for c, cs in self.pipe.cams.items() if cs.T_ws is not None})
                    self.log(f"[{self.name}] recording '{self.recorder.tag}' to {self.record_dir}")
                elif not on and self.recorder is not None:
                    self.recorder.close()
                    self.log(f"[{self.name}] recording stopped: {self.recorder.files}")
                    self.recorder = None
            elif addr == P.ADDR_RELOAD_CALIB:
                self.load_calib()
                self.load_bg()
            elif addr == P.ADDR_TRACKER_QUIT:
                self.log(f"[{self.name}] quit requested")
                self._running = False
            elif addr == P.ADDR_SET:
                key = params[0] if params else None
                if key in P.TUNABLE_KEYS and len(params) > 1:
                    val = type(getattr(self.cfg, key))(params[1])
                    if isinstance(val, float):
                        val = round(val, 6)                  # OSC carries float32: 0.9 arrives as 0.89999997
                    setattr(self.cfg, key, val)          # the pipeline shares this object: live on the next frame
                    self._tune_pending[key] = val
                    self._tune_dirty_at = time.monotonic()
                else:
                    self.log(f"[{self.name}] ignored /tracker/set {params}: tunable keys are {P.TUNABLE_KEYS}")

    # ---- main loop --------------------------------------------------------------------------------
    def once(self, timeout: float = 0.2) -> bool:
        if not self.rx.event.wait(timeout):
            return False
        self.rx.event.clear()
        now = time.monotonic()
        packets, intr = self.rx.take(max_age=0.2)
        for k in intr.values():
            self.pipe.set_intrinsics(k)
            self._persist_intrinsics(k)
        if self.robot is not None:
            q, q_age = self.robot.get()
            self.pipe.set_q(q, q_age)
        fresh = self.pacer.offer(packets, now)
        if not fresh:
            return False
        dec, dbg = self.pipe.step(packets, now)
        self._frames += 1
        self._log_hint_change(dec, dbg, now)
        self._last_decision, self._last_zones = dec, dbg.zones
        self._step_ms.append(dbg.step_ms)
        self._step_ms = self._step_ms[-300:]
        if self.recorder is not None:
            for pkt in fresh.values():
                self.recorder.add(pkt, self.pipe.q, now)
        dg = P.pack_target(P.Target(*dec.target, dec.hint, dec.track_id, dec.score))
        for o in self.outs:
            self.tx.sendto(dg, o)
        if now - self._last_tracks > 0.1:
            self._last_tracks = now
            dg = P.pack_tracks(self.pipe.tracks_rows(dbg.tracks))
            for o in self.outs:
                self.tx.sendto(dg, o)
        for z in dbg.zones:                             # only ever non-empty with zones configured
            dg = P.pack_zone(z)
            for o in self.outs:
                self.tx.sendto(dg, o)
        if self._bg_learning_started and not any(cs.bg.learning for cs in self.pipe.cams.values()):
            self._bg_learning_started = False
            self.save_bg()
        if (self.pipe.absorbed or self.pipe.unhealed) and time.monotonic() - self._last_heal_log > 300.0:
            # the learned file is never rewritten: healing is a session overlay, this line is its only trace
            healed_now = sum(cs.bg.healed_px for cs in self.pipe.cams.values() if cs.bg.ready)
            self.log(f"[{self.name}] healing: {self.pipe.absorbed} px healed, {self.pipe.unhealed} px un-healed in the last 5 min; "
                     f"{healed_now} px healed now (in memory only)")
            self.pipe.absorbed, self.pipe.unhealed, self._last_heal_log = 0, 0, time.monotonic()
        return True

    def _log_hint_change(self, dec, dbg, now: float):
        """One line per hint change (idle / watch / engage / reach / sleep) with what the presence gate saw: the
        confirmed tracks and its own state. At most one line per second; a flapping hint is counted, not spelt out."""
        prev = None if self._last_decision is None else self._last_decision.hint
        if prev == dec.hint:
            return
        if now - self._last_hint_log < 1.0:
            self._hint_skipped += 1
            return
        more = f"  (+{self._hint_skipped} more changes since the last line)" if self._hint_skipped else ""
        self._hint_skipped, self._last_hint_log = 0, now
        self.log(f"[{self.name}] hint {prev} -> {dec.hint}  tracks {len(dbg.tracks)}  presence {self.pipe.sleep.state}{more}")

    def metrics(self) -> dict:
        d = self._last_decision
        return {"targets": len(self.pipe.tracker.tracks), "hint": d.hint if d else None, "track_id": d.track_id if d else -1,
                "presence": self.pipe.sleep.state,
                "rx_hz": self.rx.rates(), "cams_calibrated": [c for c, cs in self.pipe.cams.items() if cs.T_ws is not None],
                "cams_ready": [c for c in self.pipe.cams if self.pipe.ready(c)],
                "bg": {str(k): v for k, v in self.pipe.bg_status().items()},
                "q_age": (None if self.robot is None or self.robot.get()[1] is None else round(self.robot.get()[1], 3)),
                "zones": {z.name: ("reach" if z.reaching else "lean" if z.leaning else "occupied" if z.occupied else "") for z in (self._last_zones or [])},
                "step_ms_p95": (round(float(np.percentile(self._step_ms, 95)), 2) if self._step_ms else None),
                "frames": self._frames, "step_errors": self._step_errors,
                "recording": self.recorder is not None, "interest_model": self.cfg.interest_model,
                "tune": {k: getattr(self.cfg, k) for k in P.TUNABLE_KEYS},
                "bg_absorbed": self.pipe.absorbed_total}

    def run(self):
        self.log(f"[{self.name}] up: depth:{self.rx._sock.getsockname()[1]} -> {self.outs}")
        try:
            while self._running:
                try:
                    self.once()
                except Exception as e:  # noqa: BLE001  one bad frame must never take the process down
                    self._step_errors += 1
                    n, now_e = self._step_errors, time.monotonic()
                    # one line per 30 s once it is clearly not a one-off: a permanent failure loop
                    # must not look like occasional noise in the log
                    if n <= 10 or now_e - self._last_err_t >= 30.0:
                        self._last_err_t = now_e
                        self.log(f"[{self.name}] step error #{n}: {type(e).__name__}: {e}")
                now = time.monotonic()
                self.handle_ctrl()
                self.maybe_reload_config(now)
                self.maybe_save_tune(now)
                if now - self._last_hb > 1.0:
                    self._last_hb = now
                    self.tx.sendto(P.pack_hb(self.name, self.metrics()), self.hb_addr)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self.recorder is not None:
            self.recorder.close()
        for r in (self.rx, self.robot, self.ctrl):
            if r is not None:
                r.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--calib-dir", default=None, help="default: <config dir>/calib")
    ap.add_argument("--in", dest="in_port", type=int, default=P.PORT_DEPTH)
    ap.add_argument("--out", default=f"127.0.0.1:{P.PORT_TRACKER_OUT}",
                    help="host:port[,host:port]: the consumer, then mirrors (the viewer listens on 7016)")
    ap.add_argument("--q", type=int, default=P.PORT_ROBOT_Q, help="/robot/q port (listened to only with a robot in the config)")
    ap.add_argument("--ctrl", type=int, default=P.PORT_TRACKER_CTRL)
    ap.add_argument("--hb", default=f"127.0.0.1:{P.PORT_HB}")
    ap.add_argument("--record-dir", default=None)
    a = ap.parse_args(argv)
    calib_dir = a.calib_dir or os.path.join(os.path.dirname(os.path.abspath(a.config)) if a.config else "config", "calib")
    outs = [(h, int(p)) for h, p in (s.split(":") for s in a.out.split(","))]
    hh, hp = a.hb.split(":")
    rt = TrackerRuntime(a.config, calib_dir, a.in_port, outs, a.q, a.ctrl, (hh, int(hp)), a.record_dir)
    install_quit_signals()
    rt.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
