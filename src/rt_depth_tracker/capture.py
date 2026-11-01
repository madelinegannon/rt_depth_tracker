# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""One Orbbec depth camera (Femto Mega over the network, pyorbbecsdk2) -> OSC /depth stream (organized uint16
depth, 2D stride) + /intrinsics + /hb. This is the sensor side of the plug point: any other sensor is a
process that sends the same two messages (docs/protocol.md); nothing downstream imports the camera SDK.

One process per camera. Reconnects forever with backoff.

  python -m rt_depth_tracker.capture --ip <camera ip> --cam-id 0 --probe 15
      lists depth work modes / profiles, streams the binned mode @30 for 15 s, prints fps, valid pixels,
      intrinsics, and temporal-noise stats in a centre ROI, then exits.

  python -m rt_depth_tracker.capture --ip <camera ip> --cam-id 0 [--ctrl 7017]
      runs forever, sending to 127.0.0.1:7010 (+ heartbeat to :9000); /capture/quit on --ctrl stops the
      stream, releases the camera and exits 0 (a supervisor's clean stop).

  python -m rt_depth_tracker.capture --ip <camera ip> --set-sync secondary --delay-us 160
      one-time: writes multi-device sync mode to camera flash and exits.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import numpy as np

from rt_depth_tracker import protocol as P
from rt_depth_tracker.ctrl_rx import CtrlRx, install_quit_signals

DEFAULT_MODE = "nfov 2x2 binned"
DEFAULT_RES = (320, 288)
DEFAULT_FPS = 30
DEFAULT_STRIDE = 2
NO_FRAME_TIMEOUT_S = 3.0
BACKOFF_S = (1, 2, 4, 8, 15, 30)
OPEN_LEASE_S = 45.0     # the SDK holds the GIL in create_net_device / pipe.start (a failing open can take 14 s), so the heartbeat before it carries a lease
CLOSE_LEASE_S = 15.0    # same for pipe.stop on a dead device


def _norm(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def match_work_mode(names: list[str], want: str) -> str | None:
    """Pick the depth work mode whose normalized name contains every token of `want`.
    Token '2X2' also accepts 'BINNED' spelled without the 2x2; 'UNBINNED' must not match binned."""
    tokens = [_norm(t) for t in want.split()]
    for n in names:
        nn = _norm(n)
        ok = all(t in nn for t in tokens)
        if ok and "BINNED" in tokens and "UNBINNED" in nn and "UNBINNED" not in tokens:
            ok = False
        if ok:
            return n
    return None


class Capture:
    def __init__(self, ip: str, cam_id: int, port: int = 8090, mode: str = DEFAULT_MODE,
                 res=DEFAULT_RES, fps: int = DEFAULT_FPS, stride: int = DEFAULT_STRIDE,
                 out=("127.0.0.1", P.PORT_DEPTH), hb=("127.0.0.1", P.PORT_HB), name: str | None = None,
                 clock=time.monotonic, sleep=time.sleep, ctrl_port: int | None = None):
        self.ip, self.cam_id, self.port = ip, cam_id, port
        self._clock, self._sleep = clock, sleep
        self.ctrl = CtrlRx(ctrl_port) if ctrl_port else None
        self._quit = False
        self.backoff_s, self.no_frame_timeout_s = BACKOFF_S, NO_FRAME_TIMEOUT_S
        self.mode, self.res, self.fps, self.stride = mode, res, fps, stride
        self.outs = [tuple(out)] if isinstance(out, tuple) else [tuple(o) for o in out]
        self.hb_addr = hb
        self.name = name or f"capture_{'abcdefgh'[cam_id] if cam_id < 8 else cam_id}"
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self.seq = 0
        self.intr: P.Intrinsics | None = None
        self.serial = ""
        self.fps_ema = 0.0
        self.valid_px = 0
        self.connected = False
        self.phase, self._phase_t0, self.attempt = "opening", self._clock(), 0
        self.log = lambda m: print(f"[{self.name}] {m}", flush=True)

    def _set_phase(self, phase: str):
        if phase != self.phase:
            self.phase, self._phase_t0 = phase, self._clock()

    # ---- device / pipeline -------------------------------------------------
    def open(self):
        from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline  # noqa: F401
        self._set_phase("opening")
        self.send_hb(busy_s=OPEN_LEASE_S)
        ctx = Context()
        ctx.set_logger_to_console(__import__("pyorbbecsdk").OBLogLevel.WARNING) if hasattr(ctx, "set_logger_to_console") else None
        dev = ctx.create_net_device(self.ip, self.port)
        if dev is None:
            raise RuntimeError(f"create_net_device({self.ip}:{self.port}) returned None")
        info = dev.get_device_info()
        self.serial = info.get_serial_number()
        self.log(f"connected {info.get_name()} sn={self.serial} fw={info.get_firmware_version()}")
        self.send_hb(busy_s=OPEN_LEASE_S)       # the work-mode calls and pipe.start hold the GIL too

        names = self._work_mode_names(dev)
        if names:
            pick = match_work_mode(names, self.mode)
            cur = dev.get_depth_work_mode()
            cur_name = getattr(cur, "name", str(cur))
            self.log(f"depth work modes: {names}; current={cur_name!r}; want={pick!r}")
            if pick and _norm(pick) != _norm(cur_name):
                dev.set_depth_work_mode(pick)
                self.log(f"set depth work mode -> {pick}")
        pipe = Pipeline(dev)
        plist = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        prof = None
        try:
            prof = plist.get_video_stream_profile(self.res[0], self.res[1], OBFormat.Y16, self.fps)
        except Exception as e:  # noqa: BLE001
            self.log(f"exact profile {self.res}@{self.fps} Y16 not found ({e}); using default")
            prof = plist.get_default_video_stream_profile()
        self.log(f"depth profile: {prof.get_width()}x{prof.get_height()}@{prof.get_fps()} {prof.get_format()}")
        self.intr = self._intrinsics(prof)
        cfg = Config()
        cfg.enable_stream(prof)
        pipe.start(cfg)
        self.ctx, self.dev, self.pipe = ctx, dev, pipe
        self.connected = True
        self._set_phase("streaming")
        self.send_hb()

    @staticmethod
    def _work_mode_names(dev) -> list[str]:
        try:
            lst = dev.get_depth_work_mode_list()
            return [getattr(lst.get_depth_work_mode_by_index(i), "name", str(lst.get_depth_work_mode_by_index(i)))
                    for i in range(lst.get_count())]
        except Exception:  # noqa: BLE001
            return []

    def _intrinsics(self, prof) -> P.Intrinsics | None:
        try:
            k, d = prof.get_intrinsic(), prof.get_distortion()
            intr = P.Intrinsics(self.cam_id, int(k.width), int(k.height), float(k.fx), float(k.fy),
                                float(k.cx), float(k.cy), float(d.k1), float(d.k2), float(d.k3),
                                float(d.k4), float(d.k5), float(d.k6), float(d.p1), float(d.p2))
            self.log(f"intrinsics: {json.dumps(intr.__dict__)}")
            return intr
        except Exception as e:  # noqa: BLE001
            self.log(f"WARNING intrinsics unavailable: {e}")
            return None

    def close(self):
        self.connected = False
        self.fps_ema, self.valid_px = 0.0, 0     # a reconnect must not report the old 30 fps or the old pixel count
        self._set_phase("closing" if self._quit else "reconnecting")
        self.send_hb(busy_s=CLOSE_LEASE_S)       # pipe.stop on a dead device can block
        pipe = self.__dict__.pop("pipe", None)   # drop every SDK handle: the camera is released before the process exits
        try:
            if pipe is not None:
                pipe.stop()
        except Exception:  # noqa: BLE001
            pass
        self.__dict__.pop("dev", None)
        self.__dict__.pop("ctx", None)

    def _poll_ctrl(self):
        if self.ctrl is None:
            return
        while not self.ctrl.queue.empty():
            addr, _params = self.ctrl.queue.get_nowait()
            if addr == P.ADDR_CAPTURE_QUIT:
                self.log("quit requested")
                self._quit = True

    # ---- frames --------------------------------------------------------------
    def next_frame(self, timeout_ms: int = 1000):
        """Returns (depth_mm uint16 (h,w), device_ts_s) or None on timeout."""
        fs = self.pipe.wait_for_frames(timeout_ms)
        if fs is None:
            return None
        df = fs.get_depth_frame()
        if df is None:
            return None
        h, w = df.get_height(), df.get_width()
        raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(h, w)
        scale = float(df.get_depth_scale())
        depth = raw if abs(scale - 1.0) < 1e-6 else np.clip(raw.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
        return depth, df.get_timestamp_us() * 1e-6

    def send(self, depth: np.ndarray, ts: float):
        self.seq += 1
        for dgram in P.pack_depth(self.cam_id, self.seq, depth, self.stride, ts):
            for dst in self.outs:
                self.sock.sendto(dgram, dst)

    def send_intrinsics(self):
        if self.intr is not None:
            for dst in self.outs:
                self.sock.sendto(P.pack_intrinsics(self.intr), dst)

    def send_hb(self, busy_s: float | None = None):
        """busy_s is a lease: 'expect my silence for up to this long', sent only right before an SDK call that
        holds the GIL. Never on an ordinary heartbeat, or a hung process could hide behind it."""
        m = {"fps": round(self.fps_ema, 1), "valid_px": int(self.valid_px), "serial": self.serial,
             "connected": self.connected, "seq": self.seq, "ip": self.ip,
             "phase": self.phase, "phase_s": round(self._clock() - self._phase_t0, 1), "attempt": self.attempt}
        if busy_s is not None:
            m["busy_s"] = float(busy_s)
        self.sock.sendto(P.pack_hb(self.name, m), self.hb_addr)

    # ---- main loop -----------------------------------------------------------
    def run_forever(self):
        """Stream and reconnect with backoff until /capture/quit, Ctrl-Break or SIGTERM; every exit goes through
        close(), so the morning open() finds the camera free."""
        try:
            while not self._quit:
                try:
                    self.open()
                    self.attempt = 0
                    self._stream_until_stall()
                except Exception as e:  # noqa: BLE001
                    self.log(f"ERROR {type(e).__name__}: {e}")
                self.close()
                if self._quit:
                    break
                wait = self.backoff_s[min(self.attempt, len(self.backoff_s) - 1)]
                self.attempt += 1
                self.log(f"reconnecting in {wait}s")
                for _ in range(int(wait)):     # heartbeat once a second while waiting: the supervisor must see us alive
                    self._poll_ctrl()
                    if self._quit:
                        break
                    self.send_hb()
                    self._sleep(1.0)
        except KeyboardInterrupt:
            self._quit = True
            self.close()
        if self.ctrl is not None:
            self.ctrl.stop()
        self.log("stopped")

    def _stream_until_stall(self):
        last_frame = self._clock()
        last_hb = last_intr = 0.0
        last_t = None
        while True:
            self._poll_ctrl()
            if self._quit:
                return
            fr = self.next_frame(500)
            now = self._clock()
            if fr is None:
                if now - last_frame > self.no_frame_timeout_s:
                    raise RuntimeError(f"no depth frame for {self.no_frame_timeout_s}s")
                # keep the heartbeat going while frames are missing: a supervisor's stale-heartbeat rule must not
                # kill a capture that is about to reconnect by itself
                if now - last_frame > 1.0:
                    self.connected, self.fps_ema, self.valid_px = False, 0.0, 0
                    self._set_phase("stalled")
                if now - last_hb > 1.0:
                    self.send_hb()
                    last_hb = now
                continue
            depth, ts = fr
            last_frame = now
            self.connected = True
            self._set_phase("streaming")
            if last_t is not None and now > last_t:
                inst = 1.0 / (now - last_t)
                self.fps_ema = inst if self.fps_ema == 0 else 0.9 * self.fps_ema + 0.1 * inst
            last_t = now
            self.valid_px = int(np.count_nonzero(depth[::self.stride, ::self.stride]))
            self.send(depth, ts)
            if now - last_intr > 1.0:
                self.send_intrinsics()
                last_intr = now
            if now - last_hb > 1.0:
                self.send_hb()
                last_hb = now


# ---- probe --------------------------------------------------------------------
def probe(cap: Capture, seconds: float, roi_frac: float = 0.2):
    cap.open()
    t0 = time.monotonic()
    frames, stamps = [], []
    while time.monotonic() - t0 < seconds:
        fr = cap.next_frame(1000)
        if fr is None:
            continue
        depth, ts = fr
        frames.append(depth.copy())
        stamps.append((time.monotonic(), ts))
        cap.send(depth, ts)
    cap.close()
    if len(frames) < 5:
        print(json.dumps({"cmd": "probe", "cam": cap.ip, "ok": False, "reason": f"only {len(frames)} frames"}))
        return False
    D = np.stack(frames).astype(np.float32)          # (T,h,w) mm
    T, h, w = D.shape
    wall = np.array([s[0] for s in stamps])
    fps = (T - 1) / (wall[-1] - wall[0])
    valid = (D > 0).mean()
    r = int(min(h, w) * roi_frac / 2)
    roi = D[:, h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]
    roi_valid = (roi > 0).all(axis=0)
    tnoise = roi[:, roi_valid].std(axis=0)             # per-pixel temporal std, mm
    dev_ts = np.array([s[1] for s in stamps])
    res = {
        "cmd": "probe", "cam": cap.ip, "serial": cap.serial, "ok": bool(fps >= 25 and valid > 0.3),
        "frames": T, "fps": round(float(fps), 2), "res": [w, h],
        "valid_frac": round(float(valid), 3),
        "roi_px": int(roi_valid.sum()), "roi_mean_mm": round(float(roi[:, roi_valid].mean()), 1) if roi_valid.any() else None,
        "roi_temporal_noise_mm_median": round(float(np.median(tnoise)), 2) if tnoise.size else None,
        "roi_temporal_noise_mm_p90": round(float(np.percentile(tnoise, 90)), 2) if tnoise.size else None,
        "device_ts_span_s": round(float(dev_ts[-1] - dev_ts[0]), 3),
        "datagram_bytes": P.depth_datagram_size(w, h, cap.stride),
        "intrinsics": cap.intr.__dict__ if cap.intr else None,
    }
    print(json.dumps(res, indent=1))
    return res["ok"]


def discover(timeout_s: float = 3.0) -> list[dict]:
    """List Orbbec devices reachable on the network (and USB). A new camera ships with a factory static IP and
    DHCP off (see its manual), so put the adapter on that subnet first; two new cameras answer on the same IP
    until one is changed (plug them in one at a time)."""
    from pyorbbecsdk import Context
    ctx = Context()
    ctx.enable_net_device_enumeration(True)
    time.sleep(timeout_s)
    lst = ctx.query_devices()
    out = []
    for i in range(lst.get_count()):
        row = {"index": i}
        for key, fn in (("name", lst.get_device_name_by_index), ("serial", lst.get_device_serial_number_by_index),
                        ("connection", lst.get_device_connection_type_by_index), ("ip", lst.get_device_ip_address_by_index),
                        ("netmask", lst.get_device_subnet_mask_by_index), ("gateway", lst.get_device_gateway_by_index)):
            try:
                row[key] = str(fn(i))
            except Exception as e:  # noqa: BLE001
                row[key] = f"n/a ({type(e).__name__})"
        out.append(row)
    print(json.dumps({"cmd": "discover", "count": len(out), "devices": out,
                      "local": {k: str(v) for k, v in (("ip", _safe(lst.get_local_ip)), ("if", _safe(lst.get_local_net_interface_name))) if v is not None}}, indent=1))
    return out


def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def set_ip(ip: str, port: int, new_ip: str, netmask: str = "255.255.255.0", gateway: str | None = None):
    """Best-effort static IP change via the SDK (writes camera flash; the camera reboots and the call may
    'fail' because the reply never arrives). OrbbecViewer > Advanced > IP Address Config is the documented way."""
    from pyorbbecsdk import Context, OBDeviceIpAddrConfig
    gateway = gateway or new_ip.rsplit(".", 1)[0] + ".1"
    ctx = Context()
    dev = ctx.create_net_device(ip, port)
    cfg = OBDeviceIpAddrConfig()
    try:
        cfg.dhcp = False
        cfg.address = [int(v) for v in new_ip.split(".")]
        cfg.netmask = [int(v) for v in netmask.split(".")]
        cfg.gateway = [int(v) for v in gateway.split(".")]
        dev.set_ip_config(cfg)
        print(json.dumps({"cmd": "set-ip", "ok": True, "new_ip": new_ip, "note": "camera reboots; reconnect on the new IP"}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"cmd": "set-ip", "ok": False, "error": f"{type(e).__name__}: {e}",
                          "note": "use OrbbecViewer > Advanced > IP Address Config instead"}))


def set_sync(ip: str, port: int, mode: str, delay_us: int):
    from pyorbbecsdk import Context, OBMultiDeviceSyncMode
    ctx = Context()
    dev = ctx.create_net_device(ip, port)
    cfg = dev.get_multi_device_sync_config()
    print("current sync config:", cfg)
    cfg.mode = getattr(OBMultiDeviceSyncMode, mode.upper())
    cfg.depth_delay_us = int(delay_us) if mode.upper() in ("SECONDARY", "SECONDARY_SYNCED") else 0
    cfg.color_delay_us = 0
    cfg.trigger_out_enable = mode.upper() == "PRIMARY"
    cfg.trigger_out_delay_us = 0
    cfg.frames_per_trigger = 1
    dev.set_multi_device_sync_config(cfg)
    print("written to flash (device may reboot):", dev.get_multi_device_sync_config())
    if mode.upper() != "FREE_RUN":
        print("note: start every secondary before the primary (a secondary emits no frames until triggered)")


def parse_outs(spec: str) -> list[tuple[str, int]]:
    """host:port[,host:port] -> [(host, port), ...]"""
    out = []
    for item in spec.split(","):
        h, port = item.strip().rsplit(":", 1)
        out.append((h, int(port)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=None, help="camera IP (required unless --discover)")
    ap.add_argument("--discover", action="store_true", help="list network/USB Orbbec devices and exit")
    ap.add_argument("--set-ip", default=None, metavar="NEW_IP", help="write a static IP to the camera at --ip (best effort)")
    ap.add_argument("--gateway", default=None, help="--set-ip: gateway (default: .1 on the new IP's /24)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--cam-id", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--mode", default=DEFAULT_MODE)
    ap.add_argument("--res", default=f"{DEFAULT_RES[0]}x{DEFAULT_RES[1]}")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--out", default=f"127.0.0.1:{P.PORT_DEPTH}", help="host:port[,host:port ...]: the tracker, plus a mirror for viz --live")
    ap.add_argument("--hb", default=f"127.0.0.1:{P.PORT_HB}")
    ap.add_argument("--ctrl", type=int, default=0, help="loopback port for /capture/quit (a supervisor's clean stop); 0 = none")
    ap.add_argument("--probe", type=float, default=0.0, metavar="SECONDS")
    ap.add_argument("--set-sync", choices=["free_run", "primary", "secondary", "secondary_synced", "standalone"],
                    help="write the multi-device sync mode to camera flash (Sync Hub: primary / secondary_synced)")
    ap.add_argument("--delay-us", type=int, default=160)
    a = ap.parse_args(argv)

    if a.discover:
        return 0 if discover() else 1
    if not a.ip:
        ap.error("--ip is required (or use --discover)")
    if a.set_ip:
        set_ip(a.ip, a.port, a.set_ip, gateway=a.gateway)
        return 0
    if a.set_sync:
        set_sync(a.ip, a.port, a.set_sync, a.delay_us)
        return 0
    w, h = (int(v) for v in a.res.lower().split("x"))
    hh, hp = a.hb.split(":")
    cap = Capture(a.ip, a.cam_id, a.port, a.mode, (w, h), a.fps, a.stride, parse_outs(a.out), (hh, int(hp)), a.name,
                  ctrl_port=a.ctrl or None)
    if a.probe > 0:
        return 0 if probe(cap, a.probe) else 1
    install_quit_signals()
    cap.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
