# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""rt_depth_tracker.capture without a camera: a stub pyorbbecsdk stands in for the SDK so the real open() runs.
Covers the heartbeat lease around the GIL-holding SDK calls, the phases, and the reconnect loop."""
import os
import queue
import sys
import types

import numpy as np

import _scene  # noqa: E402,F401

SDK = types.SimpleNamespace(clk=None, connects=0, fail_connects=0, connect_takes_s=0.0, stops=0)


class _Info:
    get_serial_number = lambda self: "SN1"          # noqa: E731
    get_name = lambda self: "Femto Mega"            # noqa: E731
    get_firmware_version = lambda self: "1.2.7"     # noqa: E731


class _Dev:
    get_device_info = lambda self: _Info()          # noqa: E731
    get_depth_work_mode = lambda self: "NFOV 2x2 binned"   # noqa: E731

    def get_depth_work_mode_list(self):
        raise RuntimeError("stub")


class _Prof:
    get_width = lambda self: 320                    # noqa: E731
    get_height = lambda self: 288                   # noqa: E731
    get_fps = lambda self: 30                       # noqa: E731
    get_format = lambda self: "Y16"                 # noqa: E731

    def get_intrinsic(self):
        raise RuntimeError("stub")


class _PList:
    get_video_stream_profile = lambda self, *a: _Prof()            # noqa: E731
    get_default_video_stream_profile = lambda self: _Prof()        # noqa: E731


class _Pipe:
    def __init__(self, dev):
        pass

    get_stream_profile_list = lambda self, t: _PList()             # noqa: E731

    def start(self, cfg):
        pass

    def stop(self):
        SDK.stops += 1


class _Ctx:
    def create_net_device(self, ip, port):
        SDK.connects += 1
        SDK.clk.t += SDK.connect_takes_s           # the real call blocks (GIL held) this long
        if SDK.connects <= SDK.fail_connects:
            raise RuntimeError("VendorTCPClient: Connect to server failed")
        return _Dev()


sdk = types.ModuleType("pyorbbecsdk")
sdk.Context, sdk.Pipeline = _Ctx, _Pipe
sdk.Config = lambda: types.SimpleNamespace(enable_stream=lambda p: None)
sdk.OBFormat = types.SimpleNamespace(Y16=1)
sdk.OBSensorType = types.SimpleNamespace(DEPTH_SENSOR=1)
sys.modules["pyorbbecsdk"] = sdk

from rt_depth_tracker import capture as C  # noqa: E402
from rt_depth_tracker import protocol as P  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


class Clock:
    def __init__(self, t=1000.0, stop_at=None):
        self.t, self.stop_at = t, stop_at

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.stop_at is not None and self.t >= self.stop_at:
            raise KeyboardInterrupt


class Sock:
    def __init__(self):
        self.sent = []

    def sendto(self, d, addr):
        self.sent.append((d, addr))


class Fake(C.Capture):
    """The real Capture over the stub SDK; frames come from a counter, never from a pipeline."""

    def __init__(self, clk, frames=0):
        super().__init__("10.0.0.1", 0, out=("127.0.0.1", 7310), hb=("127.0.0.1", 9310), name="capture_t",
                         clock=clk, sleep=clk.sleep)
        self.clk, self.frames_left = clk, frames
        self.sock = Sock()
        self.log = lambda m: None

    def next_frame(self, timeout_ms=1000):
        if self.frames_left > 0:
            self.frames_left -= 1
            self.clk.t += 1 / 30
            return np.full((288, 320), 1500, np.uint16), self.clk.t
        self.clk.t += timeout_ms / 1000.0
        if self.clk.stop_at is not None and self.clk.t >= self.clk.stop_at:
            raise KeyboardInterrupt
        return None


def run(clk, frames, fail_connects, connect_takes_s):
    SDK.clk, SDK.connects, SDK.fail_connects, SDK.connect_takes_s = clk, 0, fail_connects, connect_takes_s
    cap = Fake(clk, frames)
    try:
        cap.run_forever()
    except KeyboardInterrupt:
        pass
    return cap, [P.unpack(d).metrics for d, addr in cap.sock.sent if addr == cap.hb_addr]


def leases(H, phase):
    return [i for i, m in enumerate(H) if m.get("phase") == phase and "busy_s" in m]


def gaps(H):
    """Plain heartbeats (no lease) between consecutive 'opening' leases: the backoff waits, one per second."""
    op = leases(H, "opening")
    return [sum(1 for m in H[a + 1:b] if "busy_s" not in m) for a, b in zip(op, op[1:])]


print("the lease: a heartbeat before every GIL-holding SDK call")
clk = Clock(1000.0, stop_at=1200.0)
cap, H = run(clk, frames=40, fail_connects=4, connect_takes_s=14.0)
h0 = H[0]
check("first heartbeat goes out before open(): phase opening, busy_s 45, connected false, fps 0",
      h0.get("phase") == "opening" and h0.get("busy_s") == C.OPEN_LEASE_S and h0.get("connected") is False and h0.get("fps") == 0.0, str(h0))
check("every heartbeat carries phase, phase_s and attempt", all({"phase", "phase_s", "attempt"} <= set(m) for m in H))
check("open() failing 4 times: backoff 1, 2, 4, 8 with a heartbeat per second of it", gaps(H)[:4] == [1, 2, 4, 8], str(gaps(H)))
check("attempt counts up on each opening lease", [H[i]["attempt"] for i in leases(H, "opening")[:5]] == [0, 1, 2, 3, 4],
      str([H[i]["attempt"] for i in leases(H, "opening")]))
rec = [m for m in H if m.get("phase") == "reconnecting"]
check("after a failed open the capture is reconnecting: lease 15 s before pipe.stop, then plain heartbeats",
      rec and rec[0].get("busy_s") == C.CLOSE_LEASE_S and "busy_s" not in rec[1], str(rec[:2]))
second = [i for i in leases(H, "opening") if H[i]["phase_s"] > 0]
check("a successful open sends a second lease after the device answers (the work-mode calls and pipe.start block too)",
      second and H[second[0]]["phase_s"] == 14.0 and H[second[0]]["connected"] is False, str([H[i] for i in second[:1]]))
st = [i for i, m in enumerate(H) if m.get("phase") == "streaming"]
check("then phase streaming, connected true, no lease", st and H[st[0]]["connected"] is True and "busy_s" not in H[st[0]], str(H[st[0]] if st else H[-3:]))
check("the stream's own heartbeat has attempt 0 again", H[st[1]]["attempt"] == 0, str(H[st[1]]))
check("fps ~30 and valid_px > 0 after a second of frames", any(m["phase"] == "streaming" and m["fps"] > 25 and m["valid_px"] == 144 * 160 for m in H),
      str([(m["fps"], m["valid_px"]) for m in H if m["phase"] == "streaming"]))
stalled = [i for i, m in enumerate(H) if m.get("phase") == "stalled"]
check("frames stop: after 1 s connected false, fps 0, valid_px 0, phase stalled, heartbeat still flowing",
      stalled and H[stalled[0]]["connected"] is False and H[stalled[0]]["fps"] == 0.0 and H[stalled[0]]["valid_px"] == 0, str(H[stalled[0]] if stalled else H[-3:]))
after = [m for m in H[stalled[0] + 1:] if m["phase"] != "stalled"][:3] if stalled else []
check("after 3 s without frames the stream ends and the loop reconnects: close lease, one backoff heartbeat, open lease",
      len(after) == 3 and after[0]["phase"] == "reconnecting" and after[0].get("busy_s") == C.CLOSE_LEASE_S
      and after[0]["fps"] == 0.0 and after[0]["valid_px"] == 0 and after[1]["phase"] == "reconnecting" and "busy_s" not in after[1]
      and after[2]["phase"] == "opening" and after[2].get("busy_s") == C.OPEN_LEASE_S, str(after))
check("a streaming heartbeat never carries busy_s", all("busy_s" not in m for m in H if m["phase"] == "streaming"))
check("seq keeps counting across an in-process reconnect", H[-1]["seq"] == 40 and cap.seq == 40, str((H[-1]["seq"], cap.seq)))
changes = [m for prev, m in zip(H, H[1:]) if m["phase"] != prev["phase"]]
check("phase_s restarts at 0 on every phase change (a stall shows on the next 1 s heartbeat)",
      changes and all(m["phase_s"] == 0.0 for m in changes if m["phase"] != "stalled") and all(m["phase_s"] <= 1.0 for m in changes if m["phase"] == "stalled"),
      str([(m["phase"], m["phase_s"]) for m in changes]))
check("the stall is reported in the heartbeat before the raise", any(m["phase"] == "stalled" for m in H))

print("backoff cap")
clk2 = Clock(1000.0, stop_at=1400.0)
cap2, H2 = run(clk2, frames=0, fail_connects=12, connect_takes_s=0.0)
check("backoff 1, 2, 4, 8, 15, 30 then stays at 30", gaps(H2)[:8] == [1, 2, 4, 8, 15, 30, 30, 30], str(gaps(H2)))
check("still heartbeating once a second through all of it: no gap over 1 s between plain heartbeats", len(H2) > 100, str(len(H2)))

print("/capture/quit: a supervisor's clean stop")
clk3 = Clock(1000.0)
cap3 = Fake(clk3, frames=10 ** 6)
cap3.ctrl = types.SimpleNamespace(queue=queue.Queue(), stop=lambda: setattr(SDK, "ctrl_stopped", True))
calls3 = [0]
_next3 = cap3.next_frame


def _next_then_quit(timeout_ms=1000):
    calls3[0] += 1
    if calls3[0] == 30:                        # the quit lands while streaming
        cap3.ctrl.queue.put((P.ADDR_CAPTURE_QUIT, ()))
    return _next3(timeout_ms)


cap3.next_frame = _next_then_quit
SDK.clk, SDK.connects, SDK.fail_connects, SDK.connect_takes_s, SDK.stops = clk3, 0, 0, 0.0, 0
cap3.run_forever()
H3 = [P.unpack(d).metrics for d, addr in cap3.sock.sent if addr == cap3.hb_addr]
check("run_forever returned on /capture/quit: pipeline stopped once, every SDK handle dropped",
      cap3._quit and SDK.stops == 1 and not any(k in cap3.__dict__ for k in ("pipe", "dev", "ctx")),
      str((SDK.stops, [k for k in ("pipe", "dev", "ctx") if k in cap3.__dict__])))
check("the last heartbeat says closing, connected false, fps 0 (not reconnecting: nothing restarts this)",
      H3[-1]["phase"] == "closing" and H3[-1]["connected"] is False and H3[-1]["fps"] == 0.0, str(H3[-1]))
check("the ctrl socket is closed", getattr(SDK, "ctrl_stopped", False))
clk4 = Clock(1000.0)
cap4 = Fake(clk4, frames=0)
cap4.ctrl = types.SimpleNamespace(queue=queue.Queue(), stop=lambda: None)
SDK.clk, SDK.connects, SDK.fail_connects, SDK.connect_takes_s, SDK.stops = clk4, 0, 100, 0.0, 0
sleeps4 = [0]


def _sleep_then_quit(s):
    sleeps4[0] += 1
    clk4.t += s
    if sleeps4[0] == 5:                        # the quit lands during the third backoff wait
        cap4.ctrl.queue.put((P.ADDR_CAPTURE_QUIT, ()))


cap4._sleep = _sleep_then_quit
cap4.run_forever()
check("a quit during the reconnect backoff ends the loop at once: no further open, no 30 s wait",
      cap4._quit and sleeps4[0] == 5 and SDK.connects == 3, str((sleeps4[0], SDK.connects)))

print(f"test_capture: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
