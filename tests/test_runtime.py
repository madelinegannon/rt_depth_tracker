# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""TrackerRuntime over real loopback sockets: synthetic /depth + /intrinsics + /robot/q in,
/tracker/target + /tracker/tracks + /hb out, learn_bg control, bg persistence, config reload; a second runtime
without a robot (no /robot/q listener) and with a zone (/tracker/zone out, and only then)."""
import json
import os
import socket
import sys
import tempfile
import threading
import time

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import calib as C  # noqa: E402
from rt_depth_tracker import protocol as P  # noqa: E402
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel  # noqa: E402
from rt_depth_tracker.run import TrackerRuntime  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


IN, OUT, OUT2, Q, CTRL, HB = 7510, 7501, 7502, 7511, 7512, 9500
tmp = tempfile.mkdtemp()
calib_dir = os.path.join(tmp, "calib")
os.makedirs(calib_dir)
cams = S.two_camera_rig()
for cid, T in cams.items():
    C.save_sensor_calib(os.path.join(calib_dir, f"sensor_{'ab'[cid]}.json"), cid, T)
cfg_path = os.path.join(tmp, "tracker.json")
json.dump({"bg_learn_frames": 20, **_scene.SCENE}, open(cfg_path, "w"))

rt = TrackerRuntime(cfg_path, calib_dir, IN, [("127.0.0.1", OUT), ("127.0.0.1", OUT2)], Q, CTRL, ("127.0.0.1", HB), record_dir=os.path.join(tmp, "rec"), log=lambda m: None)
check("calib files loaded", len(rt.pipe.cams) == 2 and all(rt.pipe.cam(c).T_ws is not None for c in (0, 1)))
check("a robot in the config -> a /robot/q listener", rt.pipe.robot is not None and rt.robot is not None)
rt_thread = threading.Thread(target=rt.run, daemon=True)
rt_thread.start()

out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
out_sock.bind(("127.0.0.1", OUT))
out_sock.settimeout(2.0)
out2_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
out2_sock.bind(("127.0.0.1", OUT2))
out2_sock.settimeout(2.0)
hb_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hb_sock.bind(("127.0.0.1", HB))
hb_sock.settimeout(1.5)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rm = RobotModel.from_config(_scene.cfg())
scene = S.Scene(rail_radius=1.4, q=np.array([0, 0.26, -0.9, 0, 0.6, 0]), robot=rm)
rng = np.random.default_rng(0)
seq = 0


def send_frame(t):
    global seq
    seq += 1
    for c in cams:
        d = S.render(scene, cams[c], rng=rng)
        tx.sendto(P.pack_intrinsics(S.intrinsics_for(c)), ("127.0.0.1", IN))
        for dg in P.pack_depth_packet(S.packet(c, d, seq, t)):   # render() output is already strided
            tx.sendto(dg, ("127.0.0.1", IN))
    tx.sendto(P.pack_robot_q(scene.q), ("127.0.0.1", Q))


zone_msgs: list = []          # (socket port, ZoneMsg) for every /tracker/zone any drain ever saw


def drain(sock, want, n=50):
    got = None
    for _ in range(n):
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            break
        obj = P.unpack(data)
        if isinstance(obj, P.ZoneMsg):
            zone_msgs.append((sock.getsockname()[1], obj))
            if want == "zone":
                got = obj
        if want == "target" and isinstance(obj, P.Target):
            got = obj
        elif want == "tracks" and isinstance(obj, np.ndarray):
            got = obj
        elif want == "hb" and isinstance(obj, P.Heartbeat):
            got = obj          # keep draining: the newest heartbeat is the one that matters
    return got


def feed_until(runtime, sender, t, done, budget_s=60.0):
    """Stream frames until done(): the next frame goes out only once the runtime has stepped the previous one
    (or 1 s has passed), so a slow machine is paced instead of flooded, and the budget is wall-clock, not a
    frame count."""
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        n = runtime._frames
        sender(t)
        t += 1 / 30
        step_by = time.monotonic() + 1.0
        while runtime._frames == n and time.monotonic() < step_by:
            time.sleep(0.005)
        if done():
            break
    return t


def diag(runtime):
    bg = {c: (cs.bg.ready, cs.bg.learning, len(getattr(cs.bg, "_buf", []))) for c, cs in runtime.pipe.cams.items()}
    ms = runtime._step_ms[-30:]
    mean_ms = sum(ms) / len(ms) if ms else float("nan")
    return (f"frames={runtime._frames} step_errors={runtime._step_errors} rx_hz={runtime.rx.rates()} "
            f"bg(ready,learning,buffered)={bg} step_ms~{mean_ms:.1f}")


print("learn background via control message")
tx.sendto(P.pack_hb.__globals__["_msg"](P.ADDR_LEARN_BG), ("127.0.0.1", CTRL))
time.sleep(0.2)
t = feed_until(rt, send_frame, 0.0, lambda: all(rt.pipe.cam(c).bg.ready for c in (0, 1)))
time.sleep(0.5)
check("bg learned + saved", all(rt.pipe.cam(c).bg.ready for c in (0, 1)) and os.path.exists(os.path.join(calib_dir, "bg_0.npz")), diag(rt))

print("target output")
drain(out_sock, "target", 200)
scene.people = [S.Person((2.0, 0.3))]
tgt = None


def _person_seen():
    global tgt
    got = drain(out_sock, "target", 20)
    if got is not None and got.hint in ("watch", "engage"):
        tgt = got
    return tgt is not None


out_sock.settimeout(0.2)
t = feed_until(rt, send_frame, t, _person_seen)
out_sock.settimeout(2.0)
tgt2 = drain(out2_sock, "target", 100)
check("mirror address receives the same target stream", isinstance(tgt2, P.Target) and (tgt is None or tgt2.hint == tgt.hint), str(tgt2))
check("/tracker/target received with a person", tgt is not None and tgt.hint in ("watch", "engage") and abs(tgt.x - 2.0) < 0.3, f"{tgt} {diag(rt)}")
rows = drain(out_sock, "tracks", 100)
check("/tracker/tracks received", rows is None or (isinstance(rows, np.ndarray) and rows.shape[1] == 8))

print("heartbeat")
hb = None


def _hb_sees_target():
    global hb
    got = drain(hb_sock, "hb", 5)
    if got is not None and got.metrics.get("targets", 0) >= 1:
        hb = got
    return hb is not None


hb_sock.settimeout(0.2)                           # heartbeats come once a second: keep the person in view until one reflects it
t = feed_until(rt, send_frame, t, _hb_sees_target)
hb_sock.settimeout(1.5)
check("hb from rt_depth_tracker", hb is not None and hb.name == "tracker" and hb.metrics["targets"] >= 1 and 0 in hb.metrics["cams_ready"], f"{hb and hb.metrics} {diag(rt)}")
check("hb rx rates present", hb is not None and set(map(int, hb.metrics["rx_hz"].keys())) == {0, 1})

print("config reload")
json.dump({"bg_learn_frames": 20, "interest_model": False}, open(cfg_path, "w"))
os.utime(cfg_path, None)
time.sleep(1.3)
send_frame(t)
time.sleep(0.2)
check("interest_model toggled by file edit", rt.cfg.interest_model is False)

print("live tuning via control message (/tracker/set)")
tx.sendto(P._msg(P.ADDR_SET, ("near_pref", P.S), (0.9, P.F)), ("127.0.0.1", CTRL))
tx.sendto(P._msg(P.ADDR_SET, ("mount_z", P.S), (2.0, P.F)), ("127.0.0.1", CTRL))
time.sleep(0.4)
check("a tunable key applies at once", abs(rt.cfg.near_pref - 0.9) < 1e-6, str(rt.cfg.near_pref))
check("a key that is not tunable is ignored", rt.cfg.mount_z != 2.0, str(rt.cfg.mount_z))
time.sleep(2.5)                                                       # the drag settles: one write
d_cfg = json.load(open(cfg_path))
check("the settled value is saved beside the existing keys", d_cfg.get("near_pref") == 0.9 and d_cfg.get("bg_learn_frames") == 20, str(d_cfg))
hb = drain(hb_sock, "hb", 50)                                        # past the backlog to a fresh one
check("hb carries the live tune values", hb is not None and hb.metrics.get("tune", {}).get("near_pref") == 0.9, str(hb and hb.metrics.get("tune")))
d_cfg["interest_model"] = True
json.dump(d_cfg, open(cfg_path, "w"))
os.utime(cfg_path, None)
time.sleep(1.3)
check("a later file edit still reloads (the save did not wedge the mtime watch)",
      rt.cfg.interest_model is True and abs(rt.cfg.near_pref - 0.9) < 1e-6, f"{rt.cfg.interest_model} {rt.cfg.near_pref}")

print("recording control")
tx.sendto(P.pack_hb.__globals__["_msg"](P.ADDR_RECORD, ("walk_by", P.S)), ("127.0.0.1", CTRL))
time.sleep(0.2)
for i in range(5):
    send_frame(t)
    t += 1 / 30
    time.sleep(0.03)
tx.sendto(P.pack_hb.__globals__["_msg"](P.ADDR_RECORD, (0, P.I)), ("127.0.0.1", CTRL))
time.sleep(0.5)
recs = [f for f in os.listdir(os.path.join(tmp, "rec"))] if os.path.exists(os.path.join(tmp, "rec")) else []
check("tagged clip written", len(recs) == 1 and recs[0].startswith("walk_by_"), str(recs))

print("replay --loop")
from rt_depth_tracker import replay as R  # noqa: E402
clip = os.path.join(tmp, "rec", recs[0])
st1 = R.run(R.parse([clip, "--config", cfg_path, "--calib-dir", calib_dir]))
st2 = R.run(R.parse([clip, "--config", cfg_path, "--calib-dir", calib_dir, "--loop", "2"]))
check("single pass plays every packet", st1["frames"] > 0 and st1["monotonic"], str(st1))
check("--loop 2 plays twice with a continuous clock", st2["frames"] == 2 * st1["frames"] and st2["monotonic"] and st2["t_span"] > st1["t_span"], str(st2))

print("a config without a robot: no /robot/q listener; /tracker/zone only with zones configured")
IN2, OUTZ, Q2, CTRL2, HB2 = 7520, 7503, 7521, 7522, 9501
tmp2 = tempfile.mkdtemp(prefix="rt_rt2_")
calib2 = os.path.join(tmp2, "calib")
os.makedirs(calib2)
for cid, T in cams.items():
    C.save_sensor_calib(os.path.join(calib2, f"sensor_{'ab'[cid]}.json"), cid, T)
cfg2_path = os.path.join(tmp2, "tracker.json")
json.dump({"bg_learn_frames": 20, "zones": [{"name": "desk", "rect": [1.7, -0.3, 2.3, 0.3], "top_z": 0.9}]}, open(cfg2_path, "w"))
rt2 = TrackerRuntime(cfg2_path, calib2, IN2, [("127.0.0.1", OUTZ)], Q2, CTRL2, ("127.0.0.1", HB2), record_dir=os.path.join(tmp2, "rec"), log=lambda m: None)
check("no robot block -> no RobotModel and no /robot/q listener", rt2.pipe.robot is None and rt2.robot is None)
th2 = threading.Thread(target=rt2.run, daemon=True)
th2.start()
outz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
outz_sock.bind(("127.0.0.1", OUTZ))
outz_sock.settimeout(2.0)
hb2_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hb2_sock.bind(("127.0.0.1", HB2))
hb2_sock.settimeout(1.5)
scene2 = S.Scene(table={"center_xy": [2.0, 0.0], "size_xy": [0.6, 0.6], "top_z": 0.9})       # a desk, no robot
seq2 = 0


def send_frame2(t):
    global seq2
    seq2 += 1
    for c in cams:
        d = S.render(scene2, cams[c], rng=rng)
        tx.sendto(P.pack_intrinsics(S.intrinsics_for(c)), ("127.0.0.1", IN2))
        for dg in P.pack_depth_packet(S.packet(c, d, seq2, t)):
            tx.sendto(dg, ("127.0.0.1", IN2))


tx.sendto(P.pack_hb.__globals__["_msg"](P.ADDR_LEARN_BG), ("127.0.0.1", CTRL2))
time.sleep(0.2)
t2 = feed_until(rt2, send_frame2, 0.0, lambda: all(rt2.pipe.cam(c).bg.ready for c in (0, 1)))
time.sleep(0.5)
check("rt2: bg learned without a robot", all(rt2.pipe.cam(c).bg.ready for c in (0, 1)), diag(rt2))
drain(outz_sock, "zone", 200)
scene2.people = [S.Person((2.0, -0.65))]                     # standing at the desk's near edge
zm = None


def _desk_occupied():
    global zm
    got = drain(outz_sock, "zone", 20)
    if isinstance(got, P.ZoneMsg) and got.occupied == 1:
        zm = got
    return zm is not None


outz_sock.settimeout(0.2)
t2 = feed_until(rt2, send_frame2, t2, _desk_occupied)
outz_sock.settimeout(2.0)
check("/tracker/zone received: the desk is occupied by a track", isinstance(zm, P.ZoneMsg) and zm.name == "desk" and zm.occupied == 1 and zm.track_id >= 0, f"{zm} {diag(rt2)}")
check("the first runtime (no zones) never sent one", not any(port in (OUT, OUT2) for port, _ in zone_msgs), str({p for p, _ in zone_msgs}))
hb2 = None


def _hb2_sees_desk():
    global hb2
    got = drain(hb2_sock, "hb", 5)
    if got is not None and got.metrics.get("zones", {}).get("desk") == "occupied":
        hb2 = got
    return hb2 is not None


hb2_sock.settimeout(0.2)
t2 = feed_until(rt2, send_frame2, t2, _hb2_sees_desk)
hb2_sock.settimeout(1.5)
check("rt2 hb: no q_age, zone state carried", hb2 is not None and hb2.metrics["q_age"] is None and hb2.metrics["zones"].get("desk") == "occupied", str(hb2 and hb2.metrics))
tx.sendto(P.pack_hb.__globals__["_msg"](P.ADDR_TRACKER_QUIT), ("127.0.0.1", CTRL2))
th2.join(timeout=3)
rt2.stop()

print("/tracker/quit: a supervisor's clean stop")
tx.sendto(P.pack_hb.__globals__["_msg"](P.ADDR_TRACKER_QUIT), ("127.0.0.1", CTRL))
rt_thread.join(timeout=3)
check("a /tracker/quit ends run() cleanly: loop over, thread gone", not rt_thread.is_alive() and rt._running is False)

rt.stop()
print(f"test_runtime: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
