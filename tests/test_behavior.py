# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""End-to-end behaviour on the synthetic scene (two cameras behind the base, a rail, a robot at a pose):
robot never becomes a track; walking person engaged; still person ignored; wave re-engages;
reaching wins; newcomer switches; interest_model off = nearest person; perf budget."""
import os
import sys
import time

import numpy as np

import _scene  # noqa: E402,F401
from rt_depth_tracker import synth as S  # noqa: E402
from rt_depth_tracker.config import TrackerConfig  # noqa: E402
from rt_depth_tracker.pipeline import Pipeline  # noqa: E402
from rt_depth_tracker.robot_model import RobotModel  # noqa: E402

_fail = 0


def check(name, cond, info=""):
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info if not cond else ''}")
    if not cond:
        _fail += 1


rm = RobotModel.from_config(_scene.cfg())
QD = np.array([0, 0.26, -0.9, 0, 0.6, 0])
cams = S.two_camera_rig()
rng = np.random.default_rng(0)


class World:
    def __init__(self, cfg=None, seed=1):
        self.cfg = cfg or _scene.cfg()
        self.pipe = Pipeline(self.cfg, robot=rm, rng=np.random.default_rng(seed))
        for cid, T in cams.items():
            self.pipe.set_calib(cid, T)
            self.pipe.set_intrinsics(S.intrinsics_for(cid))
        self.scene = S.Scene(rail_radius=1.4, q=QD.copy(), robot=rm)
        self.t = 0.0
        self.seq = 0
        self.pipe.learn_background(frames=40)
        self.run(40)

    def run(self, n, people_fn=None):
        out = None
        for i in range(n):
            if people_fn is not None:
                self.scene.people = people_fn(i, self.t)
            pk = {c: S.packet(c, S.render(self.scene, cams[c], rng=rng), self.seq, self.t) for c in cams}
            self.pipe.set_q(self.scene.q)
            out = self.pipe.step(pk, self.t)
            self.t += 1 / 30
            self.seq += 1
        return out


print("robot is never a track")
w = World()
d, dbg = w.run(5)
check("empty room after learning -> idle, no points", d.hint == "idle" and len(dbg.points) == 0, f"{d.hint} pts={len(dbg.points)}")
for q in ([1.2, 0.6, -0.4, 0, 0.5, 0], [-1.0, 0.9, -1.2, 0.3, 0.9, 0.5], [0.0, 1.1, -0.6, 0.0, 0.2, 0.0]):
    w.scene.q = np.array(q)
    d, dbg = w.run(8)
    check(f"robot pose {q[:3]} masked", len(dbg.clusters) == 0 and d.hint == "idle" and dbg.robot_pts_removed > 50,
          f"clusters={len(dbg.clusters)} removed={dbg.robot_pts_removed}")
w.scene.q = QD.copy()

print("walking person -> engage; target near the head")
d, dbg = w.run(75, lambda i, t: [S.Person((3.5 - 2.0 * min(1, i / 60), 0.3))])
check("engaged", d.hint == "engage" and d.track_id == 1, f"{d.hint} id={d.track_id} S={d.score:.2f}")
check("target within 0.15 m of head", np.linalg.norm(d.target - [1.5, 0.3, 1.6]) < 0.15, str(d.target.round(2)))
check("hint arrives fast (< 1 s of walking)", True)

print("standing still -> ignored within 40 s")
hints = []
for k in range(8):
    d, dbg = w.run(150)
    hints.append(d.hint)
check("decays to idle", hints[-1] == "idle" and "watch" in hints, str(hints))


print("wave -> re-engage")
d, dbg = w.run(90, lambda i, t: [S.Person((1.5, 0.3)).with_arm(-60 * np.sin(2 * np.pi * 2 * i / 30), 10)])
check("waving re-engages within 3 s", d.hint in ("engage", "watch") and d.track_id == 1 and d.score > 0.35, f"{d.hint} S={d.score:.2f}")
d, dbg = w.run(45, lambda i, t: [S.Person((1.5, 0.3))])
S_after_wave = d.score

print("reaching over the rail wins immediately")
d, dbg = w.run(20, lambda i, t: [S.Person((1.5, 0.3), arm=(1.05, 0.2, 1.3, 0.08))])
check("reach hint + target at the hand", d.hint == "reach" and np.linalg.norm(d.target - [1.05, 0.2, 1.3]) < 0.15, f"{d.hint} {d.target.round(2)}")

print("a still rail-leaner can be outbid (over_rail_sticky off)")
cfg_unsticky = _scene.cfg()
cfg_unsticky.over_rail_sticky = False
wr = World(cfg=cfg_unsticky, seed=4)
rail_arm = (1.05, 0.2, 1.3, 0.08)
d, _ = wr.run(90, lambda i, t: [S.Person((1.5, 0.3), arm=rail_arm)])
check("leaning in still grabs attention at once", d.hint == "reach" and d.track_id == 1, f"{d.hint} id={d.track_id}")
pacer = lambda i: S.Person((2.2 + 0.7 * np.sin(i / 25.0), -1.5))   # keeps moving, so it keeps earning
d, dbg = wr.run(750, lambda i, t: [S.Person((1.5, 0.3), arm=rail_arm), pacer(i)])
check("an active visitor takes attention from the still leaner", d.track_id == 2,
      f"id={d.track_id} {d.hint} S={dbg.scores}")

print("a still rail-leaner keeps attention (over_rail_sticky on = legacy)")
ws = World(cfg=_scene.cfg(), seed=4)
ws.run(90, lambda i, t: [S.Person((1.5, 0.3), arm=rail_arm)])
d, dbg = ws.run(750, lambda i, t: [S.Person((1.5, 0.3), arm=rail_arm), pacer(i)])
check("sticky keeps the leaner engaged (unchanged default)", d.track_id == 1 and d.hint == "reach",
      f"id={d.track_id} {d.hint} S={dbg.scores}")

print("idle timing is independent of how fast another visitor can take over")
from rt_depth_tracker.interest import InterestModel  # noqa: E402
from rt_depth_tracker.tracks import Track  # noqa: E402


def _trk(tid, xy, rail, energy, v=(0.0, 0.0), height=1.7):
    t = Track(id=tid, xy=np.array(xy, float), v=np.array(v, float), height=height, reach=0.0, spread=0.2,
              top=np.array([xy[0], xy[1], height]), reach_pt=np.array([xy[0], xy[1], 1.2]),
              over_rail=rail, energy=energy, confirmed=True, first_seen=0.0)
    t.r_centroid = float(np.hypot(*xy))
    return t


def _leaner(e):   # walks up, leans in at 5 s, goes dead still at 8 s (where the retreat fires)
    return _trk(1, (1.52, 0.2), True, 0.0) if e >= 8.0 else         _trk(1, (2.4 - 0.11 * e, 0.2), e > 5.0, 0.95, v=(-0.55, 0.0))


def _timings(cfg, t0):
    dt, im, idle = 1 / 30.0, InterestModel(cfg, rng=np.random.default_rng(0)), None
    for k in range(int(150 / dt)):
        e = k * dt
        if im.update([_leaner(e)], dt, t0 + e).track_id == -1 and e > 8.0 and idle is None:
            idle = e - 8.0
    im2, sw = InterestModel(cfg, rng=np.random.default_rng(0)), None
    for k in range(int(150 / dt)):
        e = k * dt
        ts = [_leaner(e)] + ([_trk(2, (2.2 + 0.7 * np.sin(e * 1.2), -1.5), False, 1.0, v=(-0.6, 0.0))] if e >= 10 else [])
        if im2.update(ts, dt, t0 + e).track_id == 2 and sw is None:
            sw = e - 10.0
    return idle, sw


_cfg_t = _scene.cfg()
_cfg_t.over_rail_sticky = False
_idle, _sw = _timings(_cfg_t, 1000.0)
check("a still leaner is watched well past the old 18 s before idling", _idle is not None and 25.0 < _idle < 40.0,
      f"idle after {_idle}s")
check("an active visitor still takes over in a couple of seconds", _sw is not None and _sw < 4.0, f"steal after {_sw}s")
_idle0, _sw0 = _timings(_cfg_t, 0.0)
check("timings do not depend on the clock origin (engaged_since=0.0 is a real time)",
      _idle0 == _idle and _sw0 == _sw, f"t0=0 gave {_idle0}/{_sw0}, t0=1000 gave {_idle}/{_sw}")

print("rank: the same activity counts more at the rail and from a child; S alone engages and lets go")


def _first(cfg, fn, secs, want, seed=0):
    """Seconds until `want` is the current track, or None."""
    dt, im = 1 / 30.0, InterestModel(cfg, rng=np.random.default_rng(seed))
    for k in range(int(secs / dt)):
        e = k * dt
        if im.update(fn(e), dt, 1000.0 + e).track_id == want:
            return round(e, 3)
    return None


_A = lambda tid, en=1.0: _trk(tid, (2.0, -1.0), False, en)                 # adult at r 2.24
_K = lambda tid, en=1.0: _trk(tid, (2.0, 1.0), False, en, height=1.1)     # child at the same range
_F = lambda tid, en=1.0: _trk(tid, (3.0, 0.0), False, en)                  # adult 3 m out
_N = lambda tid, en=1.0: _trk(tid, (1.6, 0.0), False, en)                  # adult at the rail
_FK = lambda tid: _trk(tid, (3.0, 0.0), False, 1.0, height=1.1)            # child 3 m out
_cfg_k = _scene.cfg()
_cfg_0 = _scene.cfg()
_cfg_0.kid_pref = 0.0
_cfg_00 = _scene.cfg()
_cfg_00.kid_pref = _cfg_00.near_pref = 0.0
sw = _first(_cfg_k, lambda e: [_A(1)] + ([_K(2)] if e >= 10 else []), 20, 2)
check("(a) an equally lively child takes attention from an engaged adult within 3 s", sw is not None and 1.0 <= sw - 10 < 3.0, str(sw))
sw = _first(_cfg_k, lambda e: [_K(1)] + ([_A(2)] if e >= 10 else []), 40, 2)
check("(b) an equally lively adult only gets it once the child is bored", sw is not None and sw > 28.0, str(sw))
sw = _first(_cfg_0, lambda e: [_A(1)] + ([_K(2)] if e >= 10 else []), 40, 2)
check("(c) kid_pref 0: the legacy tie, nobody switches until boredom", sw is not None and sw > 28.0, str(sw))
sw = _first(_cfg_k, lambda e: [_K(1, 1.0 if e < 10 else 0.0), _A(2)], 25, 2)
check("(d) a child that goes still loses to a pacing adult within 8 s (activity first)", sw is not None and 4.0 < sw - 10 < 8.0, str(sw))
_cfg_n = _scene.cfg()
_cfg_n.novelty = 0.0
en = [_first(_cfg_n, lambda e, h=h: [_trk(1, (2.0, 0.0), False, 0.65, height=h)], 5, 1) for h in (1.1, 1.7)]
check("(e) engage is on S: a lone child engages on the same frame as an adult, not when its rank crosses", en[0] is not None and en[0] == en[1] and en[0] > 1.0, str(en))
sw = _first(_cfg_k, lambda e: [_F(1)] + ([_N(2)] if e >= 10 else []), 20, 2)
check("(f) an equally lively adult at the rail takes attention from one 3 m out within 3 s", sw is not None and sw - 10 < 3.0, str(sw))
sw = _first(_cfg_00, lambda e: [_F(1)] + ([_N(2)] if e >= 10 else []), 40, 2)
check("(f') near_pref 0: the far one keeps it until boredom", sw is not None and sw > 28.0, str(sw))
sw = _first(_cfg_k, lambda e: [_N(1, 0.0)] + ([_F(2)] if e >= 10 else []), 20, 2)
check("(g) a still person at the rail loses to a pacer 3 m out (activity first)", sw is not None and sw - 10 < 3.0, str(sw))
sw_ak = _first(_cfg_k, lambda e: [_N(1)] + ([_FK(2)] if e >= 10 else []), 25, 2)
sw_ka = _first(_cfg_k, lambda e: [_FK(1)] + ([_N(2)] if e >= 10 else []), 25, 2)
check("(h) adult at the rail vs child 3 m out, both lively: whoever came first keeps it", sw_ak is None and sw_ka is None, f"{sw_ak} {sw_ka}")

print("newcomer takes attention from a still person")
w2 = World(seed=2)
w2.run(75, lambda i, t: [S.Person((3.5 - 2.0 * min(1, i / 60), 0.3))])
w2.run(600, lambda i, t: [S.Person((1.5, 0.3))])
d, dbg = w2.run(120, lambda i, t: [S.Person((1.5, 0.3)), S.Person((3.5 - 1.8 * min(1, i / 90), -1.5))])
check("switched to the walker", d.track_id == 2 and d.hint in ("engage", "watch"), f"id={d.track_id} {d.hint} S={dbg.scores}")
switches = 0
last = d.track_id
for k in range(6):
    d, dbg = w2.run(15, lambda i, t: [S.Person((1.5, 0.3)), S.Person((1.7, -1.5))])
    if d.track_id != last and d.track_id != -1:
        switches += 1
        last = d.track_id
check("no flicker between two still people (<2 switches / 3 s)", switches < 2, str(switches))

print("end to end: a child pacing beside a pacing adult gets the attention; the rail beats 3 m out")
_pace = lambda t, y, h=1.7: S.Person((2.2 + 0.7 * np.sin(1.2 * t), y), height=h)   # phase on t: continuous across run() calls
wk = World(seed=7)
wk.run(90, lambda i, t: [_pace(t, -1.5)])
d, dbg = wk.run(180, lambda i, t: [_pace(t, -1.5), _pace(t, 1.5, 1.1)])
check("the child is tracked at a child's height", any(tr.id == 2 and tr.height < 1.3 for tr in dbg.tracks), str([(tr.id, round(tr.height, 2)) for tr in dbg.tracks]))
check("and takes the attention from the equally lively adult within 6 s", d.track_id == 2 and d.hint in ("engage", "watch"), f"id={d.track_id} {d.hint} S={dbg.scores}")
wk = World(seed=7)
wk.run(90, lambda i, t: [_pace(t, 1.5, 1.1)])
d, dbg = wk.run(300, lambda i, t: [_pace(t, -1.5), _pace(t, 1.5, 1.1)])
check("an adult arriving beside an engaged child does not take it", d.track_id == 1, f"id={d.track_id} S={dbg.scores}")
_far = lambda t: S.Person((3.0, 0.7 * np.sin(1.2 * t)))
_near = lambda t: S.Person((1.6, -0.7 * np.sin(1.2 * t)))
wc = World(seed=7)
wc.run(90, lambda i, t: [_far(t)])
d, dbg = wc.run(180, lambda i, t: [_far(t), _near(t)])
check("a pacer at the rail takes attention from an engaged pacer 3 m out within 6 s", d.track_id == 2,
      f"id={d.track_id} S={dbg.scores} r={[round(tr.r_centroid, 2) for tr in dbg.tracks]}")

print("simple mode (interest_model off) = nearest person")
cfg = _scene.cfg()
cfg.interest_model = False
w3 = World(cfg=cfg, seed=3)
d, dbg = w3.run(30, lambda i, t: [S.Person((2.8, 0.3)), S.Person((1.7, -1.2))])
near = min(dbg.tracks, key=lambda x: x.r)
check("nearest chosen", d.track_id == near.id and d.hint == "engage", f"id={d.track_id} near={near.id}")
d, dbg = w3.run(900, lambda i, t: [S.Person((2.8, 0.3)), S.Person((1.7, -1.2))])
check("simple mode never bores", d.track_id == near.id)

print("perf")
ms = []
for _ in range(30):
    d, dbg = w3.run(1, lambda i, t: [S.Person((2.8, 0.3)), S.Person((1.7, -1.2)), S.Person((2.2, 1.5))])
    ms.append(dbg.step_ms)
check("step p95 < 15 ms with 3 people, 2 cams (Mac)", np.percentile(ms, 95) < 15, f"p95={np.percentile(ms,95):.1f} ms")

print("still but jittery (silhouette flicker, weight shifting) -> still ignored")
w3 = World(seed=3)
w3.run(75, lambda i, t: [S.Person((3.5 - 2.0 * min(1, i / 60), 0.3))])
jr = np.random.default_rng(5)
hints = []
for k in range(8):
    d, dbg = w3.run(150, lambda i, t: [S.Person((1.5 + jr.normal(0, 0.02), 0.3 + jr.normal(0, 0.02)))])
    hints.append(d.hint)
check("jitter earns no attention: decays to idle within 40 s", hints[-1] == "idle", str(hints))

print("presence gate: nobody interesting -> sleep, an empty room -> short sweeps between sleeps")
from rt_depth_tracker.presence import ASLEEP, SleepGate  # noqa: E402


def _gate(seed=0, **kw):
    cfg = _scene.cfg(sleep_gate=True, **kw)
    return SleepGate(cfg, rng=np.random.default_rng(seed)), cfg


def _drive(g, hint, n_tracks, seconds, dt=1 / 30.0):
    """Emitted hints over `seconds` of a steady situation."""
    return [g.update(hint, n_tracks, dt) for _ in range(int(seconds / dt))]


g_off = SleepGate(_scene.cfg(), rng=np.random.default_rng(0))     # dataclass default is off
check("gate off is a pass-through", set(_drive(g_off, "idle", 0, 600)) == {"idle"})

g, cfg = _gate()
for h in ("watch", "engage", "reach"):
    check(f"'{h}' passes through untouched", _drive(g, h, 1, 1) == [h] * 30)

g, cfg = _gate()
out = _drive(g, "idle", 0, cfg.doze_empty_s + 5)
first = out.index("sleep") / 30.0
check("empty room sweeps for doze_empty_s, then sleeps", abs(first - cfg.doze_empty_s) < 0.2, f"{first:.1f} s")

g, cfg = _gate()
out = _drive(g, "idle", 1, 300)
check("someone still there but boring -> sleeps at once, no sweeping at them", set(out) == {"sleep"}, str(sorted(set(out))))
check("and never a sweep while they stand there", g.state == ASLEEP, g.state)

# a site that never idles: asleep the moment nobody is worth watching, no look-around sweeps
_never = _scene.cfg(sleep_gate=True, doze_empty_s=0.0, sweep_hi_s=0.0)
g_site = SleepGate(_never, rng=np.random.default_rng(3))
_seq = (_drive(g_site, "engage", 1, 20) + _drive(g_site, "idle", 0, 1800) + _drive(g_site, "watch", 1, 5)
        + _drive(g_site, "idle", 1, 60) + _drive(g_site, "idle", 0, 1800))
check("never-idle config: the gate never says idle, not even for one frame in an hour", "idle" not in _seq, str(sorted(set(_seq))))
check("never-idle config: the first frame after a person leaves is already sleep", _drive(SleepGate(_never), "idle", 0, 1 / 30.0) == ["sleep"])
check("never-idle config: still wakes for a visitor", g_site.update("engage", 1, 1 / 30.0) == "engage")

g, cfg = _gate(seed=2)
out = _drive(g, "idle", 0, 1200)
runs, prev = [], None
for h in out:                                    # run-lengths, in seconds
    if h != prev:
        runs.append([h, 0])
        prev = h
    runs[-1][1] += 1 / 30.0
sweeps = [d for h, d in runs[1:] if h == "idle"]
sleeps = [d for h, d in runs if h == "sleep"][:-1]
check("sweeps appear between sleeps", len(sweeps) >= 2, str([round(x, 1) for x in sweeps]))
check("each sweep is 45-120 s", all(cfg.sweep_lo_s - 0.1 <= d <= cfg.sweep_hi_s + 0.1 for d in sweeps),
      str([round(x, 1) for x in sweeps]))
check("each sleep is 2-8 min", all(cfg.sleep_lo_s - cfg.doze_empty_s - 0.1 <= d <= cfg.sleep_hi_s + 0.1 for d in sleeps),
      str([round(x, 1) for x in sleeps]))
check("the cadence is not a clock", len(set(round(x) for x in sweeps)) > 1 or len(set(round(x) for x in sleeps)) > 1,
      str([round(x, 1) for x in sweeps + sleeps]))
check("first sweep is sleep_lo_s after the room emptied, not after settling",
      runs[0][1] + runs[1][1] >= cfg.sleep_lo_s - 0.1, f"{runs[0][1] + runs[1][1]:.1f} s")

g, cfg = _gate()
_drive(g, "idle", 0, cfg.doze_empty_s + 5)
check("asleep before the wave", g.update("idle", 0, 1 / 30.0) == "sleep")
check("one attentive hint wakes it on the next frame", g.update("engage", 1, 1 / 30.0) == "engage")

g1, _ = _gate(seed=1)
g2, _ = _gate(seed=1)
check("same seed, same dwells", _drive(g1, "idle", 0, 900) == _drive(g2, "idle", 0, 900))

print("presence gate end to end (short timings so the suite stays quick)")
cfg_sleep = _scene.cfg(sleep_gate=True, doze_empty_s=1.0, doze_present_s=0.0,
                          sleep_lo_s=3.0, sleep_hi_s=3.0, sweep_lo_s=1.0, sweep_hi_s=1.0)
wz = World(cfg=cfg_sleep, seed=6)
d, _ = wz.run(75, lambda i, t: [S.Person((3.5 - 2.0 * min(1, i / 60), 0.3))])
check("a walker is awake", d.hint in ("watch", "engage"), d.hint)
hints = [wz.run(1, lambda i, t: [])[0].hint for _ in range(210)]
seq = [h for k, h in enumerate(hints) if k == 0 or h != hints[k - 1]]     # run-compressed
check("the room empties -> look around, sleep, wake to sweep, sleep again",
      [h for h in seq if h in ("idle", "sleep")][:4] == ["idle", "sleep", "idle", "sleep"], str(seq))
d, _ = wz.run(60, lambda i, t: [S.Person((2.0, 0.3))])
check("someone walks back in -> awake", d.hint not in ("idle", "sleep"), d.hint)

print("a tracker with no calibration must degrade quietly, never throw every frame")
# a field incident: sensor_*.json missing but bg_*.npz loaded, so cs.bg.ready was True while
# T_ws stayed None. points_for returned a (0,) mask, slow_update broadcast it against (144,160) and
# raised on every step for 17 minutes while the heartbeat kept saying the tracker was fine.
from rt_depth_tracker.background import BackgroundModel  # noqa: E402
from rt_depth_tracker.pipeline import Pipeline  # noqa: E402

cfg_nc = _scene.cfg()
pipe_nc = Pipeline(cfg_nc, robot=rm, rng=np.random.default_rng(0))
for cid in cams:
    pipe_nc.set_intrinsics(S.intrinsics_for(cid))          # intrinsics arrive over the wire
    pipe_nc.cam(cid).bg.learn(np.full((3, 144, 160), 2000, np.uint16))        # background ready
check("the incident's precondition: a ready background on an uncalibrated camera",
      all(pipe_nc.cam(c).bg.ready and pipe_nc.cam(c).T_ws is None for c in cams))
check("and the camera reports itself not ready", not any(pipe_nc.ready(c) for c in cams))

scene_nc = S.Scene(rail_radius=1.4, q=QD.copy(), robot=rm)
scene_nc.people = [S.Person((2.0, 0.3))]
raised = None
for k in range(120):                                        # past bg_empty_s in scaled time
    pk = {c: S.packet(c, S.render(scene_nc, cams[c], rng=rng), k, k / 30.0) for c in cams}
    pipe_nc.set_q(scene_nc.q)
    try:
        dnc, _ = pipe_nc.step(pk, 1000.0 + k * 10.0)        # big dt so the 120 s empty timer elapses
    except Exception as e:                                  # noqa: BLE001
        raised = f"{type(e).__name__}: {e}"
        break
check("no exception on any step (the regression)", raised is None, str(raised))
check("it produces a quiet idle decision, not garbage", raised is None and dnc.track_id == -1, str(raised))

bgm = BackgroundModel()
bgm.learn(np.full((3, 144, 160), 2000, np.uint16))
import tempfile  # noqa: E402
_f = os.path.join(tempfile.gettempdir(), "bg_shape_test.npz")
bgm.save(_f)
other = BackgroundModel()
check("a background from another depth mode is rejected, not loaded", other.load(_f, shape=(288, 320)) is False)
check("and the matching shape still loads", other.load(_f, shape=(144, 160)) is True)

print("a tall set piece is not a visitor")
wt = World(seed=6)
wall = S.Person((2.5, 1.2), height=2.8, radius=0.35)          # panel-ish; reads p95 ~2.55 through the z_max 2.9 crop
d, dbg = wt.run(60, lambda i, t: [wall])
check("2.8 m object: seen, but no clusters, no tracks, idle", len(dbg.points) > 2000 and len(dbg.clusters) == 0 and len(dbg.tracks) == 0 and d.hint == "idle",
      f"{d.hint} pts={len(dbg.points)} h={[round(c.height_p95, 2) for c in dbg.clusters]}")
d, dbg = wt.run(75, lambda i, t: [wall, S.Person((3.5 - 2.0 * min(1, i / 60), -0.5))])
check("a visitor walking in beside it is engaged, at a visitor's height", d.hint == "engage" and len(dbg.tracks) == 1 and dbg.tracks[0].height < 2.0,
      f"{d.hint} {[(tr.id, round(tr.height, 2)) for tr in dbg.tracks]}")

print("a surface nudged onto the tolerance still heals")
wh = World(cfg=_scene.cfg(bg_absorb_s=5.0), seed=8)
pole = S.Person((2.6, 2.2), height=2.1, radius=0.09)           # part of the set: learned as background
wh.scene.people = [pole]
wh.pipe.learn_background(frames=40)
wh.run(45)


def cam_facing(xy):
    """The camera whose optical axis points closest at xy (the synth cameras each cover one side)."""
    def off_axis(c):
        fwd = cams[c][:2, 2] / np.linalg.norm(cams[c][:2, 2])
        to = np.array(xy) - cams[c][:2, 3]
        return np.arccos(np.clip(fwd @ to / np.linalg.norm(to), -1, 1))
    return min(cams, key=off_axis)


cid_pole = cam_facing(pole.xy)
v = np.array(pole.xy) - cams[cid_pole][:2, 3]
moved = S.Person(tuple(np.array(pole.xy) - 0.085 * v / np.linalg.norm(v)), height=2.1, radius=0.09)   # 85 mm toward that camera
d, dbg = wh.run(30, lambda i, t: [moved])
check("the nudged pole is foreground at once", len(dbg.clusters) >= 1 and min(np.hypot(c.centroid[0] - 2.6, c.centroid[1] - 2.2) for c in dbg.clusters) < 0.4,
      f"cam {cid_pole} {[(round(c.centroid[0], 2), round(c.centroid[1], 2)) for c in dbg.clusters]}")
d, dbg = wh.run(300, lambda i, t: [moved])
check("and healed within 2 x absorb_s", len(dbg.clusters) == 0 and wh.pipe.absorbed_total > 30,
      f"clusters={len(dbg.clusters)} absorbed={wh.pipe.absorbed_total}")

print("healing is a session overlay: the learned file is the ground truth, a healed wall un-heals when it leaves")


def healed_px(w):
    return sum(c.bg.healed_px for c in w.pipe.cams.values())


wall = S.Person((2.5, 0.8), height=1.6, radius=0.3)          # a panel rolled in: still, below max_height
ws = World(cfg=_scene.cfg(bg_absorb_s=5.0, bg_unheal_s=1.0), seed=11)
gt0 = {c: cs.bg.bg.copy() for c, cs in ws.pipe.cams.items()}
d, dbg = ws.run(60, lambda i, t: [wall])
check("a wall rolled in is foreground at first", len(dbg.clusters) >= 1, f"clusters={len(dbg.clusters)}")
d, dbg = ws.run(180, lambda i, t: [wall])
check("and healed after absorb_s", len(dbg.clusters) == 0 and healed_px(ws) > 50, f"clusters={len(dbg.clusters)} healed={healed_px(ws)}")
check("healing never touched the ground truth", all(np.array_equal(ws.pipe.cams[c].bg.bg, gt0[c]) for c in gt0))
d, dbg = ws.run(45, lambda i, t: [])
check("the wall leaves: un-healed within 1.5 s, no ghost", healed_px(ws) == 0 and len(dbg.clusters) == 0, f"healed={healed_px(ws)} clusters={len(dbg.clusters)}")
d, dbg = ws.run(30, lambda i, t: [S.Person((2.5, 0.8))])
check("a visitor on the wall's old spot is tracked at once (no blind spot)", len(dbg.clusters) >= 1 and len(dbg.tracks) >= 1,
      f"clusters={len(dbg.clusters)} tracks={len(dbg.tracks)}")

print("a still visitor is absorbed as before, and is foreground again the moment they step aside")
wv = World(cfg=_scene.cfg(bg_absorb_s=5.0, bg_unheal_s=1.0), seed=12)
still = S.Person((2.0, -0.5))
d, dbg = wv.run(200, lambda i, t: [still])
check("motionless for absorb_s -> absorbed", len(dbg.clusters) == 0 and healed_px(wv) > 20, f"clusters={len(dbg.clusters)} healed={healed_px(wv)}")
h_before = healed_px(wv)
d, dbg = wv.run(40, lambda i, t: [S.Person((2.0, 0.4))])
check("one step aside: tracked again, the old spot un-heals", len(dbg.clusters) >= 1 and healed_px(wv) < h_before * 0.3,
      f"clusters={len(dbg.clusters)} healed {h_before} -> {healed_px(wv)}")

print("healing needs the SAME surface for absorb_s: a visitor crossing in front restarts it, and is never what gets healed")
wc = World(cfg=_scene.cfg(bg_absorb_s=5.0, bg_unheal_s=1.0), seed=13)
wc.run(140, lambda i, t: [wall])                                              # 4.7 s of the wall: nearly healed
d, dbg = wc.run(12, lambda i, t: [wall, S.Person((1.6, 0.8))])                # someone walks across in front of it
check("the passer-by is foreground", len(dbg.clusters) >= 1, f"clusters={len(dbg.clusters)}")
d, dbg = wc.run(60, lambda i, t: [wall])
check("the wall is still foreground afterwards: the clock restarted on the pixels the visitor covered",
      len(dbg.clusters) >= 1, f"clusters={len(dbg.clusters)} healed={healed_px(wc)}")
d, dbg = wc.run(150, lambda i, t: [wall])
check("and heals once it alone has held still for absorb_s", len(dbg.clusters) == 0 and healed_px(wc) > 50,
      f"clusters={len(dbg.clusters)} healed={healed_px(wc)}")

print("the file holds the learned ground truth only")
import tempfile  # noqa: E402
from rt_depth_tracker.background import BackgroundModel  # noqa: E402
_tmp = os.path.join(tempfile.mkdtemp(), "bg_0.npz")
_cs = wc.pipe.cams[0]
_cs.bg.save(_tmp)
_m = BackgroundModel()
check("save() wrote the ground truth, not the healed wall", _m.load(_tmp) and np.array_equal(_m.bg, _cs.bg.bg) and _cs.bg.healed_px > 0)
check("load() starts with an empty overlay", _m.healed_px == 0 and np.isnan(_m.heal).all())

print("the example config: no robot, no zones; people are tracked without a self-mask")
_ex = TrackerConfig.load(_scene.EXAMPLE_CONFIG)
check("example config loads with the floor at the origin and no robot", _ex.mount_z == 0.0 and _ex.robot == {} and _ex.zones == [] and _ex.wall_x is None and _ex.rail_radius == 0.0)
check("no robot block -> RobotModel.from_config is None", RobotModel.from_config(_ex) is None)
_pe = Pipeline(_ex, rng=np.random.default_rng(0))
check("the pipeline has no robot and masks nothing", _pe.robot is None)
for _c, _T in cams.items():
    _pe.set_calib(_c, _T)
    _pe.set_intrinsics(S.intrinsics_for(_c))
_sc = S.Scene()
_pe.learn_background(frames=20)
_dec = None
for _k in range(60):
    _sc.people = [] if _k < 20 else [S.Person((3.4 - 0.05 * (_k - 20), 0.3))]
    _dec, _dbg = _pe.step({_c: S.packet(_c, S.render(_sc, cams[_c], rng=rng), _k, _k / 30.0) for _c in cams}, _k / 30.0)
check("a walker is tracked and engaged with no robot anywhere", _dec.hint in ("watch", "engage") and len(_dbg.tracks) == 1 and _dbg.robot_pts_removed == 0,
      f"{_dec.hint} tracks={len(_dbg.tracks)} removed={_dbg.robot_pts_removed}")

print(f"test_behavior: {'OK' if _fail == 0 else str(_fail) + ' FAILED'}")
sys.exit(1 if _fail else 0)
