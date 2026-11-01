# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Visual QA of what the calibrated cameras see (matplotlib panels): each camera's depth image with its foreground,
and both cameras' points merged in the world frame (top-down + side view) with the floor marks, wall, rail and
camera poses drawn.

  python -m rt_depth_tracker.viz --live [--in 7013] [--q-port 7014]   live, beside the running tracker (capture mirrors /depth to 7013)
  python -m rt_depth_tracker.viz --clip recordings/<tag>_*.npz [--t 5.0]   one frame of a recorded clip -> PNG
  python -m rt_depth_tracker.viz --coverage [--height 3.0]       floor map of what both / one / neither camera sees (config aims)

Live keys: s = save a PNG snapshot, r = re-read calibration + background files, q = quit.
Title shows each camera's foreground centroid and their separation (the agreement number).

Robot overlay (with a robot in the config; --no-robot hides it): the tracker's own FK mask at the /robot/q pose
(live: a joint feed mirrored to --q-port; clip: the recorded q). Masked points grey, surviving points within 1.2 m of
the base red (leftover), the sphere set outlined. No q -> mask off and a red notice."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from rt_depth_tracker import calib as C
from rt_depth_tracker.background import BackgroundModel
from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.intrinsics import ray_table, unproject
from rt_depth_tracker.robot_model import RobotModel
from rt_depth_tracker.segment import crop_world

COLORS = {0: "tab:blue", 1: "tab:orange"}


def load_calib(calib_dir: str, cfg: TrackerConfig):
    calib, bgs = {}, {}
    for p in C.sensor_calib_files(calib_dir):
        T, meta = C.load_sensor_calib(p)
        c = int(meta["cam_id"])
        calib[c] = T
        bp = os.path.join(calib_dir, f"bg_{c}.npz")
        if os.path.exists(bp):
            bg = BackgroundModel(cfg.bg_tol_mm, cfg.bg_tol_sig, cfg.bg_tol_frac, cfg.bg_valid_frac)
            if bg.load(bp):
                bgs[c] = bg
    return calib, bgs


def draw_static(top, side, cfg: TrackerConfig, calib: dict):
    th = np.linspace(-np.pi, np.pi, 200)
    bx, by = cfg.base_xy
    if cfg.rail_radius > 0:
        top.plot(bx + cfg.rail_radius * np.cos(th), by + cfg.rail_radius * np.sin(th), "k--", lw=0.8, label="rail radius")
    if cfg.pedestal_radius > 0:
        top.plot(bx + cfg.pedestal_radius * np.cos(th), by + cfg.pedestal_radius * np.sin(th), "k-", lw=1)
    if cfg.wall_x is not None:
        top.axvline(cfg.wall_x, color="k", lw=2, label="wall")
        side.axvline(cfg.wall_x, color="k", lw=2)
    top.axvline(bx + cfg.x_min, color="gray", lw=0.8, ls=":", label="x_min")
    side.axhline(0, color="k", lw=1)
    if cfg.mount_z > 0:
        side.axhline(cfg.mount_z, color="k", lw=0.8, ls="--")
    for name, (mx, my) in cfg.marks.items():
        top.plot(mx, my, "r+", ms=12, mew=2)
        top.annotate(name, (mx, my), textcoords="offset points", xytext=(6, 6), color="r")
        side.axvline(mx, color="r", lw=0.5, ls=":")
    for c, T in calib.items():
        cp, zd = T[:3, 3], T[:3, 2]
        col = COLORS.get(c, "k")
        top.plot(cp[0], cp[1], marker="s", ms=8, c=col)
        top.annotate("", xy=(cp[0] + zd[0], cp[1] + zd[1]), xytext=(cp[0], cp[1]), arrowprops=dict(arrowstyle="->", color=col))
        top.annotate(f"cam {c}", (cp[0], cp[1]), textcoords="offset points", xytext=(6, -12), color=col)
        side.plot(cp[0], cp[2], marker="s", ms=8, c=col)
        side.annotate("", xy=(cp[0] + zd[0], cp[2] + zd[2]), xytext=(cp[0], cp[2]), arrowprops=dict(arrowstyle="->", color=col))
    top.set_xlim(-1.5, 4.5), top.set_ylim(-3, 3), top.set_aspect("equal")
    top.set_xlabel("x (m)"), top.set_ylabel("y (m)")
    top.set_title("top-down, world frame"), top.grid(alpha=0.3), top.legend(loc="upper right", fontsize=8)
    side.set_xlim(-1.5, 4.5), side.set_ylim(-0.3, 3.5), side.set_aspect("equal")
    side.set_xlabel("x (m)"), side.set_ylabel("z (m)"), side.set_title("side view (x-z): floor at 0, robot base height dashed"), side.grid(alpha=0.3)


def frame_points(depth: np.ndarray, intr, stride: int, T: np.ndarray, bg: BackgroundModel | None, cfg: TrackerConfig, rays_cache: dict,
                 robot: RobotModel | None = None, q=None):
    """(P_all_world subsampled, P_fg_world in the tracked volume, fg mask, mean fg column, robot-mask flags) for one
    depth frame. The robot flags come from the same inside_mask call the pipeline makes (all False without robot/q)."""
    key = (intr.cam_id, stride)
    if key not in rays_cache:
        rays_cache[key] = ray_table(intr, stride)
    rays = rays_cache[key]
    fg = bg.foreground(depth) if bg is not None and bg.ready else (depth > 0)
    P_all, _ = unproject(depth, rays)
    P_fg, idx = unproject(depth, rays, fg)
    Pw_all = C.apply(T, P_all)
    Pw_fg = C.apply(T, P_fg)
    keep = crop_world(Pw_fg, cfg) if len(Pw_fg) else np.zeros(0, bool)
    col = float(np.mean(idx[keep] % depth.shape[1])) if keep.any() else None
    if len(Pw_all) > 6000:
        Pw_all = Pw_all[np.random.default_rng(0).choice(len(Pw_all), 6000, replace=False)]
    Pk = Pw_fg[keep]
    rob = np.zeros(len(Pk), bool)
    if robot is not None and q is not None and len(Pk):
        rob = robot.inside_mask(Pk, q, cfg.robot_margin, cfg.pedestal_radius)
    return Pw_all, Pk, fg, col, rob


class View:
    """Figure with two depth panels and the merged top-down / side panels; update() redraws from frames.
    With a RobotModel and a joint vector the tracker's mask is applied: masked points grey, surviving points
    within 1.2 m of the base red (leftover), the sphere set drawn at that pose."""

    def __init__(self, cfg: TrackerConfig, calib: dict, bgs: dict, cams: list[int], plt, max_range: float = 5.0,
                 robot: RobotModel | None = None):
        self.cfg, self.calib, self.bgs, self.cams, self.plt, self.max_range, self.robot = cfg, calib, bgs, cams, plt, max_range, robot
        self.fig, self.ax = plt.subplots(2, 2, figsize=(14, 11))
        self.rays_cache: dict = {}
        self.img, self.ov, self.sc_all, self.sc_fg, self.sc_side_all, self.sc_side_fg = {}, {}, {}, {}, {}, {}
        self.sc_rob, self.sc_side_rob, self.sc_left, self.sc_side_left = {}, {}, {}, {}
        self.sph_top = self.sph_side = None
        self.last_stats: dict = {}
        self.top, self.side = self.ax[1, 0], self.ax[1, 1]
        draw_static(self.top, self.side, cfg, calib)
        for i, c in enumerate(cams[:2]):
            axi = self.ax[0, i]
            self.img[c] = axi.imshow(np.zeros((144, 160)), cmap="viridis", vmin=0.3, vmax=max_range)
            self.ov[c] = axi.imshow(np.zeros((144, 160, 4)))
            axi.set_xticks([]), axi.set_yticks([])
            col = COLORS.get(c, "k")
            self.sc_all[c] = self.top.scatter([], [], s=1, c="lightgray", alpha=0.5)
            self.sc_fg[c] = self.top.scatter([], [], s=3, c=col, label=f"cam {c} fg")
            self.sc_side_all[c] = self.side.scatter([], [], s=1, c="lightgray", alpha=0.5)
            self.sc_side_fg[c] = self.side.scatter([], [], s=3, c=col)
            self.sc_rob[c] = self.top.scatter([], [], s=3, c="dimgray", label="robot (masked)" if i == 0 else None)
            self.sc_side_rob[c] = self.side.scatter([], [], s=3, c="dimgray")
            self.sc_left[c] = self.top.scatter([], [], s=8, c="red", label="leftover < 1.2 m of base" if i == 0 else None)
            self.sc_side_left[c] = self.side.scatter([], [], s=8, c="red")
        self.top.legend(loc="upper right", fontsize=8)
        self.qtext = self.fig.text(0.01, 0.005, "", fontsize=9)

    @staticmethod
    def _xy(P, cols):
        return P[:, cols] if len(P) else np.zeros((0, 2))

    def update(self, frames: dict, intrinsics: dict, title: str = "", q=None, q_age: float | None = 0.0):
        """frames = {cam: DepthPacket}; q = joint vector for the robot mask (None = mask off, overlay hidden)."""
        q = None if (q is None or self.robot is None) else np.asarray(q, np.float64)
        base = None if self.robot is None else self.robot.T_world_base[:3, 3]
        cents, stats = {}, {}
        for c in self.cams[:2]:
            if c not in frames or c not in intrinsics or c not in self.calib:
                continue
            pkt = frames[c]
            depth = pkt.depth
            Pw_all, Pw_fg, fg, col, rob = frame_points(depth, intrinsics[c], pkt.stride, self.calib[c], self.bgs.get(c), self.cfg,
                                                       self.rays_cache, None if q is None else self.robot, q)
            img = np.where(depth > 0, depth.astype(np.float32) / 1000.0, np.nan)
            self.img[c].set_data(img)
            self.img[c].set_extent((0, img.shape[1], img.shape[0], 0))
            ov = np.zeros(img.shape + (4,))
            ov[fg] = (1, 0, 0, 0.55)
            self.ov[c].set_data(ov)
            self.ov[c].set_extent((0, img.shape[1], img.shape[0], 0))
            self.ax[0, self.cams.index(c)].set_title(f"cam {c} depth (m), red = foreground ({int(fg.sum())} px)")
            keep, masked = Pw_fg[~rob], Pw_fg[rob]
            left = np.zeros(len(Pw_fg), bool)
            if q is not None and len(Pw_fg):
                left = ~rob & (np.linalg.norm(Pw_fg - base, axis=1) < 1.2)
            lp = Pw_fg[left]
            self.sc_all[c].set_offsets(self._xy(Pw_all, [0, 1]))
            self.sc_side_all[c].set_offsets(self._xy(Pw_all, [0, 2]))
            self.sc_fg[c].set_offsets(self._xy(keep, [0, 1]))
            self.sc_side_fg[c].set_offsets(self._xy(keep, [0, 2]))
            self.sc_rob[c].set_offsets(self._xy(masked, [0, 1]))
            self.sc_side_rob[c].set_offsets(self._xy(masked, [0, 2]))
            self.sc_left[c].set_offsets(self._xy(lp, [0, 1]))
            self.sc_side_left[c].set_offsets(self._xy(lp, [0, 2]))
            stats[c] = (int(rob.sum()), int(left.sum()))
            if len(keep) > 50:
                cents[c] = (keep[:, 0].mean(), keep[:, 1].mean(), np.percentile(keep[:, 2], 97), len(keep), col)
        parts = [f"cam {c}: ({x:+.2f}, {y:+.2f}) top {z:.2f} m, {n} pts" for c, (x, y, z, n, _) in sorted(cents.items())]
        if len(cents) == 2:
            (a, b) = [cents[c] for c in sorted(cents)]
            parts.append(f"separation {np.hypot(a[0] - b[0], a[1] - b[1]) * 100:.0f} cm")
        if q is not None:
            parts.append("robot " + ", ".join(f"cam {c}: masked {m} / leftover {l}" for c, (m, l) in sorted(stats.items())))
        self.fig.suptitle((title + "\n" if title else "") + (" | ".join(parts) if parts else "no foreground"))
        self.last_stats = stats
        self._draw_spheres(q)
        if self.robot is not None:
            if q is None:
                self.qtext.set_text("no /robot/q: robot mask OFF (publish the arm's joint angles there)")
                self.qtext.set_color("red")
            else:
                stale = q_age is not None and q_age > 0.5
                self.qtext.set_text(f"q (deg) {np.round(np.degrees(q), 1).tolist()}" + (f"   STALE {q_age:.1f} s" if stale else ""))
                self.qtext.set_color("red" if stale else "black")
        return cents

    def _draw_spheres(self, q):
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle
        for old in (self.sph_top, self.sph_side):
            if old is not None:
                old.remove()
        self.sph_top = self.sph_side = None
        if q is None:
            return
        cs, rs = self.robot.sphere_world(q)
        self.sph_top = self.top.add_collection(PatchCollection([Circle((c[0], c[1]), r) for c, r in zip(cs, rs)],
                                                               facecolor="none", edgecolor="black", lw=0.6))
        self.sph_side = self.side.add_collection(PatchCollection([Circle((c[0], c[2]), r) for c, r in zip(cs, rs)],
                                                                 facecolor="black", alpha=0.12, edgecolor="black", lw=0.6))

    def reload(self, calib_dir: str):
        self.calib, self.bgs = load_calib(calib_dir, self.cfg)
        for ax in (self.top, self.side):
            for ln in list(ax.lines) + list(ax.texts):
                ln.remove()
        draw_static(self.top, self.side, self.cfg, self.calib)


def run_clip(a, cfg, plt):
    from rt_depth_tracker.record import Reader
    paths = sorted(set(sum((glob.glob(p) for p in a.clip), [])))
    if not paths:
        raise SystemExit("no clip files match --clip")
    rd = Reader(paths)
    frames, last = {}, {}
    for t, pk, q in rd:
        for c, p in pk.items():
            if last.get(c) != p.seq:
                last[c] = p.seq
                frames.setdefault(c, []).append((t, p, np.asarray(q, np.float64)))
    cams = sorted(frames)
    calib, bgs = load_calib(a.calib_dir, cfg)
    t0 = min(frames[c][0][0] for c in cams)
    if a.t is not None:
        pick = {c: min(frames[c], key=lambda tp: abs(tp[0] - t0 - a.t)) for c in cams}
    else:
        best, pick = -1, None
        n = min(len(frames[c]) for c in cams)
        for k in range(0, n, 5):
            tot = sum(int(bgs[c].foreground(frames[c][k][1].depth).sum()) if c in bgs else 0 for c in cams)
            if tot > best:
                best, pick = tot, {c: frames[c][k] for c in cams}
    t_sel = float(np.mean([pick[c][0] for c in cams]) - t0)
    robot = None if a.no_robot else RobotModel.from_config(cfg)
    q_sel = pick[cams[0]][2]
    view = View(cfg, calib, bgs, cams, plt, a.max_range, robot=robot)
    cents = view.update({c: pick[c][1] for c in cams}, rd.intrinsics, f"{os.path.basename(paths[0])}  t={t_sel:.1f}s", q=q_sel)
    for c, (x, y, z, n, _) in sorted(cents.items()):
        print(f"cam {c}: {n} fg pts in the tracked volume, centroid ({x:+.2f}, {y:+.2f}), top z {z:.2f} m")
    if robot is not None:
        print(f"robot mask at q (deg) {np.round(np.degrees(q_sel), 1).tolist()}")
        for c, (m, l) in sorted(view.last_stats.items()):
            print(f"cam {c}: masked {m} pts, leftover within 1.2 m of the base {l} pts")
    out = a.out or f"qa_{os.path.basename(paths[0]).split('_')[0]}_{t_sel:.0f}s.png"
    view.fig.tight_layout()
    view.fig.savefig(out, dpi=110)
    print("wrote", out)


def run_live(a, cfg, plt):
    from rt_depth_tracker.run import DepthRx, RobotRx
    calib, bgs = load_calib(a.calib_dir, cfg)
    cams = sorted(calib) or [0, 1]
    robot = None if a.no_robot else RobotModel.from_config(cfg)
    view = View(cfg, calib, bgs, cams, plt, a.max_range, robot=robot)
    state = {"quit": False, "snap": 0}

    def on_key(ev):
        if ev.key == "q":
            state["quit"] = True
        elif ev.key == "s":
            state["snap"] += 1
            fn = f"qa_live_{time.strftime('%Y%m%d_%H%M%S')}.png"
            view.fig.savefig(fn, dpi=110)
            print("saved", fn)
        elif ev.key == "r":
            view.reload(a.calib_dir)
            print("calibration + background re-read")
    view.fig.canvas.mpl_connect("key_press_event", on_key)
    plt.ion()
    view.fig.tight_layout()
    plt.show(block=False)
    rx = DepthRx(a.in_port)
    qrx = RobotRx(a.q_port) if robot is not None else None
    intrinsics: dict = {}
    print(f"listening for /depth on {a.in_port}" + (f" and /robot/q on {a.q_port}" if qrx is not None else "") + "; keys: s snapshot, r reload, q quit")
    period = 1.0 / max(0.5, a.rate)
    try:
        last_draw = 0.0
        while not state["quit"] and plt.fignum_exists(view.fig.number):
            rx.event.wait(0.2)
            rx.event.clear()
            pk, intr = rx.take(0.5)
            intrinsics.update(intr)
            now = time.monotonic()
            if pk and now - last_draw >= period:
                q, age = qrx.get() if qrx is not None else (None, None)
                view.update(pk, intrinsics, f"live {time.strftime('%H:%M:%S')}", q=None if age is None else q, q_age=age)
                view.fig.canvas.draw_idle()
                last_draw = now
            view.fig.canvas.flush_events()
            plt.pause(0.001)
    finally:
        rx.stop()
        if qrx is not None:
            qrx.stop()


def run_coverage(a, cfg, plt):
    from rt_depth_tracker.synth import FEMTO_NFOV_BINNED
    calib, _ = load_calib(a.calib_dir, cfg)
    cams = [int(c) for c in a.cams.split(",")]
    poses = {}
    for c in cams:
        if a.config_aims or c not in calib:
            pos, aim = cfg.cam_pose(c)
            if a.height is not None:
                pos[2] = float(a.height)
            poses[c] = C.look_at_T(pos, aim)
        else:
            poses[c] = calib[c]
    intr = FEMTO_NFOV_BINNED
    xs, ys = np.arange(-1.0, 4.5, 0.05), np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, z in zip(axes, (0.2, 1.0)):
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
        cnt = sum(C.in_view(P, poses[c], intr, cfg.stride).astype(int) for c in cams).reshape(X.shape)
        ax.contourf(X, Y, cnt, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#eeeeee", "#f2d16b", "#7fc97f"])
        draw_static(ax, plt.figure().gca(), cfg, poses)
        plt.close(plt.gcf())
        ax.set_title(f"coverage at z = {z} m: green both, yellow one, grey none")
    zone = C.zone_samples()
    both = float(np.logical_and.reduce([C.in_view(zone, poses[c], intr, cfg.stride) for c in cams]).mean())
    fig.suptitle(f"both-covered fraction of the coverage zone (x 1.2-4.0, |y| <= 2, z 0.2 & 1.0): {both:.2f}")
    out = a.out or "qa_coverage.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"both-covered fraction {both:.3f}; wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs="*", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--in", dest="in_port", type=int, default=7013, help="live: UDP port the captures mirror /depth to")
    ap.add_argument("--rate", type=float, default=5.0, help="live: redraws per second")
    ap.add_argument("--q-port", type=int, default=7014, help="live: UDP port a /robot/q feed is mirrored to")
    ap.add_argument("--no-robot", action="store_true", help="hide the robot mask overlay (spheres, masked grey, leftover red)")
    ap.add_argument("--t", type=float, default=None, help="clip: seconds into the clip; default = frame with the most foreground")
    ap.add_argument("--config", default="config/tracker.json")
    ap.add_argument("--calib-dir", default="config/calib")
    ap.add_argument("--cams", default="0,1")
    ap.add_argument("--height", type=float, default=None, help="coverage: override the camera height (m)")
    ap.add_argument("--config-aims", action="store_true", help="coverage: use cams[].pos/aim from the config instead of the calibration files")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-range", type=float, default=5.0, help="depth colour scale (m)")
    a = ap.parse_args(argv)
    import matplotlib
    if not a.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = TrackerConfig.load(a.config)
    if a.live:
        run_live(a, cfg, plt)
    elif a.coverage:
        run_coverage(a, cfg, plt)
    elif a.clip:
        run_clip(a, cfg, plt)
    else:
        raise SystemExit("one of --live, --clip, --coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
