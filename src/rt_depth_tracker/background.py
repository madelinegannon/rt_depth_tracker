# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Per-pixel static background model on the depth image, in two layers.

The GROUND TRUTH is what `learn-bg` measured of the empty room: the median depth per pixel. It is the only
thing ever written to bg_<cam>.npz, and nothing at runtime changes the file.

HEALING is a session overlay on top of it: a surface that is not the ground truth but holds perfectly still
for absorb_s (a moved chair, a sign, a wall rolled in for a day) becomes `heal` and stops being foreground.
It only ever excuses returns that look like it; anything else is still judged against the ground truth, so
a visitor standing where a healed wall used to be is foreground at once. The moment the learned surface
shows again for unheal_s the pixel un-heals. The overlay lives in memory only and starts empty at every
start, so a wall that left while the tracker was down never comes back as a blind spot.

Foreground = closer than the ground truth by more than a tolerance (never 'farther': multipath/flying
pixels), or a return inside range on a pixel that never learned a depth, unless it matches the healed
surface. A slow EMA of the ground truth after empty_s of empty scene follows thermal drift, in memory only."""
from __future__ import annotations

import time
import warnings

import numpy as np


class BackgroundModel:
    def __init__(self, tol_mm=40.0, tol_sig=3.0, tol_frac=0.01, valid_frac=0.5, ema_alpha=0.002, empty_s=120.0,
                 absorb_s=600.0, invalid_max_mm=4500.0, agree_frac=0.5, unheal_s=2.0):
        self.tol_mm, self.tol_sig, self.tol_frac = float(tol_mm), float(tol_sig), float(tol_frac)
        self.valid_frac, self.ema_alpha, self.empty_s = float(valid_frac), float(ema_alpha), float(empty_s)
        self.absorb_s, self.invalid_max_mm = float(absorb_s), float(invalid_max_mm)
        self.agree_frac = float(agree_frac)      # a return is still the ground truth only inside this fraction of tol
        self.unheal_s = float(unheal_s)          # the learned surface seen this long at a healed pixel un-heals it
        # ground truth (saved)
        self.bg: np.ndarray | None = None        # float32 mm
        self.sig: np.ndarray | None = None       # float32 mm (1.4826 * MAD)
        self.invalid: np.ndarray | None = None   # bool: no reliable background here -> any return is fg
        self.tol: np.ndarray | None = None
        self.learned_at: float | None = None
        # session overlay (never saved)
        self.heal: np.ndarray | None = None      # float32 mm, NaN = nothing healed here
        self.cand: np.ndarray | None = None      # the still surface that may be healed next, NaN = none
        self.cand_age: np.ndarray | None = None  # seconds the returns have stayed on `cand`
        self.gt_age: np.ndarray | None = None    # seconds a healed pixel has shown the ground truth again
        self._unhealed_pending = 0
        self._empty_since: float | None = None
        self._buf: list[np.ndarray] = []
        self.learn_frames_wanted = 0

    # ---- learning ----------------------------------------------------------------------
    def learn(self, frames: np.ndarray, t: float | None = None):
        """frames: (T,H,W) uint16 depth mm of an EMPTY scene."""
        F = np.asarray(frames)
        valid = F > 0
        frac = valid.mean(axis=0)
        Ff = np.where(valid, F.astype(np.float32), np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns are expected (never-valid pixels)
            med = np.nanmedian(Ff, axis=0)
            mad = np.nanmedian(np.abs(Ff - med[None]), axis=0)
        self.invalid = (frac < self.valid_frac) | ~np.isfinite(med)
        self.bg = np.where(self.invalid, 0.0, med).astype(np.float32)
        self.sig = np.where(self.invalid, 0.0, 1.4826 * np.nan_to_num(mad)).astype(np.float32)
        self._update_tol()
        self._reset_overlay()
        self.learned_at = time.time() if t is None else t
        self._empty_since = None

    def begin_learning(self, n_frames: int):
        self._buf, self.learn_frames_wanted = [], int(n_frames)

    def feed_learning(self, depth: np.ndarray) -> bool:
        """Accumulate frames; returns True when learning completed on this call."""
        if self.learn_frames_wanted <= 0:
            return False
        self._buf.append(depth.copy())
        if len(self._buf) >= self.learn_frames_wanted:
            self.learn(np.stack(self._buf))
            self._buf, self.learn_frames_wanted = [], 0
            return True
        return False

    @property
    def learning(self) -> bool:
        return self.learn_frames_wanted > 0

    def _update_tol(self):
        self.tol = np.maximum.reduce([np.full_like(self.bg, self.tol_mm), self.tol_sig * self.sig, self.tol_frac * self.bg])

    def _reset_overlay(self):
        self.heal = np.full_like(self.bg, np.nan)
        self.cand = np.full_like(self.bg, np.nan)
        self.cand_age = np.zeros_like(self.bg)
        self.gt_age = np.zeros_like(self.bg)
        self._unhealed_pending = 0

    # ---- inference ---------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.bg is not None

    @property
    def healed_px(self) -> int:
        return 0 if self.heal is None else int(np.count_nonzero(np.isfinite(self.heal)))

    def foreground(self, depth: np.ndarray) -> np.ndarray:
        valid = depth > 0
        if self.bg is None:
            return valid
        d = depth.astype(np.float32)
        closer = d < (self.bg - self.tol)
        unlearned = self.invalid & (depth < self.invalid_max_mm)   # a return where nothing learned, unless out at the range limit
        fg = valid & (closer | unlearned)
        if self.heal is not None:
            with np.errstate(invalid="ignore"):
                healed = np.abs(d - self.heal) < self.tol        # NaN heal compares False: nothing excused
            fg &= ~healed
        return fg

    def absorb(self, depth: np.ndarray, dt: float) -> int:
        """Session healing. A return that is neither the ground truth (inside agree_frac * tol) nor the healed
        surface becomes a candidate; the candidate is healed once the SAME surface has held still for absorb_s
        (a visitor who shifts restarts it, furniture does not, so a healed pixel is never a passer-by). A healed
        pixel that shows the ground truth again, or no return at all, for unheal_s is un-healed. A dropped return
        holds the candidate's age (flickery surfaces still heal). Runs whether or not the scene is empty. Returns
        pixels healed on this call."""
        if self.bg is None or self.absorb_s <= 0:
            return 0
        if self.heal is None or self.heal.shape != self.bg.shape:
            self._reset_overlay()
        d = depth.astype(np.float32)
        valid = depth > 0
        band = self.agree_frac * self.tol
        with np.errstate(invalid="ignore"):
            on_gt = valid & ~self.invalid & (np.abs(d - self.bg) < band)
            on_heal = valid & (np.abs(d - self.heal) < self.tol)
            same = valid & (np.abs(d - self.cand) < self.tol)
        # un-heal: the learned surface is back at a healed pixel, or nothing at all is (the healed object stood
        # where the room had no return: once it is gone there is nothing left to see there either)
        healed = np.isfinite(self.heal)
        back = healed & (on_gt | ~valid)
        self.gt_age[back] += dt
        self.gt_age[~back] = 0.0
        gone = self.gt_age >= self.unheal_s
        n_un = int(np.count_nonzero(gone))
        if n_un:
            self.heal[gone] = np.nan
            self.gt_age[gone] = 0.0
            self._unhealed_pending += n_un
        # candidate: a still surface that is neither layer
        off = valid & ~on_gt & ~on_heal
        self.cand_age[off & same] += dt
        fresh = off & ~same
        self.cand[fresh] = d[fresh]
        self.cand_age[fresh] = 0.0
        settled = valid & ~off                      # a return on either layer drops the candidate
        self.cand[settled] = np.nan
        self.cand_age[settled] = 0.0
        snap = self.cand_age >= self.absorb_s
        n = int(np.count_nonzero(snap))
        if n:
            self.heal[snap] = self.cand[snap]
            self.cand[snap] = np.nan
            self.cand_age[snap] = 0.0
            self.gt_age[snap] = 0.0
        return n

    def pop_unhealed(self) -> int:
        """Pixels un-healed since the last call (for the runtime's log and heartbeat)."""
        n, self._unhealed_pending = self._unhealed_pending, 0
        return n

    def slow_update(self, depth: np.ndarray, fg: np.ndarray, scene_empty: bool, now: float):
        """EMA toward the current frame on ground-truth pixels, only after `empty_s` of empty scene. Thermal
        drift, in memory only: it is never saved and re-adapts within minutes of a start."""
        if self.bg is None:
            return
        if not scene_empty:
            self._empty_since = None
            return
        if self._empty_since is None:
            self._empty_since = now
            return
        if now - self._empty_since < self.empty_s:
            return
        upd = (depth > 0) & ~fg & ~self.invalid & ~np.isfinite(self.heal)
        a = self.ema_alpha
        self.bg[upd] = (1 - a) * self.bg[upd] + a * depth[upd].astype(np.float32)
        self._update_tol()

    # ---- persistence (ground truth only) -------------------------------------------------
    def save(self, path: str):
        np.savez_compressed(path, bg=self.bg, sig=self.sig, invalid=self.invalid,
                            learned_at=np.float64(self.learned_at or 0.0),
                            params=np.array([self.tol_mm, self.tol_sig, self.tol_frac, self.valid_frac, self.ema_alpha, self.empty_s]))

    def load(self, path: str, shape: tuple | None = None) -> bool:
        try:
            d = np.load(path)
        except (OSError, ValueError):
            return False
        if shape is not None and tuple(d["bg"].shape) != tuple(shape):
            return False      # recorded at another stride or depth mode: loading it would raise every frame
        self.bg, self.sig, self.invalid = d["bg"].astype(np.float32), d["sig"].astype(np.float32), d["invalid"].astype(bool)
        self.learned_at = float(d["learned_at"])
        self._update_tol()
        self._reset_overlay()
        return True

    def age_s(self, now: float | None = None) -> float | None:
        if self.learned_at is None:
            return None
        return (time.time() if now is None else now) - self.learned_at
