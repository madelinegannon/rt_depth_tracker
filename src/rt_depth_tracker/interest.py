# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Interest model: who the robot pays attention to. 'If you ignore it, it ignores you.'

Per confirmed track a score S in [0,1] integrates approach, motion energy, reaching, being over
the rail and novelty; it decays with stillness and slowly even when active; long engagement gets
bored and enters a refractory period. A small FSM with hysteresis picks ONE target, ranking S by
closeness and shortness (kids first); S itself stays activity-only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rt_depth_tracker.config import TrackerConfig
from rt_depth_tracker.tracks import Track


@dataclass
class Decision:
    target: np.ndarray      # (3,) world metres
    hint: str               # idle | watch | engage | reach
    track_id: int
    score: float


class InterestModel:
    def __init__(self, cfg: TrackerConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.S: dict[int, float] = {}
        self.first: dict[int, float] = {}
        self.refractory_until: dict[int, float] = {}
        self._railed: set[int] = set()
        self.current: int | None = None
        self.engaged_since: float | None = None
        self.bored_at: float | None = None
        self._challenger: tuple[int, float] | None = None
        self._low_since: float | None = None

    # ---- scoring --------------------------------------------------------------------------
    def _score(self, t: Track, dt: float, now: float, fragment: bool = False) -> float:
        c = self.cfg
        S = self.S.get(t.id, 0.0 if fragment else c.novelty)
        if t.id not in self.first:
            self.first[t.id] = now
        r = t.r_centroid or t.r
        # radial speed from the track's filtered velocity: per-frame r differences rectify jitter into 'approach'
        rel = np.asarray(t.xy, float) - np.asarray(c.base_xy, float)
        dr = float(np.dot(np.asarray(t.v, float), rel / max(float(np.linalg.norm(rel)), 1e-6)))
        # dynamic gains only, above the stillness floor: a still or flickering shape accrues nothing, so decay wins
        gain = 0.0
        gain += c.w_approach * max(0.0, -dr - c.still_energy)
        # normalised above the floor: a walker at full energy still earns w_motion, jitter below the floor earns nothing
        live = max(0.0, min(t.energy, 1.0) - c.still_energy) / max(1e-6, 1.0 - c.still_energy)
        gain += c.w_motion * live
        if t.reach > c.reach_thresh:
            gain += c.w_reach
        if t.over_rail:
            # sticky is a static gain and a per-frame floor: it pins S at 1.0, so a still leaner is never outbid
            gain += c.w_over_rail if c.over_rail_sticky else c.w_over_rail * live
            if c.over_rail_sticky or t.id not in self._railed:
                S = max(S, c.over_rail_jump)      # unsticky: on the rising edge only, leaning in still grabs attention
            self._railed.add(t.id)
        else:
            self._railed.discard(t.id)
        # proximity scales what you do, it is not a reward for standing there
        gain *= 1.0 + c.proximity * float(np.clip((c.prox_far - r) / (c.prox_far - c.prox_near), 0, 1))
        if not fragment:
            gain += c.novelty * np.exp(-(now - self.first[t.id]) / c.novelty_tau) / c.novelty_tau
        decay = S / (c.decay_still_s if t.energy < c.still_energy else c.decay_active_s)
        S = S + (gain - decay) * dt
        if now < self.refractory_until.get(t.id, 0.0) and not (t.over_rail and c.over_rail_sticky):
            S = min(S, c.refractory_cap)
        S = float(np.clip(S, 0.0, 1.0))
        self.S[t.id] = S
        return S

    def _rank(self, t: Track, S: float) -> float:
        c = self.cfg
        near = np.clip((c.prox_far - (t.r_centroid or t.r)) / max(c.prox_far - c.prox_near, 1e-6), 0.0, 1.0)
        short = np.clip((c.adult_h - t.height) / max(c.adult_h - c.kid_h, 1e-6), 0.0, 1.0)
        return S * (1.0 + c.near_pref * float(near) + c.kid_pref * float(short))

    # ---- selection ----------------------------------------------------------------------------
    def update(self, tracks: list[Track], dt: float, now: float) -> Decision:
        c = self.cfg
        by_id = {t.id: t for t in tracks}
        for tid in list(self.S):
            if tid not in by_id:
                for d in (self.S, self.first, self.refractory_until):
                    d.pop(tid, None)
                self._railed.discard(tid)
        # a track born next to an older one is a fragment of it (an arm, a bag): no novelty, it cannot out-race its owner
        frag = {t.id for t in tracks
                if any(o.id != t.id and o.first_seen < t.first_seen and np.linalg.norm(o.xy - t.xy) < c.fragment_attach_m for o in tracks)}
        scores = {t.id: self._score(t, dt, now, fragment=t.id in frag) for t in tracks}

        if not c.interest_model:
            # simple mode: nearest confirmed person, with hysteresis on switching
            best = min(tracks, key=lambda t: t.r, default=None)
            if best is None:
                self.current = None
            elif self.current not in by_id or best.r < by_id[self.current].r - 0.3:
                self.current = best.id
            return self._decision(by_id, scores, now)

        # boredom on the engaged track
        if self.current is not None and self.current in by_id:
            if self.engaged_since is None:
                self.engaged_since = now
                self.bored_at = now + float(self.rng.uniform(c.bored_lo_s, c.bored_hi_s))
            elif self.bored_at is not None and now >= self.bored_at and not (by_id[self.current].over_rail and c.over_rail_sticky):
                self.S[self.current] *= c.bored_factor
                scores[self.current] = self.S[self.current]
                self.refractory_until[self.current] = now + c.refractory_s
                self.bored_at = now + float(self.rng.uniform(c.bored_lo_s, c.bored_hi_s))
        else:
            self.engaged_since = None
            self.bored_at = None

        cur_ok = self.current is not None and self.current in by_id
        cur_S = scores.get(self.current, 0.0) if cur_ok else 0.0

        # disengage
        if cur_ok:
            if cur_S < c.disengage_below:
                self._low_since = self._low_since or now
                if now - self._low_since >= c.disengage_hold_s:
                    self._release()
                    cur_ok = False
            else:
                self._low_since = None
        else:
            self._release()

        # engage / switch
        # rank orders and switches; S alone engages, bores and lets go: the same activity counts more at the rail and from a child, a still person still ranks ~0
        rank = {t.id: self._rank(t, scores[t.id]) for t in tracks}
        cands = sorted(((r, tid) for tid, r in rank.items()), reverse=True)
        if cands:
            top_R, top_id = cands[0]
            cur_R = rank.get(self.current, 0.0) if cur_ok else 0.0
            if not cur_ok:
                for _, tid in cands:
                    if scores[tid] >= c.engage_at:      # engage on S: outranking is not the same as being interesting yet
                        self.current, self.engaged_since = tid, now
                        self.bored_at = now + float(self.rng.uniform(c.bored_lo_s, c.bored_hi_s))
                        break
            elif top_id != self.current:
                rail_switch = by_id[top_id].over_rail and not by_id[self.current].over_rail
                if rail_switch:
                    self._switch(top_id, now)
                elif (top_R > cur_R + c.switch_margin and self.engaged_since is not None
                      and now - self.engaged_since >= c.min_engaged_s):   # 'or now' ate a 0.0 clock: no switch ever
                    if self._challenger and self._challenger[0] == top_id:
                        if now - self._challenger[1] >= c.switch_hold_s:
                            self._switch(top_id, now)
                    else:
                        self._challenger = (top_id, now)
                else:
                    self._challenger = None
        return self._decision(by_id, scores, now)

    def _switch(self, tid: int, now: float):
        self.current, self.engaged_since, self._challenger, self._low_since = tid, now, None, None
        self.bored_at = now + float(self.rng.uniform(self.cfg.bored_lo_s, self.cfg.bored_hi_s))

    def _release(self):
        self.current, self.engaged_since, self.bored_at, self._challenger, self._low_since = None, None, None, None, None

    def _decision(self, by_id: dict, scores: dict, now: float) -> Decision:
        c = self.cfg
        if self.current is None or self.current not in by_id:
            return Decision(np.array(c.idle_target, float), "idle", -1, 0.0)
        t = by_id[self.current]
        S = scores.get(t.id, 0.0)
        if t.over_rail or t.reach > c.reach_thresh:
            return Decision(np.asarray(t.reach_pt, float), "reach", t.id, S)
        hint = "engage" if (S >= c.watch_below or not c.interest_model) else "watch"
        return Decision(np.asarray(t.top, float), hint, t.id, S)
