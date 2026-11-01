# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Presence gate: whether the robot is awake at all. 'Nobody interesting' is not 'nobody there'.

The interest model says WHO to watch and emits "idle" when nobody earns attention -- which includes a
visitor standing right there being boring. That is the trigger for SLEEP. The idle sweep is a
different thing: bait for passersby outside the sensed zone, so it only runs when the room is
actually empty, in short bursts between sleeps. Downgrades "idle" -> "sleep"; never touches an
attentive hint. Integrates dt only, never wall time, so a clock jump cannot skip a sleep."""
from __future__ import annotations

import numpy as np

from rt_depth_tracker.config import TrackerConfig

AWAKE, DOZE, ASLEEP, SWEEP = "awake", "doze", "asleep", "sweep"


class SleepGate:
    def __init__(self, cfg: TrackerConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.state = AWAKE
        self._t = 0.0          # time in DOZE / SWEEP
        self._empty_s = 0.0    # how long the room has held zero confirmed tracks
        self._dwell = 0.0      # this sleep's drawn length
        self._sweep_len = 0.0

    def _wake(self) -> str:
        self.state, self._t = AWAKE, 0.0
        self._empty_s = 0.0
        return AWAKE

    def _sleep(self):
        c = self.cfg
        self.state, self._t = ASLEEP, 0.0
        self._dwell = float(self.rng.uniform(c.sleep_lo_s, c.sleep_hi_s))

    def update(self, hint: str, n_tracks: int, dt: float) -> str:
        if not self.cfg.sleep_gate:
            self._wake()
            return hint
        if hint != "idle":
            self._wake()
            return hint

        c = self.cfg
        empty = n_tracks == 0
        # a present-but-boring visitor resets the clock: the sweep is never performed at someone already here
        self._empty_s = self._empty_s + dt if empty else 0.0

        if self.state == AWAKE:
            self.state, self._t = DOZE, 0.0
        if self.state == DOZE:
            self._t += dt
            if self._t >= (c.doze_empty_s if empty else c.doze_present_s):
                self._sleep()
        elif self.state == ASLEEP:
            if self._empty_s >= self._dwell:
                if c.sweep_hi_s <= 0.0:
                    # sweeps disabled (a site that never idles): stay asleep, just draw the next dwell. Entering
                    # SWEEP for even one frame would emit 'idle' and fold the arm out of and back into SLEEP.
                    self._empty_s = 0.0
                    self._sleep()
                else:
                    self.state, self._t = SWEEP, 0.0
                    self._sweep_len = float(self.rng.uniform(c.sweep_lo_s, c.sweep_hi_s))
        elif self.state == SWEEP:
            self._t += dt
            if self._t >= self._sweep_len:
                self._empty_s = 0.0     # the next sleep is measured from the end of this sweep
                self._sleep()
        return "sleep" if self.state == ASLEEP else "idle"
