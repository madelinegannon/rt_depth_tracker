# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""One pipeline step per new frame set. Cameras free-run at 30 fps with an arbitrary phase offset, so a
step is due when every live camera has delivered a frame not stepped yet, or max_wait after the first
fresh frame (a camera that stops delivering no longer holds the others back). Frames are paired by
arrival proximity: a set completed more than half a frame after its first member is re-paired with the
next frames instead, so the two clouds in a step are never a whole frame apart."""
from __future__ import annotations


class Pacer:
    def __init__(self, max_wait: float = 0.034):
        self.max_wait = max_wait
        self.last_seq: dict[int, int] = {}
        self._fresh: set[int] = set()
        self._t_first: float | None = None

    def offer(self, packets: dict, now: float) -> dict | None:
        """packets = latest packet per live camera. Returns the fresh subset when a step is due, else None."""
        arrived = [c for c, p in packets.items() if self.last_seq.get(c) != p.seq]
        for c in arrived:
            self.last_seq[c] = packets[c].seq
        if arrived and self._t_first is None:
            self._t_first = now
        new_cams = [c for c in arrived if c not in self._fresh]
        self._fresh.update(arrived)
        if not self._fresh:
            return None
        live = set(packets)
        complete = self._fresh >= live
        if complete and new_cams and len(live) > 1 and now - self._t_first > self.max_wait / 2:
            self._fresh, self._t_first = set(new_cams), now      # the waiting frames are stale: pair the newcomer with the next ones
            return None
        if not (complete or now - self._t_first >= self.max_wait):
            return None
        out = {c: packets[c] for c in self._fresh if c in packets}
        self._fresh, self._t_first = set(), None
        return out
