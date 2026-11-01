# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Control messages for a running tracker (OSC to 127.0.0.1:7012):

  python -m rt_depth_tracker.ctl record <tag>       start a tagged clip -> recordings/<tag>_<ts>_<n>.npz (30 s chunks)
  python -m rt_depth_tracker.ctl record off         stop and flush the clip
  python -m rt_depth_tracker.ctl learn-bg [frames]  relearn the background (empty scene, robot parked; default 300 frames)
  python -m rt_depth_tracker.ctl reload-calib       reload sensor_*.json / bg_*.npz
  python -m rt_depth_tracker.ctl set <key> <value>  live-tune near_pref | kid_pref | kid_h | adult_h (saved to the config once it settles)

Keep clips short (1-3 min, one gesture or scenario each: walk_by, stand_still_60s, wave, reach_over_rail,
two_people_cross, arm_moving_empty ...) and replay them as a looping playlist: rt_depth_tracker.replay --loop.
"""
from __future__ import annotations

import argparse
import socket
import sys

from rt_depth_tracker import protocol as P

OFF_WORDS = ("off", "stop", "0")


def message(cmd: str, arg: str | None, value: str | None = None) -> bytes:
    if cmd == "set":
        if arg not in P.TUNABLE_KEYS or value is None:
            raise SystemExit(f"set needs <key> <value>, key one of {', '.join(P.TUNABLE_KEYS)}")
        return P._msg(P.ADDR_SET, (arg, P.S), (float(value), P.F))
    if cmd == "record":
        if arg is None:
            raise SystemExit("record needs a tag or 'off'")
        return P._msg(P.ADDR_RECORD, (0, P.I)) if arg.lower() in OFF_WORDS else P._msg(P.ADDR_RECORD, (arg, P.S))
    if cmd == "learn-bg":
        return P._msg(P.ADDR_LEARN_BG, (int(arg), P.I)) if arg else P._msg(P.ADDR_LEARN_BG)
    if cmd == "reload-calib":
        return P._msg(P.ADDR_RELOAD_CALIB)
    raise SystemExit(f"unknown command {cmd}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "learn-bg", "reload-calib", "set"])
    ap.add_argument("arg", nargs="?", default=None)
    ap.add_argument("value", nargs="?", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ctrl-port", type=int, default=P.PORT_TRACKER_CTRL)
    a = ap.parse_args(argv)
    dgram = message(a.cmd, a.arg, a.value)
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(dgram, (a.host, a.ctrl_port))
    print(f"sent {a.cmd} {a.arg or ''} {a.value or ''} -> {a.host}:{a.ctrl_port}".replace("  ", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
