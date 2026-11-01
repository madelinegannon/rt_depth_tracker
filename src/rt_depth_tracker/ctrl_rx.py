# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Loopback OSC control receiver shared by the tracker and the captures (one datagram -> one (address, params)
on a queue the main loop drains), plus the quit-signal hook that turns a console Ctrl-Break or SIGTERM into a
KeyboardInterrupt so the process's own cleanup runs instead of an abrupt 0xC000013A exit."""
from __future__ import annotations

import queue
import signal
import socket
import threading

from rt_depth_tracker import protocol as P


class CtrlRx:
    def __init__(self, port: int):
        self.queue: queue.Queue = queue.Queue()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.settimeout(0.2)
        self._running = True
        threading.Thread(target=self._loop, name="CtrlRx", daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            obj = P.unpack(data)
            if isinstance(obj, tuple):
                self.queue.put(obj)

    def stop(self):
        self._running = False
        self._sock.close()


def install_quit_signals():
    """SIGBREAK (Windows console Ctrl-Break) and SIGTERM raise KeyboardInterrupt in the main thread."""
    def _raise(signum, frame_):
        raise KeyboardInterrupt
    for name in ("SIGBREAK", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise)
        except (ValueError, OSError):   # not the main thread
            pass
