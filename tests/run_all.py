# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Run every suite as a subprocess and aggregate. Usage: .venv/bin/python tests/run_all.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ["test_protocol", "test_capture", "test_geometry", "test_calib_steps", "test_aim_side", "test_tracking", "test_behavior",
          "test_runtime", "test_rgb", "test_record", "test_pacing", "test_viewer", "test_zones"]


def main():
    results = {}
    for s in SUITES:
        r = subprocess.run([sys.executable, os.path.join(HERE, s + ".py")])
        results[s] = r.returncode == 0
    print("\n== rt_depth_tracker tests ==")
    for s, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {s}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
