# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""3D-printable calibration target for the UFACTORY 850: a 150 mm hollow sphere on a stem with an
ISO 9409-1-50-4-M6 flange adapter. Generates STLs (trimesh + manifold booleans) and a preview.

  .venv/bin/python calib_target/make_calib_target.py [--out calib_target/stl]

Parts (print all in matte white PETG or PLA, 0.2 mm layers):
  1. flange_stem.stl        flange disc + spigot + stem            print stem-up, no supports
  2. sphere_lower.stl       lower hemisphere + lip + external stem collar   print rim-down (flat rim on bed)
  3. sphere_upper.stl       upper hemisphere                                 print rim-down; supports optional above ~70 deg
  4. foam_hub.stl           ALTERNATIVE to 2+3: plug for a bought Ø150 polystyrene ball (drill Ø35 x 40 deep)
Assembly: bolt flange to the robot (4x M6x16 socket head + Ø6 dowel in the +X hole), push the stem into the
collar until the thumbscrew meets the dimple, tighten the M4 thumbscrew (heat-set M4 insert in the collar, or
tap the Ø3.4 pilot), slide the upper hemisphere over the lip, tape/glue.

Ball centre = BALL_CENTER_Z above the flange face, on the flange (link6) z axis:
  python -m rt_depth_tracker.calib_tool sphere --ball-r 0.075 --offset 0,0,0.180
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import trimesh
from trimesh.creation import cylinder, icosphere

# ---- parameters (mm) ----------------------------------------------------------------------------
BALL_D = 150.0
WALL = 1.6                     # 4 perimeters at 0.4 mm; plenty for a 150 mm shell, ~35% less time than 2.4
BALL_CENTER_Z = 180.0          # above the flange face (link6 frame z)
LIP_H = 6.0
LIP_CLEAR = 0.25               # radial clearance lip vs upper shell
STEM_R = 11.0                  # Ø22 stem
STEM_INSERT = 25.0             # stem depth inside the ball (above the shell surface)
COLLAR_H = 20.0                # collar protruding BELOW the sphere: thumbscrew lives here, outside the shell
SOCKET_CLEAR = 0.3             # radial clearance socket vs stem
BOSS_R = STEM_R + 5.0
LOCK = "insert"                # "insert": M4 heat-set pocket (Ø5.6 x 6) + Ø4.3 through; "tap": Ø3.4 pilot for M4
INSERT_D, INSERT_H, SCREW_D, TAP_D = 5.6, 6.0, 4.3, 3.4
DIMPLE_D, DIMPLE_H = 8.0, 3.0  # cone dimple in the stem for the screw tip (locks height + rotation)
FOAM_PLUG_D, FOAM_PLUG_L, FOAM_DISC_D, FOAM_DISC_T = 35.0, 40.0, 60.0, 3.0
FLANGE_D = 63.0                # ISO 9409-1 size 50 outer diameter
FLANGE_T = 8.0
PCD = 50.0
BOLT_HOLE_D = 6.6              # M6 clearance
CBORE_D = 11.0                 # M6 socket head + clearance
CBORE_T = 4.0
PIN_HOLE_D = 6.2               # Ø6 H7 dowel clearance, at +X on the PCD (ISO 9409-1)
SPIGOT_D = 31.2                # into the plate's Ø31.5 H7 recess; set SPIGOT_H = 0 if the flange has no recess
SPIGOT_H = 3.0
FILLET_R = 18.0                # cone base of the stem
FILLET_H = 10.0
SPHERE_SUBDIV = 5              # icosphere subdivisions (5 -> 20480 faces, smooth at 150 mm)

R = BALL_D / 2
STEM_TOP_Z = BALL_CENTER_Z - R + STEM_INSERT
STEM_LEN = STEM_TOP_Z - FLANGE_T
SCREW_Z = BALL_CENTER_Z - R - COLLAR_H / 2      # flange frame: middle of the external collar


def cyl(r, h, z0, x=0.0, y=0.0, sections=96):
    c = cylinder(radius=r, height=h, sections=sections)
    c.apply_translation([x, y, z0 + h / 2])
    return c


def cone(r0, r1, h, z0, sections=96):
    c = trimesh.creation.cone(radius=r0, height=h, sections=sections) if r1 == 0 else None
    if c is None:
        # frustum via truncated cone: build as a convex hull of two circles
        a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        lo = np.stack([r0 * np.cos(a), r0 * np.sin(a), np.full(sections, z0)], 1)
        hi = np.stack([r1 * np.cos(a), r1 * np.sin(a), np.full(sections, z0 + h)], 1)
        c = trimesh.convex.convex_hull(np.vstack([lo, hi]))
    return c


def halfspace_box(z_min, z_max, size=400.0):
    b = trimesh.creation.box(extents=[size, size, z_max - z_min])
    b.apply_translation([0, 0, (z_min + z_max) / 2])
    return b


def flange_stem() -> trimesh.Trimesh:
    body = cyl(FLANGE_D / 2, FLANGE_T, 0.0)
    if SPIGOT_H > 0:
        body = body.union(cyl(SPIGOT_D / 2, SPIGOT_H + 0.01, -SPIGOT_H))
    body = body.union(cone(FILLET_R, STEM_R, FILLET_H, FLANGE_T - 0.01))
    body = body.union(cyl(STEM_R, STEM_LEN + 0.01, FLANGE_T - 0.01))
    # top chamfer on the stem so it enters the socket
    body = body.difference(cone(STEM_R + 5, STEM_R - 1.5, 1.5, STEM_TOP_Z - 1.5 + 0.01).difference(cyl(STEM_R - 1.5, 3, STEM_TOP_Z - 2)))
    cutters = []
    for k in range(4):
        a = np.radians(45 + 90 * k)
        x, y = PCD / 2 * np.cos(a), PCD / 2 * np.sin(a)
        cutters.append(cyl(BOLT_HOLE_D / 2, FLANGE_T + SPIGOT_H + 2, -SPIGOT_H - 1, x, y, 48))
        cutters.append(cyl(CBORE_D / 2, CBORE_T + 1, FLANGE_T - CBORE_T, x, y, 48))
    cutters.append(cyl(PIN_HOLE_D / 2, FLANGE_T + SPIGOT_H + 2, -SPIGOT_H - 1, PCD / 2, 0.0, 48))
    # conical dimple on the +X side of the stem where the collar thumbscrew seats
    dimple = cone(DIMPLE_D / 2 + 1.0, 0.5, DIMPLE_H + 1.0, 0.0, 32)
    dimple.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    dimple.apply_translation([STEM_R + 1.0, 0, SCREW_Z])
    cutters.append(dimple)
    return body.difference(trimesh.util.concatenate(cutters))


def sphere_shell() -> trimesh.Trimesh:
    outer = icosphere(SPHERE_SUBDIV, R)
    inner = icosphere(SPHERE_SUBDIV, R - WALL)
    return outer.difference(inner)


def lock_cutters(z, sections=32):
    """Radial M4 lock at height z (part frame), screw axis along +X, from the collar surface inward."""
    out = []
    if LOCK == "insert":
        pocket = cyl(INSERT_D / 2, INSERT_H + 0.5, 0.0, 0, 0, sections)
        pocket.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        pocket.apply_translation([BOSS_R - INSERT_H, 0, z])
        out.append(pocket)
        d = SCREW_D
    else:
        d = TAP_D
    through = cyl(d / 2, BOSS_R + 2, 0.0, 0, 0, sections)
    through.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
    through.apply_translation([BOSS_R + 1, 0, z])
    out.append(through)
    return out


def sphere_lower(shell: trimesh.Trimesh) -> trimesh.Trimesh:
    lower = shell.intersection(halfspace_box(-R - 1, 0.0))
    lip = cyl(R - WALL - LIP_CLEAR, LIP_H, -0.01).difference(cyl(R - WALL - LIP_CLEAR - 2.0, LIP_H + 1, -0.5))
    boss_top = -R + STEM_INSERT + 3.0
    collar = cyl(BOSS_R, boss_top - (-R - COLLAR_H), -R - COLLAR_H)           # runs from below the shell to inside it
    part = lower.union(lip).union(collar)
    socket = cyl(STEM_R + SOCKET_CLEAR, COLLAR_H + STEM_INSERT + 1, -R - COLLAR_H - 1)
    part = part.difference(socket)
    for c in lock_cutters(-R - COLLAR_H / 2):
        part = part.difference(c)
    return part


def foam_hub() -> trimesh.Trimesh:
    """Plug for a bought Ø150 polystyrene ball: disc sits on the ball surface (z=0 here = sphere bottom),
    plug goes into a drilled Ø35 x 40 hole, stem socket + external collar identical to sphere_lower."""
    disc = cyl(FOAM_DISC_D / 2, FOAM_DISC_T, 0.0)
    plug = cyl(FOAM_PLUG_D / 2 - 0.2, FOAM_PLUG_L, FOAM_DISC_T - 0.01)
    collar = cyl(BOSS_R, COLLAR_H + 0.01, -COLLAR_H)
    part = disc.union(plug).union(collar)
    socket = cyl(STEM_R + SOCKET_CLEAR, COLLAR_H + FOAM_DISC_T + STEM_INSERT + 1, -COLLAR_H - 1)
    part = part.difference(socket)
    for c in lock_cutters(-COLLAR_H / 2):
        part = part.difference(c)
    return part


def sphere_upper(shell: trimesh.Trimesh) -> trimesh.Trimesh:
    return shell.intersection(halfspace_box(0.0, R + 1))


def preview(parts: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    colors = {"flange_stem": "#888", "sphere_lower": "#4aa3ff", "sphere_upper": "#ffb347", "foam_hub": "#7c7"}
    for ax_, exploded in ((ax, False), (ax2, True)):
        for name, m in parts.items():
            if name == "foam_hub":
                continue
            mm = m.copy()
            if name.startswith("sphere"):
                mm.apply_translation([0, 0, BALL_CENTER_Z + (40 if exploded and name == "sphere_upper" else 0)])
            if exploded and name == "flange_stem":
                mm.apply_translation([0, 0, -40])
            v, f = mm.vertices, mm.faces
            step = max(1, len(f) // 4000)
            ax_.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f[::step], color=colors[name], alpha=0.85, linewidth=0)
        ax_.set_box_aspect([1, 1, 2.2])
        ax_.set_xlim(-90, 90)
        ax_.set_ylim(-90, 90)
        ax_.set_zlim(-50, 300)
        ax_.set_title("assembled" if not exploded else "exploded")
        ax_.set_xlabel("x (mm)")
        ax_.set_zlabel("z from flange face (mm)")
    fig.suptitle(f"Calibration target: Ø{BALL_D:.0f} sphere, centre {BALL_CENTER_Z:.0f} mm above the UF850 flange")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stl"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    shell = sphere_shell()
    parts = {"flange_stem": flange_stem(), "sphere_lower": sphere_lower(shell), "sphere_upper": sphere_upper(shell),
             "foam_hub": foam_hub()}
    density = 1.27e-3  # g/mm^3 PETG
    total = 0.0
    for name, m in parts.items():
        m.merge_vertices()
        ok = m.is_watertight and m.is_winding_consistent
        vol = abs(m.volume)
        total += vol
        m.export(os.path.join(a.out, f"{name}.stl"))
        ext = m.bounds[1] - m.bounds[0]
        print(f"{name:14s} watertight={ok} faces={len(m.faces):6d} bbox={np.round(ext, 1)} mm  volume={vol / 1000:.1f} cm3  ~{vol * density:.0f} g")
    assembly = trimesh.util.concatenate([parts["flange_stem"], parts["sphere_lower"].copy().apply_translation([0, 0, BALL_CENTER_Z]),
                                    parts["sphere_upper"].copy().apply_translation([0, 0, BALL_CENTER_Z])])
    # clearance check: thumbscrew head (Ø12 x 8 tall, at the collar) vs the sphere surface above it
    head_top_z = BALL_CENTER_Z - R - COLLAR_H / 2 + 6.0          # flange frame, top edge of a Ø12 head
    r_head = BOSS_R + 8.0
    sphere_z_at_r = BALL_CENTER_Z - np.sqrt(R ** 2 - r_head ** 2)
    print(f"thumbscrew clearance to shell at r={r_head:.0f} mm: {sphere_z_at_r - head_top_z:.1f} mm (want > 3)")
    assembly.export(os.path.join(a.out, "assembly_preview.stl"))
    preview(parts, os.path.join(a.out, "preview.png"))
    print(f"total (excl. foam_hub) ~{(total - abs(parts['foam_hub'].volume)) * density:.0f} g PETG; ball centre offset in link6 frame = (0, 0, {BALL_CENTER_Z / 1000:.3f}) m; "
          f"stem {STEM_LEN:.0f} mm; overall height {BALL_CENTER_Z + R:.0f} mm above the flange face")
    return 0


if __name__ == "__main__":
    sys.exit(main())
