# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""Robot self-mask: forward kinematics from a URDF plus collision spheres from a robot-description YAML, so
the tracker can delete the robot's own points from the cloud at its current pose. Optional: with no robot in
the config, from_config returns None and nothing is masked. Reads only; never talks to a controller."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import yaml


def rpy_to_R(r, p, y) -> np.ndarray:
    """URDF rpy = fixed-axis XYZ: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def rot_z(q: float) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class RobotModel:
    def __init__(self, urdf_path: str, yaml_path: str, mount_z: float = 0.0, base_xy=(0.0, 0.0),
                 extra_spheres: dict | None = None, base_yaw_deg: float = 0.0, sphere_scale: float = 1.0):
        self.joints, self.root = self._parse_urdf(urdf_path)       # [(name, parent, child, T_origin, axis, type)], root link
        self.links = [self.root] + [j[2] for j in self.joints]
        self.n_dof = sum(1 for j in self.joints if j[5] != "fixed")
        self.spheres = self._parse_spheres(yaml_path)          # link -> (centers (n,3), radii (n,))
        if sphere_scale != 1.0:
            self.spheres = {l: (c, r * float(sphere_scale)) for l, (c, r) in self.spheres.items()}
        if extra_spheres:
            for link, lst in extra_spheres.items():
                c = np.array([s["center"] for s in lst], dtype=np.float64)
                r = np.array([s["radius"] for s in lst], dtype=np.float64)
                if link in self.spheres:
                    self.spheres[link] = (np.vstack([self.spheres[link][0], c]), np.concatenate([self.spheres[link][1], r]))
                else:
                    self.spheres[link] = (c, r)
        self.T_world_base = rot_z(np.radians(base_yaw_deg))
        self.T_world_base[:3, 3] = [base_xy[0], base_xy[1], mount_z]
        self._link_index = {n: i for i, n in enumerate(self.links)}
        self.n_spheres = int(sum(len(r) for _, r in self.spheres.values()))
        self._all_radii = np.concatenate([self.spheres[l][1] for l in self.links if l in self.spheres])

    @classmethod
    def from_config(cls, cfg, **kw) -> "RobotModel | None":
        """The robot the config describes, or None when it describes none (no urdf): the pipeline then masks nothing."""
        r = getattr(cfg, "robot", None) or {}
        urdf = kw.pop("urdf_path", None) or cfg.resolve(r.get("urdf"))
        desc = kw.pop("yaml_path", None) or cfg.resolve(r.get("robot_description"))
        if not urdf or not desc:
            return None
        # the YAML is the arm; extras are for what is bolted to it
        kw.setdefault("extra_spheres", getattr(cfg, "robot_spheres_extra", None) or None)
        kw.setdefault("sphere_scale", getattr(cfg, "robot_sphere_scale", 1.0))
        return cls(urdf, desc, mount_z=cfg.mount_z, base_xy=cfg.base_xy, base_yaw_deg=cfg.base_yaw_deg, **kw)

    @staticmethod
    def _parse_urdf(path):
        root = ET.parse(path).getroot()
        joints = []
        for j in root.findall("joint"):
            if j.get("type") not in ("revolute", "continuous", "fixed"):
                continue
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = [float(v) for v in (ax.get("xyz", "0 0 1") if ax is not None else "0 0 1").split()]
            joints.append((j.get("name"), j.find("parent").get("link"), j.find("child").get("link"),
                           make_T(rpy_to_R(*rpy), xyz), np.array(axis), j.get("type")))
        children = {j[2] for j in joints}
        roots = [j[1] for j in joints if j[1] not in children]
        base = roots[0] if roots else "base_link"
        # order by chain from the root link (a serial arm: one child per link)
        ordered, parent = [], base
        remaining = list(joints)
        while remaining:
            nxt = [j for j in remaining if j[1] == parent]
            if not nxt:
                break
            ordered.append(nxt[0])
            remaining.remove(nxt[0])
            parent = nxt[0][2]
        return ordered, base

    @staticmethod
    def _parse_spheres(path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        out = {}
        for entry in d.get("collision_spheres", []):
            for link, lst in entry.items():
                out[link] = (np.array([s["center"] for s in lst], dtype=np.float64),
                             np.array([s["radius"] for s in lst], dtype=np.float64))
        return out

    # ---- kinematics -----------------------------------------------------------------
    def link_transforms(self, q) -> dict:
        """World transforms of every link for joint vector q (rad, one per moving joint)."""
        q = np.asarray(q, dtype=np.float64)
        T = self.T_world_base.copy()
        out = {self.root: T.copy()}
        qi = 0
        for name, parent, child, T_origin, axis, jtype in self.joints:
            T = T @ T_origin
            if jtype != "fixed":
                if not np.allclose(axis, [0, 0, 1]):
                    raise NotImplementedError("only z-axis joints modelled")
                T = T @ rot_z(float(q[qi]) if qi < len(q) else 0.0)
                qi += 1
            out[child] = T.copy()
        return out

    def ee_position(self, q, link: str | None = None, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        """World position of a point `offset` in the frame of `link` (default: the last link, the flange)."""
        T = self.link_transforms(q)[link or self.links[-1]]
        return (T @ np.array([*offset, 1.0]))[:3]

    def sphere_world(self, q):
        """(centers (N,3), radii (N,)) in world for joint vector q."""
        Ts = self.link_transforms(q)
        cs, rs = [], []
        for link in self.links:
            if link not in self.spheres:
                continue
            c, r = self.spheres[link]
            T = Ts[link]
            cs.append((c @ T[:3, :3].T) + T[:3, 3])
            rs.append(r)
        return np.vstack(cs), np.concatenate(rs)

    def inside_mask(self, P: np.ndarray, q, margin: float = 0.08, pedestal_radius: float = 0.0,
                    pedestal_z_max: float | None = None) -> np.ndarray:
        """True for points within (radius + margin) of any robot sphere, or inside the pedestal cylinder."""
        P = np.asarray(P, dtype=np.float32)
        if len(P) == 0:
            return np.zeros(0, dtype=bool)
        cs, rs = self.sphere_world(q)
        cs, rs = cs.astype(np.float32), rs.astype(np.float32)
        # chunked so a crowd frame never allocates a giant (N x n_spheres) float64 temp
        out = np.zeros(len(P), dtype=bool)
        thr2 = (rs + margin) ** 2
        for s in range(0, len(P), 16384):
            blk = P[s:s + 16384]
            d2 = ((blk[:, None, :] - cs[None, :, :]) ** 2).sum(-1)
            out[s:s + 16384] = (d2 < thr2[None, :]).any(axis=1)
        if pedestal_radius > 0:
            zmax = self.T_world_base[2, 3] + 0.05 if pedestal_z_max is None else pedestal_z_max
            bx, by = self.T_world_base[0, 3], self.T_world_base[1, 3]
            ped = ((P[:, 0] - bx) ** 2 + (P[:, 1] - by) ** 2 < (pedestal_radius + margin) ** 2) & (P[:, 2] < zmax)
            out |= ped
        return out

    def surface_samples(self, q, n_per_sphere: int = 40, rng=None) -> np.ndarray:
        """Points on the sphere surfaces (for tests and the refine ICP)."""
        rng = rng or np.random.default_rng(0)
        cs, rs = self.sphere_world(q)
        d = rng.normal(size=(len(cs), n_per_sphere, 3))
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return (cs[:, None, :] + d * rs[:, None, None]).reshape(-1, 3)
