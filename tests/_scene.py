# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 ATONATON, LLC
"""The synthetic scene the suites share: the origin on the floor, a six-axis arm (the UF850 fixture) standing on
it, a rail 1.4 m out, two cameras behind the base looking at +x. Importing this also puts src/ on the path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
FIXTURES = os.path.join(HERE, "fixtures")
ROBOT = {"urdf": os.path.join(FIXTURES, "lula_uf850.urdf"), "robot_description": os.path.join(FIXTURES, "uf850_robot_description.yaml")}
SCENE = dict(mount_z=0.0, rail_radius=1.4, rail_top_z=0.85, over_rail_inset=0.15, pedestal_radius=0.0, x_min=-0.5, robot=ROBOT)
EXAMPLE_CONFIG = os.path.join(HERE, "..", "config", "tracker.example.json")


def cfg(**kw):
    """A TrackerConfig for the scene; keyword overrides on top."""
    from rt_depth_tracker.config import TrackerConfig
    return TrackerConfig(**{**SCENE, **kw})


def robot(**kw):
    """The fixture arm standing at the origin."""
    from rt_depth_tracker.robot_model import RobotModel
    return RobotModel(ROBOT["urdf"], ROBOT["robot_description"], **kw)
