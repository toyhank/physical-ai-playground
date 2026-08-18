from __future__ import annotations

import numpy as np

from app.simulation.mujoco_engine import MujocoEngine


def test_verification_uses_simulator_ground_truth() -> None:
    engine = MujocoEngine(enable_mujoco=False)
    assert not engine.verify_task()
    engine.cube_position = engine.box_position + np.asarray((0.0, 0.0, 0.03))
    engine.grasped = False
    assert engine.verify_task()
    engine.grasped = True
    assert not engine.verify_task()

