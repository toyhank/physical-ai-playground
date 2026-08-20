from __future__ import annotations

import numpy as np
import pytest

from app.simulation.mujoco_engine import MujocoEngine


def test_deterministic_pick_place_100_random_resets() -> None:
    successes = 0
    failures: list[tuple[int, list[dict]]] = []
    engine = MujocoEngine(seed=0, enable_mujoco=True)
    for seed in range(100):
        engine.reset(seed)
        results = engine.deterministic_pick_place()
        if engine.verify_task():
            successes += 1
        else:
            failures.append((seed, results))
    assert successes >= 95, {"successes": successes, "failures": failures[:5]}


@pytest.mark.parametrize(
    ("object_id", "seed"),
    [("red_cube", 7), ("green_cube", 11), ("yellow_cube", 19), ("purple_cube", 29)],
)
def test_semantic_pick_and_place_skills(object_id: str, seed: int) -> None:
    engine = MujocoEngine(seed=seed, enable_mujoco=True)
    engine.camera_png_base64(camera="scene", latch_observation=True)
    original_other_positions = {
        name: position.copy()
        for name, position in engine.cube_positions.items()
        if name != object_id
    }
    pick = engine.pick_object(object_id)
    assert pick.success, pick.as_dict()
    place = engine.place_object("blue_box")
    assert place.success, place.as_dict()
    assert engine.verify_task()
    assert engine.task_object_id == object_id
    for name, original in original_other_positions.items():
        assert np.linalg.norm(engine.cube_positions[name][:2] - original[:2]) < 0.02


def test_reset_randomizes_four_separated_colored_cubes() -> None:
    engine = MujocoEngine(seed=0, enable_mujoco=False)
    layouts = []
    for seed in range(5):
        engine.reset(seed)
        assert set(engine.cube_positions) == {
            "red_cube", "green_cube", "yellow_cube", "purple_cube"
        }
        points = list(engine.cube_positions.values())
        assert all(
            np.linalg.norm(left[:2] - right[:2]) >= 0.105
            for index, left in enumerate(points)
            for right in points[index + 1 :]
        )
        layouts.append(tuple(tuple(position[:2]) for position in points))
    assert len(set(layouts)) == 5
