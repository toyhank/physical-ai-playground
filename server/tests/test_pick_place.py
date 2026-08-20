from __future__ import annotations

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


def test_semantic_pick_and_place_skills() -> None:
    engine = MujocoEngine(seed=7, enable_mujoco=True)
    engine.camera_png_base64(camera="scene", latch_observation=True)
    pick = engine.pick_object("red_cube")
    assert pick.success, pick.as_dict()
    place = engine.place_object("blue_box")
    assert place.success, place.as_dict()
    assert engine.verify_task()
