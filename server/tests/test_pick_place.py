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

