"""Benchmark every colored-cube semantic skill across randomized scenes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app.simulation.mujoco_engine import CUBE_NAMES, MujocoEngine  # noqa: E402


def main() -> None:
    engine = MujocoEngine(seed=0)
    failures: list[dict[str, object]] = []
    successes = 0
    for seed in range(25):
        for object_id in CUBE_NAMES:
            engine.reset(seed)
            engine._latch_world_model()
            pick = engine.pick_object(object_id)
            place = engine.place_object("blue_box") if pick.success else pick
            passed = pick.success and place.success and engine.verify_task()
            successes += int(passed)
            if not passed:
                failures.append(
                    {
                        "seed": seed,
                        "object_id": object_id,
                        "pick_error": pick.error,
                        "place_error": place.error,
                    }
                )

    trials = 25 * len(CUBE_NAMES)
    print(
        json.dumps(
            {
                "trials": trials,
                "successes": successes,
                "success_rate": successes / trials,
                "failures": failures[:5],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if successes == trials else 1)


if __name__ == "__main__":
    main()
