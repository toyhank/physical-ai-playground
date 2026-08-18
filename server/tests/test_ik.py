from __future__ import annotations

import numpy as np

from app.robot.ik import PlanarDampedLeastSquaresIK


def test_damped_least_squares_reaches_workspace_targets() -> None:
    solver = PlanarDampedLeastSquaresIK()
    for target in (np.asarray((0.18, 0.42)), np.asarray((-0.25, 0.48)), np.asarray((0.32, 0.34))):
        result = solver.solve(target)
        assert result.success, result
        assert result.error <= solver.tolerance
        assert np.all(result.joints >= solver.lower)
        assert np.all(result.joints <= solver.upper)

