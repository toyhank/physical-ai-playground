from __future__ import annotations

import numpy as np

from app.simulation.camera import FixedCameraProjector


def test_projection_round_trip_is_below_two_centimeters() -> None:
    projector = FixedCameraProjector()
    samples = [(-0.32, 0.02), (-0.18, 0.22), (0.0, 0.10), (0.20, 0.30), (0.34, 0.04)]
    for x, y in samples:
        source = np.asarray((x, y, projector.table_z))
        normalized_x, normalized_y = projector.world_to_normalized(source)
        recovered = projector.image_point_to_world(normalized_x, normalized_y)
        assert np.linalg.norm(recovered - source) < 0.02


def test_projection_rejects_invalid_normalized_coordinates() -> None:
    projector = FixedCameraProjector()
    try:
        projector.image_point_to_world(1001, 500)
    except ValueError as error:
        assert "0..1000" in str(error)
    else:
        raise AssertionError("invalid coordinate was accepted")

