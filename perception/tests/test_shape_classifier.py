import numpy as np

from perception.shape_classifier import CUBE, RECTANGULAR_PRISM, SPHERE, WEDGE, classify_shape

RNG = np.random.default_rng(0)


def _flat_square_top(height: float, half_extent: float = 0.008, n: int = 200) -> np.ndarray:
    xy = RNG.uniform(-half_extent, half_extent, size=(n, 2))
    z = height + RNG.normal(0.0, 0.0002, size=n)
    return np.column_stack([xy, z])


def test_classifies_short_flat_top_as_cube():
    points = _flat_square_top(height=0.018)
    assert classify_shape(points, ground_z=0.0).label == CUBE


def test_classifies_tall_flat_top_as_rectangular_prism():
    points = _flat_square_top(height=0.030)
    assert classify_shape(points, ground_z=0.0).label == RECTANGULAR_PRISM


def test_classifies_curved_round_cap_as_sphere():
    n = 300
    radius = 0.009
    center_z = radius
    theta = RNG.uniform(0, 2 * np.pi, size=n)
    r = radius * np.sqrt(RNG.uniform(0, 1, size=n)) * 0.9
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = center_z + np.sqrt(np.clip(radius**2 - r**2, 0.0, None))
    points = np.column_stack([x, y, z])
    assert classify_shape(points, ground_z=0.0).label == SPHERE


def test_classifies_tilted_plane_as_wedge():
    n = 200
    xy = RNG.uniform(-0.008, 0.008, size=(n, 2))
    z = 0.015 + xy[:, 0] * np.tan(np.radians(30.0))
    points = np.column_stack([xy, z])
    assert classify_shape(points, ground_z=0.0).label == WEDGE


def test_too_few_points_is_unknown():
    points = np.zeros((2, 3))
    result = classify_shape(points, ground_z=0.0)
    assert result.label == "unknown"
