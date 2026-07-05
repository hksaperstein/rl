"""Geometric shape classification from a per-object 3D point cloud.

Classifies a cluster of world-frame points (as produced by segmenting a
single top-down depth image) into one of: cube, rectangular_prism, sphere,
wedge, or unknown (too few points to say). See module docstring reasoning
in docs/superpowers/specs/2026-07-04-ar4-perception-integration-design.md.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Thresholds tuned and verified against this project's actual object sizes
# (tasks/ar4/objects_cfg.py): cube 18mm, rectangular prism 16x16x30mm,
# sphere 9mm radius. See the design doc's "Approaches considered" section -
# revisit these if real camera noise proves them too tight/loose.
HEIGHT_THRESHOLD = 0.024  # m, midpoint between the cube (0.018m) and prism (0.030m) heights
PLANARITY_RESIDUAL_THRESHOLD = 0.0008  # m, above this the top surface reads as curved, not flat
TILT_THRESHOLD_RAD = np.radians(15.0)  # above this a flat surface reads as a slanted (wedge) face
CIRCULARITY_THRESHOLD = 0.7  # footprint roundness (1.0 = circle) above which a curved cap reads as a sphere

UNKNOWN = "unknown"
CUBE = "cube"
RECTANGULAR_PRISM = "rectangular_prism"
SPHERE = "sphere"
WEDGE = "wedge"


@dataclass
class ShapeClassification:
    label: str
    height: float
    planarity_residual: float
    tilt_rad: float
    circularity: float


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fits a plane to `points` (P, 3) via SVD. Returns (unit normal, residual std)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    residual = float(np.std(centered @ normal))
    return normal, residual


def _footprint_circularity(points_xy: np.ndarray) -> float:
    """4*pi*area/perimeter^2 of the convex hull of the XY footprint. 1.0 for a circle."""
    if len(points_xy) < 3:
        return 0.0
    try:
        hull = ConvexHull(points_xy)
    except QhullError:
        return 0.0
    area = hull.volume  # `volume` is the 2D area for a 2D ConvexHull
    hull_points = points_xy[hull.vertices]
    perimeter = sum(np.linalg.norm(hull_points[i] - hull_points[i - 1]) for i in range(len(hull_points)))
    if perimeter == 0.0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter**2))


def classify_shape(points: np.ndarray, ground_z: float = 0.0) -> ShapeClassification:
    """Classifies a single object's world-frame point cloud (P, 3)."""
    if points.shape[0] < 3:
        return ShapeClassification(UNKNOWN, 0.0, 0.0, 0.0, 0.0)

    height = float(points[:, 2].max() - ground_z)
    normal, residual = _fit_plane(points)
    tilt_rad = float(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0)))
    circularity = _footprint_circularity(points[:, :2])

    if residual > PLANARITY_RESIDUAL_THRESHOLD and circularity > CIRCULARITY_THRESHOLD:
        label = SPHERE
    elif tilt_rad > TILT_THRESHOLD_RAD:
        label = WEDGE
    else:
        label = CUBE if height < HEIGHT_THRESHOLD else RECTANGULAR_PRISM

    return ShapeClassification(label, height, residual, tilt_rad, circularity)
