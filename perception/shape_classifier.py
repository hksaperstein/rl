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
# Reverted to the original 0.0008 after an empirical recalibration attempt
# (scripts/measure_planarity_residual.py,
# docs/superpowers/plans/2026-07-05-perception-threshold-recalibration-report.md)
# proved no single threshold value can work: the real camera has ~zero sensor
# noise, but off-center objects' segmented clusters include a genuine sliver of
# oblique-visible side wall that inflates the plane-fit residual - and at this
# camera's real object layout, the cube's own residual (~0.0029m) exceeds the
# sphere's own residual (~0.0021m), so raising the threshold to fix the cube
# only creates a new sphere-misclassification regression (measured 2/4 -> 1/4
# objects correct). The real fix is upstream (exclude side-wall points before
# the plane fit, not retune this constant) - see ROADMAP.md item 2.
PLANARITY_RESIDUAL_THRESHOLD = 0.0008  # m, above this the top surface reads as curved, not flat
TILT_THRESHOLD_RAD = np.radians(15.0)  # above this a flat surface reads as a slanted (wedge) face
CIRCULARITY_THRESHOLD = 0.7  # footprint roundness (1.0 = circle) above which a curved cap reads as a sphere

# Structural fix for the residual-ordering-inversion documented above: none of
# HEIGHT_THRESHOLD/PLANARITY_RESIDUAL_THRESHOLD/TILT_THRESHOLD_RAD can separate
# cube/rect_prism from sphere, because the segmented cluster handed to
# classify_shape genuinely contains a sliver of the object's own oblique-
# visible side wall (real 3D geometry, not sensor noise - see
# docs/superpowers/plans/2026-07-05-perception-threshold-recalibration-report.md).
# Fix: restrict the plane fit to a band near the cluster's own top before
# computing the residual/tilt, so that sliver is excluded before the SVD fit
# instead of trying to threshold around its effect afterwards.
#
# Margin derivation: started from a geometric estimate (at these objects' real
# fixed scene positions, tasks/ar4/objects_cfg.py, the camera (0, 0.31, 0.55)
# views them from roughly atan(0.20 / 0.52) ~= 21 deg off nadir, so the visible
# side-wall sliver's vertical depth below the top edge is approximately the
# object's own top-face half-width times tan(oblique_angle) ~= 8-9mm * tan(21
# deg) ~= 3-3.5mm), then empirically swept against the real camera's actual
# per-object residual/tilt at every margin from 0.5mm-10mm (one-off sweep,
# findings captured in the report below, script not kept) to find where
# cube/rect_prism/sphere separate robustly (not just barely crossing
# PLANARITY_RESIDUAL_THRESHOLD). Cube and rect_prism's
# top-band residual plateaus at ~0.0003m/0.00025m (far below threshold) for any
# margin in [1.5mm, 4mm] and only exceeds it above 5-6mm; the real sphere's
# top-band residual rises smoothly with margin and only clears the 0.0008m
# threshold with a comfortable margin (~35% headroom, vs. only ~2% at 3mm) at
# 4mm. 4mm is the largest value still inside the cube/rect_prism safe zone,
# maximizing the sphere's separation from the threshold.
#
# Caveat (does NOT fully resolve ROADMAP item 2 - see
# docs/superpowers/plans/2026-07-05-perception-sidewall-fix-report.md): this
# fixes cube/rect_prism (previously misread as sphere) and keeps sphere
# correct, but the real wedge - whose genuine top face spans nearly the
# object's whole height range, not just a thin cap - loses its real tilt
# signal once cropped to a top-of-cluster band this thin (measured tilt drops
# from ~53 deg on the full cluster to ~3.5 deg within a 4mm top band, well
# under TILT_THRESHOLD_RAD), so the wedge now misclassifies as cube. No single
# global margin fixes all four: wedge's tilt only recovers past
# TILT_THRESHOLD_RAD at a margin (~10mm) that reintroduces the cube/rect_prism
# regression. This is a real, characterized limitation, not a bug in this
# constant's tuning - the wedge needs different handling (e.g. a
# tilt-plane-fit using the full cluster gated separately from the
# curvature/residual check), which is out of this task's scope.
TOP_BAND_MARGIN = 0.004  # m, plane-fit input is restricted to this band below the cluster's own max Z

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


def _restrict_to_top_band(points: np.ndarray, margin: float) -> np.ndarray:
    """Keeps only points within `margin` of the cluster's own max Z, to exclude
    an oblique-viewing side-wall sliver before the plane fit (see
    TOP_BAND_MARGIN's derivation comment above). Falls back to the full cluster
    if the band would leave too few points for a meaningful plane fit (e.g. a
    tiny or noisy cluster) rather than risk an unstable SVD on 1-2 points."""
    max_z = points[:, 2].max()
    band = points[points[:, 2] >= max_z - margin]
    if band.shape[0] < 3:
        return points
    return band


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
    # Fit the plane (residual + tilt) only to the cluster's own top band, not
    # the full cluster - see TOP_BAND_MARGIN's derivation comment. Circularity
    # intentionally still uses the full footprint: the object's overall
    # top-down silhouette (not just its top slice) is what should read as
    # round vs. square/rectangular.
    top_band_points = _restrict_to_top_band(points, TOP_BAND_MARGIN)
    normal, residual = _fit_plane(top_band_points)
    tilt_rad = float(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0)))
    circularity = _footprint_circularity(points[:, :2])

    if residual > PLANARITY_RESIDUAL_THRESHOLD and circularity > CIRCULARITY_THRESHOLD:
        label = SPHERE
    elif tilt_rad > TILT_THRESHOLD_RAD:
        label = WEDGE
    else:
        label = CUBE if height < HEIGHT_THRESHOLD else RECTANGULAR_PRISM

    return ShapeClassification(label, height, residual, tilt_rad, circularity)
