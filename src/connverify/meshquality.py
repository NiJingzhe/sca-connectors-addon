"""Deterministic mesh quality gate for linear tetrahedral meshes.

Every metric is pure geometry over node coordinates (numpy, vectorized), so
two runs over the same mesh produce identical numbers:

- **signed volume** ``V = det[p1-p0 | p2-p0 | p3-p0] / 6`` — non-positive
  means an inverted or degenerate element: the stiffness integration is
  invalid, the mesh must not be solved;
- **aspect ratio** = normalized radius ratio ``R_circum / (3 r_in)`` — 1.0
  for a regular tetrahedron. Industry bands for structural FEM: <= 3 good,
  3-5 acceptable, 5-10 accuracy risk, > 10 must be corrected;
- **minSICN** from gmsh (``getElementQualities``) — the mesher's own signed
  inverse condition number: 1 ideal, near 0 degenerate, <= 0 inverted. It is
  a cross-check, not the primary gate, so hand-built meshes without it still
  verify deterministically.

The gate is deliberately open-source-dependency-free: gmsh (already the
mesher) supplies SICN, numpy supplies the rest. VTK/pyvista cell-quality was
considered and rejected as a whole-tree dependency for two numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .mesh_model import Tet

ASPECT_RATIO_WARN = 5.0    # above this an element risks stress accuracy
ASPECT_RATIO_FAIL = 10.0   # above this the gate fails the whole mesh
FRACTION_ABOVE_WARN = 0.05 # more than 5% of tets in the warn band warns
SICN_WARN = 0.05           # gmsh signed inverse condition number floor

# local face node triples of a tet (all four faces)
_TET_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


@dataclass(frozen=True)
class MeshQualityStats:
    """Aggregated quality of one mesh plus the gate verdict."""

    tet_count: int
    inverted_tets: int
    volume_min_mm3: float
    volume_mean_mm3: float
    volume_max_mm3: float
    aspect_ratio_min: float
    aspect_ratio_mean: float
    aspect_ratio_max: float
    aspect_ratio_p95: float
    fraction_above_ar_warn: float
    sicn_min: Optional[float]           # None when gmsh sampling unavailable
    sicn_mean: Optional[float]
    worst_ar_element_centroid_mm: Tuple[float, float, float]
    failures: Tuple[str, ...]
    warnings: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "tet_count": self.tet_count,
            "inverted_tets": self.inverted_tets,
            "volume_min_mm3": _clean(self.volume_min_mm3),
            "volume_mean_mm3": _clean(self.volume_mean_mm3),
            "volume_max_mm3": _clean(self.volume_max_mm3),
            "aspect_ratio_min": _clean(self.aspect_ratio_min),
            "aspect_ratio_mean": _clean(self.aspect_ratio_mean),
            "aspect_ratio_max": _clean(self.aspect_ratio_max),
            "aspect_ratio_p95": _clean(self.aspect_ratio_p95),
            "fraction_above_ar_warn": _clean(self.fraction_above_ar_warn),
            "sicn_min": _clean(self.sicn_min),
            "sicn_mean": _clean(self.sicn_mean),
            "worst_ar_element_centroid_mm": [
                _clean(v) for v in self.worst_ar_element_centroid_mm],
            "aspect_ratio_warn_limit": ASPECT_RATIO_WARN,
            "aspect_ratio_fail_limit": ASPECT_RATIO_FAIL,
            "passed": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def compute_mesh_quality(
    nodes: Dict[int, Tuple[float, float, float]],
    tets: Tuple[Tet, ...],
    *,
    sicn: Optional[Iterable[float]] = None,
) -> MeshQualityStats:
    """Compute all metrics and evaluate the gate.

    ``sicn`` (optional) are gmsh minSICN values aligned with ``tets`` order.
    """
    if not tets:
        return MeshQualityStats(
            tet_count=0, inverted_tets=0,
            volume_min_mm3=0.0, volume_mean_mm3=0.0, volume_max_mm3=0.0,
            aspect_ratio_min=0.0, aspect_ratio_mean=0.0,
            aspect_ratio_max=0.0, aspect_ratio_p95=0.0,
            fraction_above_ar_warn=0.0, sicn_min=None, sicn_mean=None,
            worst_ar_element_centroid_mm=(0.0, 0.0, 0.0),
            failures=("mesh has no elements — nothing to verify",),
            warnings=(),
        )

    coords = _tet_coordinates(nodes, tets)
    volumes = tet_signed_volumes_array(coords)
    aspect = tet_aspect_ratios_array(coords)

    inverted = int(np.count_nonzero(volumes <= 0.0))
    finite_aspect = aspect[np.isfinite(aspect)]
    worst_index = int(np.argmax(aspect))
    worst_centroid = tuple(
        float(v) for v in coords[worst_index].mean(axis=0))
    fraction_warn = float(np.mean(aspect > ASPECT_RATIO_WARN))

    sicn_values = None
    if sicn is not None:
        sicn_values = np.asarray(tuple(sicn), dtype=float)
        if sicn_values.size != len(tets):
            sicn_values = None

    failures: list[str] = []
    warnings: list[str] = []
    if inverted:
        worst_abs = float(np.min(np.abs(volumes)))
        failures.append(
            f"{inverted} inverted or degenerate element(s) "
            f"(signed volume <= 0; smallest magnitude {worst_abs:.3e} mm³) — "
            "the stiffness integral is invalid; remesh with a smaller target "
            "size or repair the local geometry"
        )
    ar_max = float(np.max(aspect))
    if ar_max > ASPECT_RATIO_FAIL:
        failures.append(
            f"max aspect ratio {ar_max:.1f} at "
            f"({worst_centroid[0]:.1f}, {worst_centroid[1]:.1f}, "
            f"{worst_centroid[2]:.1f}) exceeds {ASPECT_RATIO_FAIL:.0f} — "
            "remesh with a smaller target size or repair the local geometry"
        )
    if fraction_warn > FRACTION_ABOVE_WARN:
        warnings.append(
            f"{fraction_warn:.1%} of elements exceed aspect ratio "
            f"{ASPECT_RATIO_WARN:.0f} — local stress accuracy may degrade"
        )
    sicn_min = sicn_mean = None
    if sicn_values is not None:
        sicn_min = float(np.min(sicn_values))
        sicn_mean = float(np.mean(sicn_values))
        if sicn_min <= 0.0 and not inverted:
            failures.append(
                f"gmsh minSICN {sicn_min:.3f} <= 0 marks an inverted element "
                "the volume check did not catch — remesh before solving"
            )
        elif sicn_min < SICN_WARN:
            warnings.append(
                f"min SICN {sicn_min:.3f} below {SICN_WARN} — near-degenerate "
                "element present; expect local stress noise"
            )
    if not np.all(np.isfinite(aspect)):
        warnings.append(
            "non-finite aspect ratio computed for at least one element "
            "(near-coplanar nodes); the volume check governs the gate"
        )

    return MeshQualityStats(
        tet_count=len(tets),
        inverted_tets=inverted,
        volume_min_mm3=float(np.min(volumes)),
        volume_mean_mm3=float(np.mean(volumes)),
        volume_max_mm3=float(np.max(volumes)),
        aspect_ratio_min=float(np.min(finite_aspect)) if finite_aspect.size else 0.0,
        aspect_ratio_mean=float(np.mean(finite_aspect)) if finite_aspect.size else 0.0,
        aspect_ratio_max=ar_max,
        aspect_ratio_p95=float(np.percentile(finite_aspect, 95))
        if finite_aspect.size else 0.0,
        fraction_above_ar_warn=fraction_warn,
        sicn_min=sicn_min,
        sicn_mean=sicn_mean,
        worst_ar_element_centroid_mm=worst_centroid,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------------ geometry

def tet_signed_volumes(nodes, tets) -> "np.ndarray":
    """Signed volumes in mm³, aligned with ``tets`` (positive = valid)."""
    return tet_signed_volumes_array(_tet_coordinates(nodes, tets))


def tet_aspect_ratios(nodes, tets) -> "np.ndarray":
    """Normalized radius ratios R/(3r), aligned with ``tets`` (1 = regular)."""
    return tet_aspect_ratios_array(_tet_coordinates(nodes, tets))


def _tet_coordinates(nodes, tets) -> "np.ndarray":
    index = {node_id: i for i, node_id in enumerate(nodes)}
    table = np.array(
        [[index[n] for n in tet.node_ids] for tet in tets], dtype=int)
    points = np.array([nodes[node_id] for node_id in nodes], dtype=float)
    return points[table]  # (n_tets, 4, 3)


def tet_signed_volumes_array(coords: "np.ndarray") -> "np.ndarray":
    edge1 = coords[:, 1] - coords[:, 0]
    edge2 = coords[:, 2] - coords[:, 0]
    edge3 = coords[:, 3] - coords[:, 0]
    return np.einsum("ij,ij->i", edge1, np.cross(edge2, edge3)) / 6.0


def tet_aspect_ratios_array(coords: "np.ndarray") -> "np.ndarray":
    """R_circum / (3 r_in) per tet — 1.0 for the regular tetrahedron.

    r_in = 3|V| / S (total surface area), so the ratio equals
    R·S / (9|V|); coplanar nodes make |V| -> 0 and the ratio -> inf, which
    the caller treats as a failure-grade distortion.
    """
    volumes = np.abs(tet_signed_volumes_array(coords))
    surface = np.zeros(len(coords))
    for i, j, k in _TET_FACES:
        edge_a = coords[:, j] - coords[:, i]
        edge_b = coords[:, k] - coords[:, i]
        surface += 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    radii = _circumradii(coords)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = radii * surface / (9.0 * volumes)
    return aspect


def _circumradii(coords: "np.ndarray") -> "np.ndarray":
    """Circumscribed-sphere radii by solving |d| from 2 d·(pi-p0) = |pi-p0|²."""
    rows = coords[:, 1:] - coords[:, :1]                     # (n, 3, 3)
    rhs = 0.5 * np.einsum("nij,nij->ni", rows, rows)         # (n, 3)
    dets = np.linalg.det(rows)
    scale = np.max(np.abs(rows), axis=(1, 2))
    singular = np.abs(dets) <= 1e-12 * np.maximum(scale ** 3, 1e-30)
    radii = np.full(len(coords), np.inf)
    solvable = ~singular
    if np.any(solvable):
        offsets = np.linalg.solve(rows[solvable], rhs[solvable, :, None])[:, :, 0]
        radii[solvable] = np.linalg.norm(offsets, axis=1)
    return radii


def _clean(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, 6)
