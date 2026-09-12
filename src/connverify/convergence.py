"""Mesh-independence (convergence) analysis over recorded QoI series.

Standard h-refinement evidence, computed deterministically from the
quantity-of-interest series a study run produces:

- **adjacent relative change** ΔQ between successively refined meshes — the
  primary pass/fail screen against the user's tolerance;
- **observed order** p_obs = ln|ΔQ_coarse/ΔQ_fine| / ln r  (needs three
  geometrically graded meshes);
- **Richardson extrapolation** Q_ext = Q_fine + (Q_fine - Q_medium)/(r^p - 1);
- **grid convergence index** GCI = Fs·|Q_fine - Q_medium|/|Q_fine|/(r^p - 1)
  with the standard safety factor Fs = 1.25 — an estimate of the remaining
  discretization uncertainty, not a re-measurement.

Behaviour guards: oscillatory or non-monotone differences, non-geometric
size families, and monotonically *rising* peaks (the sharp-re-entrant-corner
singularity signature) are flagged in ``notes`` instead of being averaged
into a false plateau. Solver convergence is a different question: here it is
assumed handled by the pipeline (direct linear solve, exit code + result
file checks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

GCI_SAFETY_FACTOR = 1.25
MAX_REASONABLE_ORDER = 8.0   # p_obs above this signals noise, not physics
_RATIO_TOLERANCE = 1e-6      # refinement ratios must match this closely


@dataclass(frozen=True)
class QoiPoint:
    """One quantity-of-interest measurement on one mesh of the family."""

    size_mm: float                                   # mesher target actually used
    node_count: int
    tet_count: int
    max_von_mises_mpa: Optional[float] = None
    safety_factor: Optional[float] = None
    max_displacement_mm: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "size_mm": self.size_mm,
            "node_count": self.node_count,
            "tet_count": self.tet_count,
            "max_von_mises_mpa": _clean(self.max_von_mises_mpa),
            "safety_factor": _clean(self.safety_factor),
            "max_displacement_mm": _clean(self.max_displacement_mm),
        }


@dataclass(frozen=True)
class CaseConvergence:
    """Mesh-independence verdict for one load case."""

    name: str
    points: Tuple[QoiPoint, ...]                 # coarse -> fine
    adjacent_deltas_pct: Tuple[float, ...]       # von Mises deltas, coarse -> fine
    displacement_deltas_pct: Tuple[float, ...]
    max_adjacent_delta_pct: float
    converged: bool
    observed_order: Optional[float]
    richardson_extrapolated_mpa: Optional[float]
    gci_fine_pct: Optional[float]
    extrapolation_reliable: bool
    notes: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "points": [p.to_dict() for p in self.points],
            "adjacent_deltas_pct": [_clean(d) for d in self.adjacent_deltas_pct],
            "displacement_deltas_pct": [
                _clean(d) for d in self.displacement_deltas_pct],
            "max_adjacent_delta_pct": _clean(self.max_adjacent_delta_pct),
            "converged": self.converged,
            "observed_order": _clean(self.observed_order),
            "richardson_extrapolated_mpa": _clean(self.richardson_extrapolated_mpa),
            "gci_fine_pct": _clean(self.gci_fine_pct),
            "extrapolation_reliable": self.extrapolation_reliable,
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------- pure math

def relative_change_pct(q_fine: float, q_coarse: float,
                        eps: float = 1e-12) -> float:
    """|ΔQ|/max(|Q_fine|, eps)·100 — anchored on the finer (better) value."""
    scale = max(abs(q_fine), eps)
    return abs(q_fine - q_coarse) / scale * 100.0


def observed_order(q3: float, q2: float, q1: float, r: float) -> Optional[float]:
    """p_obs from coarse (q3), medium (q2), fine (q1) at uniform ratio r.

    None when the differences are not strictly monotone (oscillatory or
    flat) — extrapolating from those is exactly the false plateau this
    module exists to catch.
    """
    d_coarse = q3 - q2
    d_fine = q2 - q1
    if d_coarse == 0.0 or d_fine == 0.0:
        return None
    if d_coarse * d_fine <= 0.0:
        return None
    return math.log(abs(d_coarse / d_fine)) / math.log(r)


def richardson_extrapolated(q1: float, q2: float, *, p: float, r: float) -> float:
    return q1 + (q1 - q2) / (r ** p - 1.0)


def grid_convergence_index(q1: float, q2: float, *, p: float, r: float) -> float:
    """Fine-grid GCI in percent: Fs·|ΔQ|/|Q1|/(r^p - 1)·100."""
    return GCI_SAFETY_FACTOR * (abs(q1 - q2) / max(abs(q1), 1e-12)) \
        / (r ** p - 1.0) * 100.0


# ----------------------------------------------------------------- analysis

def analyze_convergence(
    name: str,
    points: Sequence[QoiPoint],
    *,
    qoi_tolerance_pct: float,
) -> CaseConvergence:
    """Evaluate one case's QoI series (any input order) against the tolerance."""
    ordered = tuple(sorted(points, key=lambda p: -p.size_mm))  # coarse -> fine
    notes: List[str] = []

    vm = [p.max_von_mises_mpa for p in ordered]
    disp = [p.max_displacement_mm for p in ordered]

    missing = [ordered[i].size_mm for i, v in enumerate(vm) if v is None]
    if missing:
        notes.append(
            f"no data at size(s) {', '.join(f'{s:g} mm' for s in missing)} — "
            "a failed solve cannot be convergence evidence"
        )

    vm_deltas = _adjacent_deltas(vm)
    disp_deltas = _adjacent_deltas(disp)
    max_delta = max(vm_deltas) if vm_deltas else 0.0

    # the screen is the FINEST adjacent pair: a coarse mesh being far off is
    # what the study exists to show, not a failure of the fine result
    finest_vm_delta = vm_deltas[-1] if vm_deltas else None
    finest_disp_delta = disp_deltas[-1] if disp_deltas else None

    converged = (
        not missing
        and finest_vm_delta is not None
        and finest_vm_delta <= qoi_tolerance_pct
        and (finest_disp_delta is None
             or finest_disp_delta <= qoi_tolerance_pct)
    )

    if finest_disp_delta is not None and finest_disp_delta > qoi_tolerance_pct:
        notes.append(
            f"max displacement still changes {finest_disp_delta:.2f}% between "
            f"the finest pair (tolerance {qoi_tolerance_pct:g}%) — a "
            "stiffness-level quantity should settle before stress is trusted"
        )

    order = ext = gci = None
    reliable = False
    if len(ordered) >= 3 and not missing:
        sizes = [p.size_mm for p in ordered]
        ratios = [sizes[i] / sizes[i + 1] for i in range(len(sizes) - 1)]
        if any(abs(ratios[i] - ratios[0]) > _RATIO_TOLERANCE
               for i in range(1, len(ratios))):
            notes.append(
                "size family is not geometric — Richardson/GCI need a constant "
                "refinement ratio (e.g. 12 / 8 / 5.33 mm at r = 1.5)"
            )
        else:
            r = ratios[0]
            order = observed_order(vm[0], vm[1], vm[2], r)
            if order is None:
                notes.append(
                    "von Mises differences are non-monotone (oscillatory or "
                    "flat) — extrapolation skipped; check solver state or "
                    "model changes between meshes"
                )
            elif not (0.0 < order <= MAX_REASONABLE_ORDER):
                notes.append(
                    f"observed order {order:.2f} is outside (0, "
                    f"{MAX_REASONABLE_ORDER:g}] — meshes are not in the "
                    "asymptotic range; refine the whole family"
                )
            else:
                reliable = True
                ext = richardson_extrapolated(vm[2], vm[1], p=order, r=r)
                gci = grid_convergence_index(vm[2], vm[1], p=order, r=r)
    elif not missing:
        notes.append(
            "two meshes only: pass/fail is a relative-change screen; add a "
            "third size for observed order, Richardson limit and GCI"
        )

    if not missing and len(vm) >= 2 and _strictly_rising(vm) \
            and not converged:
        notes.append(
            "peak von Mises rises monotonically with refinement — the maximum "
            "may sit on a singularity (sharp re-entrant edge, point load, "
            "constrained boundary); add a fillet or judge a path-averaged "
            "stress instead of the point peak"
        )

    return CaseConvergence(
        name=name,
        points=ordered,
        adjacent_deltas_pct=tuple(vm_deltas),
        displacement_deltas_pct=tuple(disp_deltas),
        max_adjacent_delta_pct=max_delta,
        converged=converged,
        observed_order=order,
        richardson_extrapolated_mpa=ext,
        gci_fine_pct=gci,
        extrapolation_reliable=reliable,
        notes=tuple(notes),
    )


def qoi_point_from_frd(*, size_mm: float, frd, yield_strength_mpa: float,
                       node_count: int = 0, tet_count: int = 0) -> QoiPoint:
    """Extract the study's QoIs from one solved result file."""
    from .frd import FrdParseError

    max_vm = sf = max_disp = None
    try:
        _node, max_vm = frd.max_von_mises()
        if max_vm > 0.0:
            sf = yield_strength_mpa / max_vm
    except FrdParseError:
        pass
    try:
        best = 0.0
        for node_id in frd.block("DISP").values:
            best = max(best, frd.displacement_at(node_id).magnitude_mm)
        max_disp = best if best > 0.0 else None
    except FrdParseError:
        pass
    return QoiPoint(
        size_mm=float(size_mm),
        node_count=node_count or len(frd.nodes),
        tet_count=tet_count,
        max_von_mises_mpa=max_vm,
        safety_factor=sf,
        max_displacement_mm=max_disp,
    )


# ------------------------------------------------------------------ helpers

def _adjacent_deltas(values: Sequence[Optional[float]]) -> Tuple[float, ...]:
    deltas = []
    for coarse, fine in zip(values, values[1:]):
        if coarse is None or fine is None:
            continue
        deltas.append(relative_change_pct(fine, coarse))
    return tuple(deltas)


def _strictly_rising(values: Sequence[Optional[float]]) -> bool:
    known = [v for v in values if v is not None]
    return len(known) >= 2 and all(a < b for a, b in zip(known, known[1:]))


def _clean(value) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)
