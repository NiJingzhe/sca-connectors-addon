"""Geometry-directed feedback report.

Contract with the consuming agent: every failing fact names WHERE it lives —
an interface, a BREP feature provenance (graph node that produced the face),
or coordinates — plus a concrete change to make. Never a bare pass/fail.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .conncheck import InterfaceCheckResult
from .env import VerificationEnv
from .envelope import EnvelopeCheckResult
from .frd import FrdParseError, FrdResult
from .mesh_model import Mesh

SCHEMA_VERSION = "1.0"
_HOTSPOT_COUNT = 3


@dataclass(frozen=True)
class CaseOutcome:
    name: str
    frd: Optional[FrdResult]
    error: Optional[str]
    duration_s: Optional[float]


@dataclass(frozen=True)
class Hotspot:
    node_id: int
    von_mises_mpa: float
    location_mm: Tuple[float, float, float]
    on_interface: Optional[str]
    owning_feature: Optional[Dict]


@dataclass(frozen=True)
class CaseReport:
    name: str
    passed: bool
    error: Optional[str]
    max_von_mises_mpa: Optional[float]
    safety_factor: Optional[float]
    max_displacement_mm: Optional[float]
    hotspots: Tuple[Hotspot, ...]
    suggestions: Tuple[str, ...]
    duration_s: Optional[float]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "error": self.error,
            "max_von_mises_mpa": _round(self.max_von_mises_mpa),
            "safety_factor": _round(self.safety_factor),
            "max_displacement_mm": _round(self.max_displacement_mm),
            "hotspots": [
                {
                    "node": h.node_id,
                    "von_mises_mpa": _round(h.von_mises_mpa),
                    "location_mm": [_round(v) for v in h.location_mm],
                    "on_interface": h.on_interface,
                    "owning_feature": h.owning_feature,
                }
                for h in self.hotspots
            ],
            "suggestions": list(self.suggestions),
            "duration_s": _round(self.duration_s),
        }


@dataclass(frozen=True)
class VerificationReport:
    verdict: str
    environment: dict
    connections: Tuple[InterfaceCheckResult, ...]
    envelopes: Tuple[EnvelopeCheckResult, ...]
    cases: Tuple[CaseReport, ...]
    feedback: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "units": {"length": "mm", "force": "N", "stress": "MPa"},
            "verdict": self.verdict,
            "environment": self.environment,
            "connections": [c.to_dict() for c in self.connections],
            "envelopes": [e.to_dict() for e in self.envelopes],
            "load_cases": [c.to_dict() for c in self.cases],
            "feedback": list(self.feedback),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def to_markdown(self) -> str:
        lines: List[str] = []
        verdict_word = "PASS" if self.verdict == "pass" else "FAIL"
        lines.append(f"# Verification report — {verdict_word}")
        lines.append("")
        lines.append(f"Part: `{self.environment['part_package']}` · "
                     f"Material: {self.environment['material']['name']} "
                     f"(Re={self.environment['material']['yield_strength_mpa']} MPa) · "
                     f"Required SF ≥ {self.environment['safety_factor_required']}")
        lines.append("")
        lines.append("## Interfaces")
        for c in self.connections:
            mark = "ok" if c.passed else "FAIL"
            lines.append(f"- [{mark}] {c.interface_name} ({c.method.value}): "
                         f"{c.face_count} face(s), {c.total_area_mm2:.2f} mm², "
                         f"planarity ±{c.planarity_deviation_mm:.4f} mm")
            for message in c.messages:
                lines.append(f"  - {message}")
        lines.append("")
        if self.envelopes:
            lines.append("## Envelopes")
            for e in self.envelopes:
                mark = "ok" if e.passed else "FAIL"
                extra = ("" if e.passed
                         else f" — intersection {e.intersection_volume_mm3:.2f} mm³")
                lines.append(f"- [{mark}] keep-out {e.name}{extra}")
            lines.append("")
        lines.append("## Load cases")
        for case in self.cases:
            mark = "ok" if case.passed else "FAIL"
            if case.error:
                lines.append(f"- [{mark}] {case.name}: solver error — {case.error}")
                continue
            lines.append(
                f"- [{mark}] {case.name}: von Mises max "
                f"{case.max_von_mises_mpa:.2f} MPa, SF "
                f"{case.safety_factor:.2f} (≥ {self.environment['safety_factor_required']} "
                f"required), max displacement {case.max_displacement_mm:.4f} mm")
            for h in case.hotspots:
                where = h.on_interface or "free surface"
                feature = (f", feature {h.owning_feature['node_id']}"
                           if h.owning_feature else "")
                lines.append(f"  - hotspot {h.von_mises_mpa:.2f} MPa at "
                             f"({h.location_mm[0]:.1f}, {h.location_mm[1]:.1f}, "
                             f"{h.location_mm[2]:.1f}) on {where}{feature}")
            for suggestion in case.suggestions:
                lines.append(f"  - next: {suggestion}")
        lines.append("")
        if self.feedback:
            lines.append("## Feedback")
            for item in self.feedback:
                lines.append(f"- {item}")
        return "\n".join(lines) + "\n"


def build_report(
    *,
    env: VerificationEnv,
    mesh: Mesh,
    conn_results,
    envelope_results,
    outcomes,
    face_provenance,
) -> VerificationReport:
    constrained = [i.name for i in env.interfaces if i.method.constrains]
    loaded = [i.name for i in env.interfaces if not i.method.constrains]

    case_reports = []
    for outcome in outcomes:
        case_reports.append(_build_case(
            env, mesh, outcome, face_provenance, constrained, loaded,
        ))

    feedback: List[str] = []
    for envelope in envelope_results:
        if not envelope.passed:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = envelope.intersection_bbox_mm
            feedback.append(
                f"envelope violation: part intersects keep-out "
                f"'{envelope.name}' with {envelope.intersection_volume_mm3:.2f} mm³ "
                f"in the region x[{xmin:.1f}..{xmax:.1f}] "
                f"y[{ymin:.1f}..{ymax:.1f}] z[{zmin:.1f}..{zmax:.1f}] — "
                "remove material there or negotiate the keep-out"
            )
    for conn in conn_results:
        if not conn.passed:
            for message in conn.messages:
                feedback.append(
                    f"interface '{conn.interface_name}' ({conn.method.value}): {message}"
                )
    for case in case_reports:
        for suggestion in case.suggestions:
            feedback.append(f"load case '{case.name}': {suggestion}")

    all_passed = (
        all(c.passed for c in conn_results)
        and all(e.passed for e in envelope_results)
        and all(c.passed for c in case_reports)
    )

    return VerificationReport(
        verdict="pass" if all_passed else "fail",
        environment={
            "name": env.name,
            "part_package": env.part_package,
            "material": {
                "name": env.material.name,
                "youngs_modulus_mpa": env.material.youngs_modulus_mpa,
                "poisson_ratio": env.material.poisson_ratio,
                "yield_strength_mpa": env.material.yield_strength_mpa,
            },
            "safety_factor_required": env.safety_factor_required,
            "mesh_nodes": mesh.node_count,
            "mesh_tets": mesh.tet_count,
            "stress_source": "nodal extrapolation, linear tets (C3D4)",
        },
        connections=tuple(conn_results),
        envelopes=tuple(envelope_results),
        cases=tuple(case_reports),
        feedback=tuple(feedback),
    )


def _build_case(
    env: VerificationEnv,
    mesh: Mesh,
    outcome: CaseOutcome,
    face_provenance,
    constrained: List[str],
    loaded: List[str],
) -> CaseReport:
    if outcome.error is not None or outcome.frd is None:
        error = outcome.error or "no result file was produced"
        return CaseReport(
            name=outcome.name, passed=False, error=error,
            max_von_mises_mpa=None, safety_factor=None,
            max_displacement_mm=None, hotspots=(), suggestions=(
                f"case could not be solved ({error}) — repair the environment "
                "or mesh before judging the geometry",),
            duration_s=outcome.duration_s,
        )

    frd = outcome.frd
    try:
        ranked = sorted(
            ((node, _von_mises(frd, node)) for node in frd.block("STRESS").values),
            key=lambda item: (-item[1], item[0]),
        )
    except FrdParseError as exc:
        return CaseReport(
            name=outcome.name, passed=False, error=str(exc),
            max_von_mises_mpa=None, safety_factor=None,
            max_displacement_mm=None, hotspots=(), suggestions=(
                f"result file unusable: {exc}",),
            duration_s=outcome.duration_s,
        )

    max_node, max_vm = ranked[0]
    safety_factor = (env.material.yield_strength_mpa / max_vm) if max_vm > 0 else float("inf")
    max_disp = _max_displacement(frd)

    degenerate = (max_vm <= 0.0) or (max_disp is not None and max_disp <= 0.0)

    hotspots = []
    for node_id, vm in ranked[:_HOTSPOT_COUNT]:
        coords = frd.nodes.get(node_id)
        if coords is None:
            continue
        on_interface, feature = _attribute(
            mesh, node_id, face_provenance, constrained_first=constrained)
        hotspots.append(Hotspot(
            node_id=int(node_id), von_mises_mpa=vm,
            location_mm=tuple(float(v) for v in coords),
            on_interface=on_interface, owning_feature=feature,
        ))

    suggestions: List[str] = []
    passed = safety_factor >= env.safety_factor_required and not degenerate
    if degenerate:
        suggestions.append(
            "load case produced zero response (no displacement, no stress): "
            "the loads most likely act on fully constrained surfaces. Move "
            "the load target to a free face or point, or reconsider which "
            "interface is the support"
        )
    elif not passed:
        top = hotspots[0] if hotspots else None
        ratio = env.safety_factor_required / safety_factor if safety_factor > 0 else None
        if top is not None and top.on_interface in constrained:
            feature_note = (
                f" (face produced by feature {top.owning_feature['node_id']} "
                f"of graph {top.owning_feature['graph_id']})"
                if top.owning_feature else ""
            )
            area_note = (f"; with the same load the local bearing section needs "
                         f"roughly {ratio:.2f}x area to reach SF "
                         f"{env.safety_factor_required}" if ratio else "")
            suggestions.append(
                f"stress concentrates at the constrained interface "
                f"'{top.on_interface}'{feature_note} at "
                f"({top.location_mm[0]:.1f}, {top.location_mm[1]:.1f}, "
                f"{top.location_mm[2]:.1f}): add a fillet or gusset at the joint, "
                f"or raise the local section{area_note}"
            )
        else:
            suggestions.append(
                f"max von Mises {max_vm:.1f} MPa exceeds the allowable "
                f"{env.material.yield_strength_mpa / env.safety_factor_required:.1f} MPa "
                f"(SF {safety_factor:.2f} < {env.safety_factor_required}): increase "
                f"wall thickness or add ribs along the load path from "
                f"{', '.join(loaded) or 'the loaded point'} toward "
                f"{', '.join(constrained)}"
            )

    return CaseReport(
        name=outcome.name, passed=passed, error=None,
        max_von_mises_mpa=max_vm,
        safety_factor=safety_factor,
        max_displacement_mm=max_disp,
        hotspots=tuple(hotspots),
        suggestions=tuple(suggestions),
        duration_s=outcome.duration_s,
    )


def _attribute(mesh: Mesh, node_id: int, face_provenance, constrained_first):
    def order_key(name: str):
        # constrained interfaces first (that is where repair suggestions
        # bite), then alphabetical — deterministic either way
        return (0 if name in constrained_first else 1, name)

    for iface_name in sorted(mesh.interface_faces, key=order_key):
        for mesh_face in mesh.interface_faces[iface_name]:
            if node_id in mesh_face.nodes:
                return iface_name, face_provenance.get(mesh_face.topo_id)
    return None, None


def _von_mises(frd: FrdResult, node_id: int) -> float:
    return frd.von_mises_at(node_id)


def _max_displacement(frd: FrdResult) -> Optional[float]:
    try:
        block = frd.block("DISP")
    except FrdParseError:
        return None
    best = 0.0
    for node_id in block.values:
        best = max(best, frd.displacement_at(node_id).magnitude_mm)
    return best


def _round(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return value
    return round(float(value), 6)
