"""Connection-interface verification, dispatched by mechanical joint type.

Two layers:
1. generic face facts (planarity, area, normal, centroid) — every interface;
2. joint-type rules (hole layout vs EN 1993 / GB 50017 pitch and edge
   limits, clearance-hole diameter, through/blind requirement, H7 bore
   limits per ISO 286, fillet-leg minimum per AWS D1.1/GB 985, wrench and
   nut accessibility) — run only when the interface carries a joint spec.

Every violation message cites its numbers and its standard so the agent can
repair the geometry at the named place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_SurfaceType

from .env import ConnectionMethod, Interface
from .installability import check_tool_access
from .joint_geometry import (
    detect_holes,
    hole_axis_intersections,
    measure_local_thickness_mm,
    min_edge_distance_mm,
    pairwise_pitch_mm,
)
from .joint_types import (
    JointKind,
    WeldedFilletSpec,
    min_fillet_leg_mm,
)
from .package_reader import LoadedPart

_DIAMETER_BAND_MM = 0.75   # measured clearance hole vs design d0
_BORE_BAND_MM = 0.02       # measured bore vs ISO 286 limits


@dataclass(frozen=True)
class InterfaceCheckResult:
    interface_name: str
    method: ConnectionMethod
    passed: bool
    face_count: int
    total_area_mm2: float
    planarity_deviation_mm: float
    normal: Optional[Tuple[float, float, float]]
    centroid_mm: Optional[Tuple[float, float, float]]
    min_area_required_mm2: Optional[float]
    messages: Tuple[str, ...] = field(default_factory=tuple)
    joint: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "interface": self.interface_name,
            "method": self.method.value,
            "passed": self.passed,
            "face_count": self.face_count,
            "total_area_mm2": round(self.total_area_mm2, 6),
            "planarity_deviation_mm": round(self.planarity_deviation_mm, 9),
            "normal": list(self.normal) if self.normal else None,
            "centroid_mm": list(self.centroid_mm) if self.centroid_mm else None,
            "min_area_required_mm2": self.min_area_required_mm2,
            "messages": list(self.messages),
            "joint_check": self.joint,
        }


def check_interface(loaded: LoadedPart, iface: Interface) -> InterfaceCheckResult:
    info = loaded.interfaces[f"interface.{iface.name}"]
    messages: List[str] = []

    deviations = [_planarity_deviation(f.sdk_face) for f in info.faces]
    planarity = max(deviations) if deviations else float("inf")
    needs_planar_mating = iface.method in (
        ConnectionMethod.BOLTED, ConnectionMethod.WELDED, ConnectionMethod.FIXED,
    )
    planarity_ok = True
    if needs_planar_mating and planarity > iface.planarity_tol_mm:
        planarity_ok = False
        messages.append(
            f"mating face not planar: deviation "
            f"{planarity:.4f} mm exceeds tolerance {iface.planarity_tol_mm:.4f} mm"
        )

    total_area = float(sum(f.sdk_face.get_area() for f in info.faces))
    area_ok = True
    if iface.min_area_mm2 is not None and total_area < iface.min_area_mm2:
        area_ok = False
        messages.append(
            f"bearing area {total_area:.2f} mm² below required "
            f"{iface.min_area_mm2:.2f} mm²"
        )

    joint_report = None
    joint_ok = True
    if iface.spec is not None:
        joint_report = check_joint(loaded, iface)
        messages.extend(joint_report["violations"])
        joint_ok = joint_report["passed"]

    normal = centroid = None
    if len(info.faces) == 1:
        normal = tuple(float(v) for v in info.faces[0].sdk_face.get_normal_at().to_tuple())
        c = info.faces[0].sdk_face.get_center()
        centroid = (float(c.x), float(c.y), float(c.z))

    passed = planarity_ok and area_ok and joint_ok
    return InterfaceCheckResult(
        interface_name=iface.name,
        method=iface.method,
        passed=passed,
        face_count=len(info.faces),
        total_area_mm2=total_area,
        planarity_deviation_mm=planarity,
        normal=normal,
        centroid_mm=centroid,
        min_area_required_mm2=iface.min_area_mm2,
        messages=tuple(messages),
        joint=joint_report,
    )


# --------------------------------------------------------------------------
# Joint-type dispatch
# --------------------------------------------------------------------------

def check_joint(loaded: LoadedPart, iface: Interface) -> dict:
    spec = iface.spec
    info = loaded.interfaces[f"interface.{iface.name}"]
    kind = spec.kind

    report = {
        "kind": kind.value,
        "standard": spec.standard,
        "holes": [],
        "notes": [],
        "violations": [],
        "passed": True,
    }

    if kind in (JointKind.WELDED_FILLET, JointKind.WELDED_BUTT):
        _check_weld(loaded, info, spec, report)
        report["passed"] = not report["violations"]
        return report

    if kind in (JointKind.PINNED, JointKind.INTERFERENCE,
                JointKind.TRANSITION, JointKind.BEARING_SEAT):
        _check_bore(loaded, info, spec, report)
        report["passed"] = not report["violations"]
        return report

    if kind in (JointKind.KEYED, JointKind.SPLINED):
        report["notes"].append(
            "keyway/spline profile geometry checks arrive in v2; "
            "the interface is treated as a load-entry face"
        )
        return report

    if kind in (JointKind.CLAMPED, JointKind.CONTACT_PAD, JointKind.ADHESIVE):
        if kind == JointKind.CONTACT_PAD:
            report["notes"].append("contact pad: no fastener geometry to check")
        return report

    # fastener families: bolted through / tapped / stud / riveted
    _check_fastener_layout(loaded, info, spec, report)
    report["passed"] = not report["violations"]
    return report


def _check_fastener_layout(loaded, info, spec, report) -> None:
    kind = spec.kind
    standard = spec.standard
    holes_total = 0

    for face_info in info.faces:
        face = face_info.sdk_face
        holes = detect_holes(face)
        holes_total += len(holes)

        pitch_map = pairwise_pitch_mm(holes)
        for (i, j), pitch in sorted(pitch_map.items()):
            limit = spec.min_pitch_mm
            if pitch < limit - 1e-9:
                report["violations"].append(
                    f"holes at {_xy(holes[i])} and {_xy(holes[j])}: pitch "
                    f"{pitch:.1f} mm < {limit:.1f} mm (2.5d, {standard})"
                )

        for hole in holes:
            entry = {
                "center_mm": [round(v, 6) for v in hole.center_mm],
                "diameter_mm": round(hole.diameter_mm, 6),
                "edge_mm": round(min_edge_distance_mm(face, hole), 6),
                "through": hole_axis_intersections(loaded.body, hole, face) == 0,
                "tool_access": None,
            }
            edge_limit = spec.min_edge_mm
            if entry["edge_mm"] < edge_limit - 1e-9:
                report["violations"].append(
                    f"hole at {_xy(hole)}: edge distance {entry['edge_mm']:.1f} mm "
                    f"< {edge_limit:.1f} mm (1.5d, {standard})"
                )
            d0 = getattr(spec, "resolved_hole_diameter_mm", None)
            if d0 is not None and abs(hole.diameter_mm - d0) > _DIAMETER_BAND_MM:
                report["violations"].append(
                    f"hole at {_xy(hole)}: diameter {hole.diameter_mm:.2f} mm "
                    f"deviates from design d0 = {d0:.2f} mm by more than "
                    f"±{_DIAMETER_BAND_MM} mm"
                )
            through_required = kind in (JointKind.BOLTED_THROUGH,
                                        JointKind.STUD, JointKind.RIVETED)
            if through_required and not entry["through"]:
                report["violations"].append(
                    f"hole at {_xy(hole)}: not a through hole — a shank path is "
                    "required for bolt-and-nut / riveted joints"
                )
            if kind == JointKind.BOLTED_TAPPED and entry["through"]:
                report["violations"].append(
                    f"hole at {_xy(hole)}: through hole — a tapped joint expects "
                    "a blind thread hole in this part"
                )
            if hasattr(spec, "tool_radius_mm"):
                access = check_tool_access(loaded, face, hole, spec)
                entry["tool_access"] = access.to_dict()
                if not access.passed:
                    report["violations"].extend(access.messages)
            report["holes"].append(entry)

    expected = getattr(spec, "expected_count", None)
    if expected is not None and expected != holes_total:
        report["violations"].append(
            f"expected {expected} fastener holes on the interface, "
            f"found {holes_total}"
        )
    if holes_total == 0:
        report["violations"].append(
            f"no fastener holes found on interface {info.name!r} — the joint "
            f"type {kind.value} requires round holes in the mating face"
        )


def _check_bore(loaded, info, spec, report) -> None:
    lo, hi = spec.bore_diameter_limits_mm
    nominal = getattr(spec, "nominal_diameter_mm", None) \
        or getattr(spec, "pin_diameter_mm", None) \
        or getattr(spec, "bore_diameter_mm", None)

    for face_info in info.faces:
        face = face_info.sdk_face
        adaptor = BRepAdaptor_Surface(face.wrapped)
        if adaptor.GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinder = adaptor.Cylinder()
            measured = 2.0 * cylinder.Radius()
            _bore_verdict(measured, lo, hi, nominal, report,
                          origin=None)
            continue
        for hole in detect_holes(face):
            _bore_verdict(hole.diameter_mm, lo, hi, nominal, report,
                          origin=hole.center_mm)


def _bore_verdict(measured, lo, hi, nominal, report, origin) -> None:
    entry = {
        "bore_diameter_mm": round(measured, 6),
        "h7_limits_mm": [round(lo, 4), round(hi, 4)],
    }
    if origin is not None:
        entry["center_mm"] = [round(v, 6) for v in origin]
    if not (lo - _BORE_BAND_MM <= measured <= hi + _BORE_BAND_MM):
        report["violations"].append(
            f"bore ⌀{measured:.3f} mm outside H7 limits "
            f"[{lo:.3f}, {hi:.3f}] mm for ⌀{nominal:g} "
            f"(ISO 286; spec: {report['standard']})"
        )
    report["holes"].append(entry)


def _check_weld(loaded, info, spec, report) -> None:
    if spec.kind != JointKind.WELDED_FILLET:
        report["notes"].append(
            "butt weld: groove geometry checks arrive in v2; planarity is checked"
        )
        return
    for face_info in info.faces:
        thickness = measure_local_thickness_mm(loaded.body, face_info.sdk_face)
        if thickness is None:
            report["notes"].append(
                "local plate thickness could not be measured at the face center; "
                "fillet-leg minimum not evaluated"
            )
            continue
        minimum = spec.min_leg_mm(thickness)
        if spec.design_leg_mm is None:
            report["notes"].append(
                f"local plate thickness {thickness:.1f} mm -> minimum fillet leg "
                f"{minimum:.1f} mm (AWS D1.1 Table 7.7 / GB 985); declare "
                "design_leg_mm to enforce"
            )
        elif spec.design_leg_mm < minimum - 1e-9:
            report["violations"].append(
                f"design weld leg {spec.design_leg_mm:.1f} mm < minimum "
                f"{minimum:.1f} mm for local plate thickness {thickness:.1f} mm "
                f"({spec.standard})"
            )
        else:
            report["notes"].append(
                f"weld leg {spec.design_leg_mm:.1f} mm >= minimum {minimum:.1f} mm "
                f"for local plate thickness {thickness:.1f} mm ({spec.standard})"
            )


def _xy(hole) -> str:
    return (f"({hole.center_mm[0]:.1f}, {hole.center_mm[1]:.1f}, "
            f"{hole.center_mm[2]:.1f})")


# --------------------------------------------------------------------------
# planarity
# --------------------------------------------------------------------------

def _planarity_deviation(sdk_face) -> float:
    """0.0 for exact planes; max deviation from a least-squares plane else."""
    shape = sdk_face.wrapped
    adaptor = BRepAdaptor_Surface(shape)
    if adaptor.GetType() == GeomAbs_SurfaceType.GeomAbs_Plane:
        return 0.0

    umin, umax, vmin, vmax = BRepTools.UVBounds_s(shape)
    samples = []
    for i in range(5):
        for j in range(5):
            u = umin + (umax - umin) * i / 4.0
            v = vmin + (vmax - vmin) * j / 4.0
            p = adaptor.Value(u, v)
            samples.append((p.X(), p.Y(), p.Z()))
    pts = np.asarray(samples)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal_vec = vh[0]
    deviations = np.abs((pts - centroid) @ normal_vec)
    return float(deviations.max())
