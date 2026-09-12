"""Counterpart-based assemblability: the joint spec IS the mating face.

Declared per interface, the counterpart describes the mating part's NOMINAL
geometry in the interface face's local frame. connverify generates the
counterpart solid FROM the spec and verifies assemblability computationally
(boolean intersection, point classification, ISO 286 fit arithmetic) — a
real mating model is never built.

Every :class:`~connverify.joint_types.JointKind` maps to one counterpart
family (see ``COUNTERPART_FOR_KIND``):

- **plate family** (bolted through / tapped / stud / riveted):
  hole-pattern match, mating-plane interference (穿模), clamp land support
  (压紧), fastener stack length.
- **bore family** (pin / interference / transition / key / spline / bearing
  seat): nominal-diameter match, ISO 286 fit-band arithmetic, hub insertion
  path clearance.
- **plane family** (fillet/butt weld, adhesive, clamped, contact pad):
  mating-plane interference + contact coverage over the declared window.
- **snap family** (snap fit): hook presence/height at the declared catch
  position + slot clearance at the declared 扣 position.

Local frame contract (deterministic): origin = face bounding-box center,
n = outward normal at center, u = the global axis least aligned with n
projected onto the plane and normalized, v = n × u.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from OCP.Bnd import Bnd_Box
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.GProp import GProp_GProps
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopAbs import TopAbs_State
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from .joint_geometry import detect_holes, measure_local_thickness_mm
from .joint_types import (
    JointKind,
    it7_tolerance_mm,
)

_VOLUME_EPS_MM3 = 1e-6

# ISO 286 shaft classes (µm over nominal), embedded for 18 < d <= 30 mm —
# the size band used by the embedded demos; outside it, declare explicit
# numbers instead of guessing. (ei, es)
_SHAFT_CLASS_UM_18_30 = {
    "k6": (2, 15), "m6": (8, 21), "n6": (15, 28),
    "p6": (22, 35), "r6": (28, 41), "s6": (35, 48), "u6": (41, 62),
}


def fit_interference_range_mm(
    *, shaft_diameter_mm: float, shaft_class: str, bore_class: str = "H7"
) -> dict:
    """ISO 286 interference arithmetic for a shaft-in-hole pair.

    interference = shaft − bore over both tolerance bands: the range is
    [shaft_min − bore_max, shaft_max − bore_min]. Positive means the pair
    always interferes; a range straddling zero is a transition pair.
    """
    if bore_class != "H7":
        raise ValueError("bore side embedded for H7 only; declare numbers otherwise")
    band = _SHAFT_CLASS_UM_18_30.get(shaft_class)
    if band is None:
        raise ValueError(
            f"shaft class {shaft_class!r} embedded for 18-30 mm only "
            f"(allowed: {sorted(_SHAFT_CLASS_UM_18_30)})")
    if not 18.0 < shaft_diameter_mm <= 30.0:
        raise ValueError(
            "shaft tolerance table embedded for 18-30 mm; declare explicit "
            "limits for other sizes")
    ei, es = band
    bore_lo, bore_hi = it7_tolerance_mm(shaft_diameter_mm)
    lo = (shaft_diameter_mm + ei / 1000.0) - bore_hi
    hi = (shaft_diameter_mm + es / 1000.0) - bore_lo
    return {
        "interference_range_mm": (round(lo, 6), round(hi, 6)),
        "shaft_limits_mm": (round(shaft_diameter_mm + ei / 1000.0, 6),
                            round(shaft_diameter_mm + es / 1000.0, 6)),
        "bore_limits_mm": (round(bore_lo, 6), round(bore_hi, 6)),
    }


# --------------------------------------------------------------------------
# face-local frame
# --------------------------------------------------------------------------

def face_local_frame(sdk_face):
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(sdk_face.wrapped)
    origin = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)
    n = _unit(sdk_face.get_normal_at().to_tuple())
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    u_raw = min(axes, key=lambda a: abs(_dot(a, n)))
    u = _unit(_sub(u_raw, _scale(n, _dot(u_raw, n))))
    v = _cross(n, u)
    return origin, u, v, n


def project_point_to_face_local(sdk_face, point):
    origin, u, v, _n = face_local_frame(sdk_face)
    d = _sub(point, origin)
    return (_dot(d, u), _dot(d, v))


def local_to_global(sdk_frame, uvw):
    origin, u, v, n = sdk_frame
    return tuple(
        origin[i] + u[i] * uvw[0] + v[i] * uvw[1] + n[i] * uvw[2]
        for i in range(3)
    )


# --------------------------------------------------------------------------
# counterpart specs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PlateCounterpart:
    """Perforated mating plate: bolted / tapped / stud / riveted joints."""

    thickness_mm: float = 8.0
    holes: Tuple[Tuple[float, float, float], ...] = ()   # (u, v, d0) local mm
    window: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-50.0, -50.0), (50.0, 50.0))
    fastener_length_mm: Optional[float] = None
    position_tol_mm: float = 0.5

    def check_against(self, part, sdk_face, *, bolt=None, joint=None) -> dict:
        frame = face_local_frame(sdk_face)
        measured = detect_holes(sdk_face)
        measured_uv = [project_point_to_face_local(sdk_face, h.center_mm)
                       for h in measured]
        plate = _plate_solid(frame, self.window, self.thickness_mm, self.holes)

        result = {
            "counterpart": "plate",
            "pattern": _check_pattern(measured_uv, [h.diameter_mm for h in measured],
                                      self.holes, self.position_tol_mm),
            "interference": _check_interference(part, plate),
            "clamp": _check_clamp(part, sdk_face, measured, bolt),
            "stack": _check_stack(part, sdk_face, self, bolt),
        }
        return result


@dataclass(frozen=True)
class BoreCounterpart:
    """Mating bore/hub: pin, interference, transition, key, spline, bearing."""

    bore_diameter_mm: float = 20.0
    bore_class: str = "H7"
    bore_length_mm: float = 20.0
    hub_wall_mm: float = 3.0

    def check_against(self, part, sdk_face, *, joint=None, bolt=None) -> dict:
        adaptor = BRepAdaptor_Surface(sdk_face.wrapped)
        if adaptor.GetType() != GeomAbs_SurfaceType.GeomAbs_Cylinder:
            measured = None
            cylinder = None
            for hole in detect_holes(sdk_face):
                measured = hole.diameter_mm
                break
        else:
            cylinder = adaptor.Cylinder()
            measured = 2.0 * cylinder.Radius()
        result = {
            "counterpart": "bore",
            "fit": _check_fit(measured, self, joint),
        }
        if cylinder is not None:
            result["insertion"] = _check_hub_insertion(
                part, sdk_face, cylinder, self)
        else:
            result["insertion"] = {
                "passed": True,
                "messages": ["insertion path checked only for cylindrical seats"],
            }
        return result


@dataclass(frozen=True)
class PlaneCounterpart:
    """Flat mating surface: welds, adhesive, clamped pads, contact pads."""

    thickness_mm: float = 8.0
    window: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-50.0, -50.0), (50.0, 50.0))
    min_contact_ratio: float = 0.6

    def check_against(self, part, sdk_face, *, joint=None, bolt=None) -> dict:
        frame = face_local_frame(sdk_face)
        plate = _plate_solid(frame, self.window, self.thickness_mm, ())
        (umin, vmin), (umax, vmax) = self.window
        window_area = max(1e-9, (umax - umin) * (vmax - vmin))
        face_area = float(sdk_face.get_area())
        ratio = min(1.0, face_area / window_area)
        contact_ok = ratio >= self.min_contact_ratio
        return {
            "counterpart": "plane",
            "interference": _check_interference(part, plate),
            "contact": {
                "passed": contact_ok,
                "coverage_ratio": round(ratio, 4),
                "messages": ([] if contact_ok else [
                    f"contact coverage {ratio:.2f} of the declared window is "
                    f"below the required {self.min_contact_ratio}"
                ]),
            },
        }


@dataclass(frozen=True)
class SnapCounterpart:
    """Mating part with a catch recess (扣): the hook (卡) on our part must
    land on it. Declared in the face-local frame on the mating plane."""

    catch_uv: Tuple[float, float] = (0.0, 0.0)
    catch_height_mm: float = 2.0
    catch_window_uv: Tuple[float, float] = (6.0, 6.0)
    slot_uv: Tuple[float, float] = (20.0, 0.0)
    slot_width_uv: Tuple[float, float] = (8.0, 8.0)

    def check_against(self, part, sdk_face, *, joint=None, bolt=None) -> dict:
        frame = face_local_frame(sdk_face)
        hook = _check_material_above(part, frame, self.catch_uv,
                                     self.catch_window_uv)
        passed_hook = hook["max_height_mm"] >= self.catch_height_mm
        slot = _check_material_above(part, frame, self.slot_uv,
                                     self.slot_width_uv)
        passed_slot = slot["max_height_mm"] <= 1e-6
        return {
            "counterpart": "snap",
            "hook": {
                "passed": passed_hook,
                **hook,
                "messages": ([] if passed_hook else [
                    f"hook material above the mating plane at "
                    f"{self.catch_uv} is {hook['max_height_mm']:.2f} mm high, "
                    f"below the required {self.catch_height_mm:.2f} mm"
                ]),
            },
            "slot": {
                "passed": passed_slot,
                **slot,
                "messages": ([] if passed_slot else [
                    f"material occupies the counterpart slot region at "
                    f"{self.slot_uv} up to {slot['max_height_mm']:.2f} mm — "
                    "the catch recess must stay clear"
                ]),
            },
        }


COUNTERPART_FOR_KIND: Dict[JointKind, type] = {
    JointKind.BOLTED_THROUGH: PlateCounterpart,
    JointKind.BOLTED_TAPPED: PlateCounterpart,
    JointKind.STUD: PlateCounterpart,
    JointKind.RIVETED: PlateCounterpart,
    JointKind.PINNED: BoreCounterpart,
    JointKind.INTERFERENCE: BoreCounterpart,
    JointKind.TRANSITION: BoreCounterpart,
    JointKind.KEYED: BoreCounterpart,
    JointKind.SPLINED: BoreCounterpart,
    JointKind.BEARING_SEAT: BoreCounterpart,
    JointKind.WELDED_FILLET: PlaneCounterpart,
    JointKind.WELDED_BUTT: PlaneCounterpart,
    JointKind.ADHESIVE: PlaneCounterpart,
    JointKind.CLAMPED: PlaneCounterpart,
    JointKind.CONTACT_PAD: PlaneCounterpart,
    JointKind.SNAP_FIT: SnapCounterpart,
}


# --------------------------------------------------------------- serialize

_COUNTERPART_TYPES = {
    "plate": PlateCounterpart,
    "bore": BoreCounterpart,
    "plane": PlaneCounterpart,
    "snap": SnapCounterpart,
}


def counterpart_to_dict(counterpart) -> dict:
    from dataclasses import asdict

    payload = {}
    if isinstance(counterpart, PlateCounterpart):
        payload["kind"] = "plate"
    elif isinstance(counterpart, BoreCounterpart):
        payload["kind"] = "bore"
    elif isinstance(counterpart, PlaneCounterpart):
        payload["kind"] = "plane"
    elif isinstance(counterpart, SnapCounterpart):
        payload["kind"] = "snap"
    else:
        raise ValueError(f"unknown counterpart type {type(counterpart).__name__}")
    payload.update(asdict(counterpart))
    return payload


def counterpart_from_dict(payload: dict):
    kind = payload.get("kind")
    cls = _COUNTERPART_TYPES.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown counterpart kind {kind!r}; allowed: {sorted(_COUNTERPART_TYPES)}")
    kwargs = {k: v for k, v in payload.items() if k != "kind"}
    if isinstance(kwargs.get("holes"), list):
        kwargs["holes"] = tuple(tuple(h) for h in kwargs["holes"])
    if isinstance(kwargs.get("window"), list):
        kwargs["window"] = tuple(tuple(w) for w in kwargs["window"])
    for tuple_field in ("catch_uv", "slot_uv", "slot_width_uv", "catch_window_uv"):
        if isinstance(kwargs.get(tuple_field), list):
            kwargs[tuple_field] = tuple(kwargs[tuple_field])
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"bad parameters for counterpart {kind}: {exc}") from exc


def validate_counterpart(counterpart) -> list:
    errors = []
    thickness = getattr(counterpart, "thickness_mm", None)
    if thickness is not None and not thickness > 0:
        errors.append(("thickness_mm",
                       f"must be > 0, got {thickness}"))
    holes = getattr(counterpart, "holes", None) or ()
    for index, (u, v, d) in enumerate(holes):
        if not d > 0:
            errors.append((f"holes[{index}].d", f"diameter must be > 0, got {d}"))
    bore = getattr(counterpart, "bore_diameter_mm", None)
    if bore is not None and not bore > 0:
        errors.append(("bore_diameter_mm", f"must be > 0, got {bore}"))
    return errors


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def _check_pattern(measured_uv, measured_d, expected, tol) -> dict:
    messages = []
    holes = []
    if len(measured_uv) != len(expected):
        messages.append(
            f"counterpart expects {len(expected)} holes, part has "
            f"{len(measured_uv)}")
    used = set()
    for (eu, ev, ed) in expected:
        best_i, best_d = None, float("inf")
        for i, (mu, mv) in enumerate(measured_uv):
            if i in used:
                continue
            d = math.hypot(mu - eu, mv - ev)
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None:
            holes.append({"expected_uv": [eu, ev], "deviation_mm": None,
                          "diameter_mm": None})
            messages.append(f"no measured hole near ({eu:.1f}, {ev:.1f})")
            continue
        used.add(best_i)
        holes.append({
            "expected_uv": [round(eu, 4), round(ev, 4)],
            "deviation_mm": round(best_d, 4),
            "diameter_mm": round(measured_d[best_i], 4),
        })
        if best_d > tol:
            messages.append(
                f"hole near ({eu:.1f}, {ev:.1f}) is {best_d:.2f} mm off the "
                f"counterpart pattern (tolerance {tol:.2f} mm)")
    return {"passed": not messages, "holes": holes, "messages": messages,
            "position_tol_mm": tol}


def _check_interference(part, counterpart_solid) -> dict:
    common = BRepAlgoAPI_Common(part.body.wrapped, counterpart_solid)
    common.Build()
    if not common.IsDone():
        raise RuntimeError("OCC boolean failed during counterpart interference")
    shape = common.Shape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = float(props.Mass())
    if volume <= _VOLUME_EPS_MM3:
        return {"passed": True, "volume_mm3": 0.0, "bbox_mm": None,
                "messages": []}
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(shape)
    return {
        "passed": False,
        "volume_mm3": round(volume, 4),
        "bbox_mm": {"min": [xmin, ymin, zmin], "max": [xmax, ymax, zmax]},
        "messages": [
            f"counterpart interferes with the part: {volume:.1f} mm³ of "
            "material lies inside the mating body — remove the overlap or "
            "relocate the feature"
        ],
    }


def _check_clamp(part, sdk_face, measured_holes, bolt) -> dict:
    if bolt is None:
        return {"passed": True, "messages": [],
                "note": "clamp land checked when a bolt spec is present"}
    origin, u, v, n = face_local_frame(sdk_face)
    washer_d = 2.0 * bolt.nominal_diameter_mm          # ISO 7089-style
    r_in = bolt.resolved_hole_diameter_mm / 2.0 + 0.3
    r_out = 0.45 * washer_d
    messages = []
    holes_out = []
    for hole in measured_holes:
        missing = 0
        probes_total = 0
        for r in ((r_in + r_out) / 2.0, r_out * 0.95):
            for k in range(12):
                theta = 2 * math.pi * k / 12
                radial = (
                    u[0] * math.cos(theta) + v[0] * math.sin(theta),
                    u[1] * math.cos(theta) + v[1] * math.sin(theta),
                    u[2] * math.cos(theta) + v[2] * math.sin(theta),
                )
                probe = tuple(hole.center_mm[i] + radial[i] * r - n[i] * 0.05
                              for i in range(3))
                probes_total += 1
                if not _point_in_solid(part, probe):
                    missing += 1
        uv = project_point_to_face_local(sdk_face, hole.center_mm)
        ok = missing == 0
        holes_out.append({"uv": [round(uv[0], 3), round(uv[1], 3)],
                          "missing_samples": missing,
                          "probes": probes_total, "passed": ok})
        if not ok:
            messages.append(
                f"hole at ({uv[0]:.1f}, {uv[1]:.1f}): washer land overhangs "
                f"free material on {missing} of {probes_total} radial probes — "
                "the joint cannot develop clamp force there")
    return {"passed": not messages, "holes": holes_out, "messages": messages}


def _check_stack(part, sdk_face, counterpart, bolt) -> dict:
    if bolt is None or counterpart.fastener_length_mm is None:
        return {"passed": True, "messages": [],
                "note": "stack length checked when bolt + fastener length are declared"}
    thickness = measure_local_thickness_mm(part.body, sdk_face)
    if thickness is None:
        return {"passed": True, "messages": ["local thickness unmeasurable"]}
    grip = thickness + counterpart.thickness_mm
    nut = 0.8 * bolt.nominal_diameter_mm
    ok = counterpart.fastener_length_mm >= grip + nut - 1e-9
    return {
        "passed": ok,
        "grip_mm": round(grip, 4),
        "fastener_length_mm": counterpart.fastener_length_mm,
        "required_min_mm": round(grip + nut, 4),
        "messages": ([] if ok else [
            f"fastener {counterpart.fastener_length_mm:.1f} mm cannot span the "
            f"stack: grip {grip:.1f} mm + nut {nut:.1f} mm needs at least "
            f"{grip + nut:.1f} mm"
        ]),
    }


def _check_fit(measured_diameter, counterpart, joint) -> dict:
    messages = []
    shaft_class = None
    if joint is not None and hasattr(joint, "fit"):
        shaft_class = joint.fit.split("/")[-1] if "/" in joint.fit else None
    if measured_diameter is None:
        return {"passed": False, "messages": ["no cylindrical seat detected"]}
    nominal_match = abs(measured_diameter - counterpart.bore_diameter_mm) <= 0.05
    if not nominal_match:
        messages.append(
            f"measured seat ⌀{measured_diameter:.3f} mm does not match the "
            f"counterpart bore ⌀{counterpart.bore_diameter_mm:.3f} mm "
            "(nominal mismatch)")
    fit_info = None
    if shaft_class:
        try:
            fit_info = fit_interference_range_mm(
                shaft_diameter_mm=counterpart.bore_diameter_mm,
                shaft_class=shaft_class,
                bore_class=counterpart.bore_class)
        except ValueError as exc:
            messages.append(str(exc))
        else:
            lo, hi = fit_info["interference_range_mm"]
            kind = getattr(joint, "kind", None)
            if kind == JointKind.INTERFERENCE and lo <= 0:
                messages.append(
                    f"fit {joint.fit} yields clearance cases "
                    f"(min interference {lo:.3f} mm) — not an interference pair")
    return {
        "passed": not messages,
        "measured_diameter_mm": (round(measured_diameter, 4)
                                 if measured_diameter else None),
        "shaft_class": shaft_class,
        "interference_range_mm": (fit_info["interference_range_mm"]
                                  if fit_info else None),
        "messages": messages,
    }


def _check_hub_insertion(part, sdk_face, cylinder, counterpart) -> dict:
    axis_dir = _unit((cylinder.Axis().Direction().X(),
                      cylinder.Axis().Direction().Y(),
                      cylinder.Axis().Direction().Z()))
    location = cylinder.Axis().Location()
    # outward = direction pointing away from our material along the axis
    n = _unit(sdk_face.get_normal_at().to_tuple())
    approach = _scale(axis_dir, 1.0 if _dot(axis_dir, n) >= 0 else -1.0)
    xmin, ymin, zmin, xmax, ymax, zmax = _bbox(sdk_face.wrapped)
    corners = [(xmin, ymin, zmin), (xmax, ymax, zmax)]
    far = max(_dot(_sub(c, (location.X(), location.Y(), location.Z())), approach)
              for c in corners)
    r_in = counterpart.bore_diameter_mm / 2.0
    r_out = r_in + counterpart.hub_wall_mm
    base = gp_Pnt(location.X() + approach[0] * (far - counterpart.bore_length_mm),
                  location.Y() + approach[1] * (far - counterpart.bore_length_mm),
                  location.Z() + approach[2] * (far - counterpart.bore_length_mm))
    axis = gp_Ax2(base, gp_Dir(*approach))
    outer = BRepPrimAPI_MakeCylinder(axis, r_out,
                                     counterpart.bore_length_mm).Solid()
    inner = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(base.X() - approach[0] * 1.0,
                      base.Y() - approach[1] * 1.0,
                      base.Z() - approach[2] * 1.0),
               gp_Dir(*approach)),
        r_in, counterpart.bore_length_mm + 2.0).Solid()
    ring = BRepAlgoAPI_Cut(outer, inner)
    ring.Build()
    common = BRepAlgoAPI_Common(part.body.wrapped, ring.Shape())
    common.Build()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(common.Shape(), props)
    volume = float(props.Mass())
    ok = volume <= _VOLUME_EPS_MM3
    return {
        "passed": ok,
        "obstruction_mm3": round(volume, 4),
        "messages": ([] if ok else [
            f"hub travel path obstructed by {volume:.1f} mm³ of material — "
            "the mating hub cannot slide onto this seat"
        ]),
    }


def _check_material_above(part, frame, center_uv, window_uv) -> dict:
    origin, u, v, n = frame
    max_height = 0.0
    for i in range(5):
        for j in range(5):
            uu = center_uv[0] + (window_uv[0] / 2) * (i / 2 - 1)
            vv = center_uv[1] + (window_uv[1] / 2) * (j / 2 - 1)
            for height in (8.0, 4.0, 2.0, 1.0, 0.5):   # tallest first
                probe = local_to_global(frame, (uu, vv, height))
                if _point_in_solid(part, probe):
                    max_height = max(max_height, height)
                    break
    return {"max_height_mm": max_height}


# ------------------------------------------------------------------ helpers

def _plate_solid(frame, window, thickness, holes):
    (umin, vmin), (umax, vmax) = window
    origin, u, v, n = frame
    plate = BRepPrimAPI_MakeBox(
        gp_Ax2(gp_Pnt(*local_to_global(frame, (umin, vmin, 0.0))),
               gp_Dir(*n), gp_Dir(*u)),
        umax - umin, vmax - vmin, thickness).Solid()
    for (hu, hv, hd) in holes:
        center = local_to_global(frame, (hu, hv, -thickness * 0.25))
        tool = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(*center), gp_Dir(*n)),
            hd / 2.0, thickness * 1.5).Solid()
        cut = BRepAlgoAPI_Cut(plate, tool)
        cut.Build()
        plate = cut.Shape()
    return plate


def _point_in_solid(part, point) -> bool:
    classifier = BRepClass3d_SolidClassifier(part.body.wrapped)
    classifier.Perform(gp_Pnt(*point), 1e-7)
    return classifier.State() == TopAbs_State.TopAbs_IN


def _bbox(shape):
    box = Bnd_Box()
    box.SetGap(0.0)
    BRepBndLib.Add_s(shape, box, False)
    return box.Get()


def _unit(a):
    length = math.sqrt(sum(x * x for x in a)) or 1.0
    return tuple(x / length for x in a)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)
