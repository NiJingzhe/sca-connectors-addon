"""Mechanical connection-type registry: taxonomy, parameters, design rules.

Taxonomy (anchored to standard references, see each spec's ``standard``):

- **detachable** (可拆卸): threaded joints (through-bolt + nut, screw into
  tapped hole, stud), pins, keys, splines
- **semi-permanent**: interference / transition fits (ISO 286)
- **permanent** (不可拆): welding, riveting, adhesives
- load-entry faces (no fastener): plain contact pad, bearing seat (H7 bore)

Every spec knows its JointKind, the ConnectionMethod it idealizes to in the
linear-static model, and the quantitative layout rules used by the geometry
checks. Rule constants cite their standards; where a rule is a synthesis
across standards the docstring says so.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Type

from .env import ConnectionMethod


class JointKind(Enum):
    BOLTED_THROUGH = "bolted_through"    # 螺栓通孔 + 螺母
    BOLTED_TAPPED = "bolted_tapped"      # 螺钉/螺栓拧入螺纹孔
    STUD = "stud"                        # 双头螺柱
    PINNED = "pinned"                    # 圆柱/圆锥销 (ISO 2338 / 8734)
    KEYED = "keyed"                      # 平键 (GB/T 1095 / DIN 6885)
    SPLINED = "splined"                  # 花键 (GB/T 1144)
    INTERFERENCE = "interference"        # 过盈配合 (ISO 286: H7/p6 H7/r6 H7/s6 H7/u6)
    TRANSITION = "transition"            # 过渡配合 (ISO 286: H7/k6 H7/m6 H7/n6)
    WELDED_FILLET = "welded_fillet"      # 角焊缝 (AWS D1.1 / GB 985)
    WELDED_BUTT = "welded_butt"          # 对接焊缝
    RIVETED = "riveted"                  # 铆接
    ADHESIVE = "adhesive"                # 粘接
    CLAMPED = "clamped"                  # 刚性夹持 (台面/夹具压紧)
    BEARING_SEAT = "bearing_seat"        # 轴承座孔 (H7)
    CONTACT_PAD = "contact_pad"          # 纯接触传力面（载荷入口）
    SNAP_FIT = "snap_fit"                # 卡扣连接（卡 hook + 扣 recess）


PERMANENT_KINDS = frozenset({
    JointKind.WELDED_FILLET, JointKind.WELDED_BUTT,
    JointKind.RIVETED, JointKind.ADHESIVE,
})
SEMI_PERMANENT_KINDS = frozenset({JointKind.INTERFERENCE, JointKind.TRANSITION})

# Kinds whose mating face fully restrains the part in the linear-static model
# (clamped support). Key/spline/contact/bearing faces are load-entry surfaces.
CONSTRAINING_KINDS = frozenset({
    JointKind.BOLTED_THROUGH, JointKind.BOLTED_TAPPED, JointKind.STUD,
    JointKind.WELDED_FILLET, JointKind.WELDED_BUTT,
    JointKind.RIVETED, JointKind.ADHESIVE, JointKind.CLAMPED,
    JointKind.INTERFERENCE, JointKind.TRANSITION, JointKind.PINNED,
})

KIND_METHOD: Dict[JointKind, ConnectionMethod] = {
    JointKind.BOLTED_THROUGH: ConnectionMethod.BOLTED,
    JointKind.BOLTED_TAPPED: ConnectionMethod.BOLTED,
    JointKind.STUD: ConnectionMethod.BOLTED,
    JointKind.WELDED_FILLET: ConnectionMethod.WELDED,
    JointKind.WELDED_BUTT: ConnectionMethod.WELDED,
    JointKind.RIVETED: ConnectionMethod.WELDED,
    JointKind.ADHESIVE: ConnectionMethod.WELDED,
    JointKind.CLAMPED: ConnectionMethod.FIXED,
    JointKind.INTERFERENCE: ConnectionMethod.FIXED,
    JointKind.TRANSITION: ConnectionMethod.FIXED,
    JointKind.PINNED: ConnectionMethod.FIXED,
    JointKind.KEYED: ConnectionMethod.CONTACT,
    JointKind.SPLINED: ConnectionMethod.CONTACT,
    JointKind.BEARING_SEAT: ConnectionMethod.CONTACT,
    JointKind.CONTACT_PAD: ConnectionMethod.CONTACT,
    JointKind.SNAP_FIT: ConnectionMethod.CONTACT,
}


def kind_constrains(kind: JointKind) -> bool:
    return kind in CONSTRAINING_KINDS


# --------------------------------------------------------------------------
# Fastener data
# --------------------------------------------------------------------------

# ISO 4014 / 4017 hex-head bolts: nominal d -> (across flats s, head height k)
_HEX_HEAD_MM = {
    3.0: (5.5, 2.0), 4.0: (7.0, 2.8), 5.0: (8.0, 3.5), 6.0: (10.0, 4.0),
    8.0: (13.0, 5.3), 10.0: (16.0, 6.4), 12.0: (18.0, 7.5), 16.0: (24.0, 10.0),
    20.0: (30.0, 12.5), 24.0: (36.0, 15.0), 30.0: (46.0, 18.7),
    36.0: (55.0, 22.5), 42.0: (65.0, 26.0), 48.0: (75.0, 30.0),
}


def hex_head_dimensions(nominal_diameter_mm: float) -> Tuple[float, float]:
    d = float(nominal_diameter_mm)
    if d in _HEX_HEAD_MM:
        return _HEX_HEAD_MM[d]
    raise ValueError(
        f"no ISO 4014 hex-head entry for M{d:g}; supported: "
        f"M{min(_HEX_HEAD_MM):g}..M{max(_HEX_HEAD_MM):g}"
    )


# EN 1090-2 / ISO 273 normal clearance holes, synthesized:
# medium = d+1 for d<=16, d+2 above; coarse = medium +1. An explicit
# hole_diameter_mm on the spec always overrides.
def _clearance_hole_mm(d: float, fit: str) -> float:
    if fit == "medium":
        return d + (1.0 if d <= 16.0 else 2.0)
    if fit == "coarse":
        return d + (2.0 if d <= 16.0 else 3.0)
    raise ValueError(f"hole fit must be 'medium' or 'coarse', got {fit!r}")


# ISO 286 IT7 tolerance (µm) over nominal ranges up to (mm)
_IT7_UM = [
    (3.0, 10), (6.0, 12), (10.0, 15), (18.0, 18), (30.0, 21), (50.0, 25),
    (80.0, 30), (120.0, 35), (180.0, 40),
]


def it7_tolerance_mm(nominal_diameter_mm: float) -> Tuple[float, float]:
    """H7 hole limits (lower = nominal, upper = nominal + IT7)."""
    d = float(nominal_diameter_mm)
    if d <= 0:
        raise ValueError("diameter must be positive")
    if d > 180:
        raise ValueError("IT7 table embedded up to 180 mm only")
    for upper_bound, micron in _IT7_UM:
        if d <= upper_bound:
            return (d, d + micron / 1000.0)
    raise ValueError("unreachable")


# AWS D1.1 Table 7.7 minimum fillet size vs thinner plate thickness (mm)
def _aws_min_leg_mm(t: float) -> float:
    if t <= 6.0:
        return 3.0
    if t <= 12.0:
        return 5.0
    if t <= 20.0:
        return 6.0
    return 8.0


def min_fillet_leg_mm(thinner_plate_mm: float) -> float:
    """min(leg) = max(AWS D1.1 T7.7 floor, 1.5·√t rounded up to 0.5 mm)."""
    t = float(thinner_plate_mm)
    formula = 1.5 * math.sqrt(t)
    rounded = math.ceil(formula * 2.0) / 2.0
    return max(_aws_min_leg_mm(t), rounded)


_VALID_INTERFERENCE_FITS = frozenset({"H7/p6", "H7/r6", "H7/s6", "H7/u6"})
_VALID_TRANSITION_FITS = frozenset({"H7/k6", "H7/m6", "H7/n6"})


# --------------------------------------------------------------------------
# Joint specs
# --------------------------------------------------------------------------

class _SpecBase:
    """Non-dataclass base: kind/method/standard are behaviour, not fields."""

    standard: str = "BS 5950 / EN 1993-1-8 Table 3.3"

    @property
    def kind(self) -> JointKind:
        raise NotImplementedError

    @property
    def method(self) -> ConnectionMethod:
        return KIND_METHOD[self.kind]


@dataclass(frozen=True)
class BoltedThroughSpec(_SpecBase):
    nominal_diameter_mm: float = 10.0
    bolt_grade: str = "8.8"            # GB/T 3098.1
    hole_fit: str = "medium"           # EN 1090-2 / ISO 273
    hole_diameter_mm: Optional[float] = None
    expected_count: Optional[int] = None

    @property
    def kind(self) -> JointKind:
        return JointKind.BOLTED_THROUGH

    @property
    def resolved_hole_diameter_mm(self) -> float:
        return (self.hole_diameter_mm if self.hole_diameter_mm is not None
                else _clearance_hole_mm(self.nominal_diameter_mm, self.hole_fit))

    @property
    def min_pitch_mm(self) -> float:
        return 2.5 * self.nominal_diameter_mm

    @property
    def min_edge_mm(self) -> float:
        return 1.5 * self.nominal_diameter_mm

    @property
    def full_strength_edge_mm(self) -> float:
        return 2.0 * self.nominal_diameter_mm

    def tool_radius_mm(self) -> float:
        """Wrench envelope radius: across-corners/2 of the head + clearance."""
        s, _k = hex_head_dimensions(self.nominal_diameter_mm)
        return 0.75 * s

    def tool_height_mm(self) -> float:
        _s, k = hex_head_dimensions(self.nominal_diameter_mm)
        return k + 0.5 * self.nominal_diameter_mm + 5.0


@dataclass(frozen=True)
class BoltedTappedSpec(_SpecBase):
    nominal_diameter_mm: float = 8.0
    bolt_grade: str = "8.8"
    thread_depth_mm: Optional[float] = None   # >= 1.5d in steel for full load

    @property
    def kind(self) -> JointKind:
        return JointKind.BOLTED_TAPPED

    @property
    def min_thread_depth_mm(self) -> float:
        return 1.5 * self.nominal_diameter_mm

    @property
    def min_pitch_mm(self) -> float:
        return 2.5 * self.nominal_diameter_mm

    @property
    def min_edge_mm(self) -> float:
        return 1.5 * self.nominal_diameter_mm


@dataclass(frozen=True)
class StudSpec(BoltedTappedSpec):
    @property
    def kind(self) -> JointKind:
        return JointKind.STUD


@dataclass(frozen=True)
class PinnedSpec(_SpecBase):
    pin_diameter_mm: float = 6.0
    pin_type: str = "dowel"             # ISO 8734 (dowel) / ISO 2338 (unhardened)

    @property
    def kind(self) -> JointKind:
        return JointKind.PINNED

    @property
    def bore_diameter_limits_mm(self) -> Tuple[float, float]:
        # dowel pin bores: H7 (ISO 286); the pin itself is h6
        return it7_tolerance_mm(self.pin_diameter_mm)

    @property
    def min_pitch_mm(self) -> float:
        return 2.0 * self.pin_diameter_mm

    @property
    def min_edge_mm(self) -> float:
        return 1.5 * self.pin_diameter_mm

    @property
    def standard(self) -> str:
        return "ISO 8734 / ISO 286 H7"


@dataclass(frozen=True)
class KeyedSpec(_SpecBase):
    key_width_mm: float = 8.0
    key_height_mm: float = 7.0          # GB/T 1095 section b×h

    @property
    def kind(self) -> JointKind:
        return JointKind.KEYED

    @property
    def standard(self) -> str:
        return "GB/T 1095 / DIN 6885"


@dataclass(frozen=True)
class SplinedSpec(_SpecBase):
    module_mm: float = 2.0
    teeth: int = 8

    @property
    def kind(self) -> JointKind:
        return JointKind.SPLINED

    @property
    def standard(self) -> str:
        return "GB/T 1144"


@dataclass(frozen=True)
class InterferenceSpec(_SpecBase):
    nominal_diameter_mm: float = 20.0
    fit: str = "H7/r6"                  # ISO 286

    @property
    def kind(self) -> JointKind:
        return JointKind.INTERFERENCE

    def __post_init__(self):
        if self.fit not in _VALID_INTERFERENCE_FITS:
            raise ValueError(
                f"fit {self.fit!r} is not an interference class "
                f"(allowed: {sorted(_VALID_INTERFERENCE_FITS)})")

    @property
    def bore_diameter_limits_mm(self) -> Tuple[float, float]:
        return it7_tolerance_mm(self.nominal_diameter_mm)

    @property
    def standard(self) -> str:
        return "ISO 286"


@dataclass(frozen=True)
class TransitionSpec(InterferenceSpec):
    def __post_init__(self):
        if self.fit not in _VALID_TRANSITION_FITS:
            raise ValueError(
                f"fit {self.fit!r} is not a transition class "
                f"(allowed: {sorted(_VALID_TRANSITION_FITS)})")

    @property
    def kind(self) -> JointKind:
        return JointKind.TRANSITION


@dataclass(frozen=True)
class WeldedFilletSpec(_SpecBase):
    design_leg_mm: Optional[float] = None   # declared kf, checked vs min rule

    @property
    def kind(self) -> JointKind:
        return JointKind.WELDED_FILLET

    @property
    def standard(self) -> str:
        return "AWS D1.1 Table 7.7 / GB 985"

    def min_leg_mm(self, thinner_plate_mm: float) -> float:
        return min_fillet_leg_mm(thinner_plate_mm)


@dataclass(frozen=True)
class WeldedButtSpec(_SpecBase):
    groove: str = "single_v"    # GB 985: single_v / double_v / u / square

    @property
    def kind(self) -> JointKind:
        return JointKind.WELDED_BUTT

    @property
    def standard(self) -> str:
        return "GB 985 / ISO 9692"


@dataclass(frozen=True)
class RivetedSpec(_SpecBase):
    rivet_diameter_mm: float = 5.0
    head: str = "snap"          # GB 875 / DIN 660..663

    @property
    def kind(self) -> JointKind:
        return JointKind.RIVETED

    @property
    def min_pitch_mm(self) -> float:
        return 2.5 * self.rivet_diameter_mm   # riveted layout follows bolt rules

    @property
    def min_edge_mm(self) -> float:
        return 1.5 * self.rivet_diameter_mm


@dataclass(frozen=True)
class AdhesiveSpec(_SpecBase):
    bond_type: str = "epoxy"

    @property
    def kind(self) -> JointKind:
        return JointKind.ADHESIVE

    @property
    def standard(self) -> str:
        return "manufacturer bond-line spec"


@dataclass(frozen=True)
class ClampedSpec(_SpecBase):
    clamp_pressure_mpa: Optional[float] = None

    @property
    def kind(self) -> JointKind:
        return JointKind.CLAMPED

    @property
    def standard(self) -> str:
        return "fixture design practice"


@dataclass(frozen=True)
class BearingSeatSpec(_SpecBase):
    bore_diameter_mm: float = 52.0
    tolerance: str = "H7"       # typical outer-ring seat for normal loads

    @property
    def kind(self) -> JointKind:
        return JointKind.BEARING_SEAT

    @property
    def bore_diameter_limits_mm(self) -> Tuple[float, float]:
        return it7_tolerance_mm(self.bore_diameter_mm)

    @property
    def standard(self) -> str:
        return "ISO 286 (bearing seat practice)"


@dataclass(frozen=True)
class ContactPadSpec(_SpecBase):
    friction_coefficient: Optional[float] = None

    @property
    def kind(self) -> JointKind:
        return JointKind.CONTACT_PAD

    @property
    def standard(self) -> str:
        return "—"


@dataclass(frozen=True)
class SnapFitSpec(_SpecBase):
    """卡扣: a flexible hook on this part latching into the counterpart's
    recess. Hook deflection/retention force is v2; positions are v1."""

    catch_height_mm: float = 2.0

    @property
    def kind(self) -> JointKind:
        return JointKind.SNAP_FIT

    @property
    def standard(self) -> str:
        return "snap-fit design practice (deflection checks v2)"


_SPEC_TYPES: Dict[str, Type] = {
    "bolted_through": BoltedThroughSpec,
    "bolted_tapped": BoltedTappedSpec,
    "stud": StudSpec,
    "pinned": PinnedSpec,
    "keyed": KeyedSpec,
    "splined": SplinedSpec,
    "interference": InterferenceSpec,
    "transition": TransitionSpec,
    "welded_fillet": WeldedFilletSpec,
    "welded_butt": WeldedButtSpec,
    "riveted": RivetedSpec,
    "adhesive": AdhesiveSpec,
    "clamped": ClampedSpec,
    "bearing_seat": BearingSeatSpec,
    "contact_pad": ContactPadSpec,
    "snap_fit": SnapFitSpec,
}


def spec_to_dict(spec) -> dict:
    payload = {"kind": spec.kind.value}
    payload.update(asdict(spec))
    return payload


_BOLT_GRADES = frozenset({
    "4.6", "4.8", "5.6", "5.8", "6.8", "8.8", "9.8", "10.9", "12.9",
    "A2-70", "A2-80", "A4-70", "A4-80",  # stainless per GB/T 3098.6
})


def validate_spec(spec) -> list:
    """Collect-ALL validation for a joint spec; returns (field, message) pairs."""
    errors: list = []
    for attr in ("nominal_diameter_mm", "pin_diameter_mm", "rivet_diameter_mm",
                 "key_width_mm", "key_height_mm", "bore_diameter_mm",
                 "design_leg_mm", "thread_depth_mm"):
        value = getattr(spec, attr, None)
        if value is not None and not (isinstance(value, (int, float))
                                      and value > 0):
            errors.append((attr, f"must be > 0, got {value!r}"))
    grade = getattr(spec, "bolt_grade", None)
    if grade is not None and grade not in _BOLT_GRADES:
        errors.append((
            "bolt_grade",
            f"{grade!r} is not a GB/T 3098 property class "
            f"(allowed: {sorted(_BOLT_GRADES)})",
        ))
    fit = getattr(spec, "hole_fit", None)
    if fit is not None and fit not in ("medium", "coarse"):
        errors.append(("hole_fit", f"must be 'medium' or 'coarse', got {fit!r}"))
    count = getattr(spec, "expected_count", None)
    if count is not None and (not isinstance(count, int) or count < 1):
        errors.append(("expected_count", f"must be a positive integer, got {count!r}"))
    return errors


def spec_from_dict(payload: dict):
    kind = payload.get("kind")
    cls = _SPEC_TYPES.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown joint spec kind {kind!r}; allowed: {sorted(_SPEC_TYPES)}")
    kwargs = {k: v for k, v in payload.items() if k != "kind"}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"bad parameters for {kind}: {exc}") from exc
