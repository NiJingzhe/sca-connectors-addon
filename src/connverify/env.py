"""Formal, deterministic verification-environment DSL for static connectors.

Three object classes:

- :class:`Interface` — a named connection end face (matching the part's
  ``interface.<name>`` tag) plus a connection method.
- :class:`LoadCase` — forces / moments / pressures applied to interfaces or points.
- Keep-out envelopes (:class:`KeepOutBox`) the part must not invade.

Units are explicit in every field name: mm, N, N·mm, MPa (the N-mm-MPa system
shared by ``.scadpkg`` geometry and FEMaster decks).

Design contract: constructors are dumb; :meth:`VerificationEnv.validate` is
the single validator and reports ALL violations at once so an agent can repair
the whole environment in one pass.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0"

_UNITS = {"length": "mm", "force": "N", "stress": "MPa"}
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")


class EnvValidationError(ValueError):
    """Raised when a verification environment is invalid.

    ``errors`` is a list of ``(field, message)`` tuples covering every
    violation found — never just the first.
    """

    def __init__(self, errors: Sequence[Tuple[str, str]]):
        self.errors = list(errors)
        rendered = "\n".join(f"  [{f}] {m}" for f, m in self.errors)
        super().__init__(
            f"verification environment invalid ({len(self.errors)} error(s)):\n{rendered}"
        )


class ConnectionMethod(Enum):
    """How a connector end face mates with its counterpart.

    FIXED / BOLTED / WELDED all idealize to a fully constrained support in
    linear-static v1 (bolts: full clamp; welds: tie). The distinction drives
    geometric sanity checks (e.g. planarity) and report language.
    CONTACT is a free mating surface: it never constrains; it is the legal
    entry point for externally applied loads.
    """

    FIXED = "fixed"
    BOLTED = "bolted"
    WELDED = "welded"
    CONTACT = "contact"

    @property
    def constrains(self) -> bool:
        return self in (ConnectionMethod.FIXED, ConnectionMethod.BOLTED,
                        ConnectionMethod.WELDED)


@dataclass(frozen=True)
class Material:
    name: str
    youngs_modulus_mpa: float
    poisson_ratio: float
    yield_strength_mpa: float
    density_t_per_mm3: float


@dataclass(frozen=True)
class Interface:
    """A named connection end face.

    Exactly one of ``method=`` (simple FEM boundary-condition label) or
    ``spec=`` (a full mechanical joint specification from
    :mod:`connverify.joint_types`) must be given; a spec implies its method.
    """

    name: str
    method: Optional[ConnectionMethod] = None
    spec: Optional[object] = None
    counterpart: Optional[object] = None
    planarity_tol_mm: float = 0.1
    min_area_mm2: Optional[float] = None

    def __post_init__(self):
        if self.spec is not None:
            if self.method is not None:
                raise TypeError(
                    "Interface accepts method= or spec=, not both — "
                    "a joint spec implies its connection method"
                )
            derived = getattr(self.spec, "method", None)
            if derived is None:
                raise TypeError("spec has no method; use a joint_types spec")
            object.__setattr__(self, "method", derived)
        elif self.method is None:
            raise TypeError("Interface needs method= or spec=")
        if not isinstance(self.method, ConnectionMethod):
            raise TypeError(
                f"Interface.method must be a ConnectionMethod, got {self.method!r}"
            )


@dataclass(frozen=True)
class ForceLoad:
    fx_n: float = 0.0
    fy_n: float = 0.0
    fz_n: float = 0.0
    target: Optional[str] = None
    point_mm: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.point_mm is not None:
            object.__setattr__(self, "point_mm", tuple(self.point_mm))


@dataclass(frozen=True)
class MomentLoad:
    mx_nmm: float = 0.0
    my_nmm: float = 0.0
    mz_nmm: float = 0.0
    target: Optional[str] = None
    point_mm: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.point_mm is not None:
            object.__setattr__(self, "point_mm", tuple(self.point_mm))


@dataclass(frozen=True)
class PressureLoad:
    """Uniform pressure on an interface's faces. Positive = pushing onto the face
    (acting along the inward surface normal)."""

    interface: str
    magnitude_mpa: float


@dataclass(frozen=True)
class LoadCase:
    name: str
    loads: Tuple[ForceLoad | MomentLoad | PressureLoad, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "loads", tuple(self.loads))


@dataclass(frozen=True)
class KeepOutBox:
    """Axis-aligned keep-out box. The part solid must not intersect it."""

    name: str
    min_corner_mm: Tuple[float, float, float]
    max_corner_mm: Tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "min_corner_mm", tuple(self.min_corner_mm))
        object.__setattr__(self, "max_corner_mm", tuple(self.max_corner_mm))


@dataclass(frozen=True)
class MeshStudy:
    """Optional h-refinement mesh-independence study.

    ``sizes_mm`` is canonicalized coarse -> fine (descending). Use a
    geometric family with a constant ratio (e.g. 12 / 8 / 5.333 at r = 1.5)
    — only then can observed order, the Richardson limit and GCI be
    computed. ``qoi_tolerance_pct`` is the acceptance threshold on the
    finest-pair relative change of the study's quantities of interest.
    """

    sizes_mm: Tuple[float, ...]
    qoi_tolerance_pct: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "sizes_mm",
                           tuple(sorted(self.sizes_mm, reverse=True)))

    def to_dict(self) -> dict:
        return {
            "sizes_mm": list(self.sizes_mm),
            "qoi_tolerance_pct": self.qoi_tolerance_pct,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MeshStudy":
        sizes = data.get("sizes_mm")
        if not isinstance(sizes, (list, tuple)) or not all(
                _finite(s) for s in sizes):
            raise EnvValidationError([
                ("mesh_study.sizes_mm",
                 "must be a list of finite mesh sizes in mm"),
            ])
        return cls(sizes_mm=tuple(float(s) for s in sizes),
                   qoi_tolerance_pct=data.get("qoi_tolerance_pct", 2.0))


@dataclass(frozen=True)
class VerificationEnv:
    name: str
    part_package: str
    material: Material
    interfaces: Tuple[Interface, ...]
    load_cases: Tuple[LoadCase, ...]
    envelopes: Tuple[KeepOutBox, ...] = ()
    safety_factor_required: float = 1.5
    mesh_study: Optional[MeshStudy] = None

    def __post_init__(self):
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "load_cases", tuple(self.load_cases))
        object.__setattr__(self, "envelopes", tuple(self.envelopes))

    # ------------------------------------------------------------------ query

    def interface(self, name: str) -> Interface:
        for iface in self.interfaces:
            if iface.name == name:
                return iface
        raise KeyError(name)

    def required_interface_names(self) -> frozenset:
        """The ``interface.*`` tag names this environment requires on the part."""
        return frozenset(f"interface.{i.name}" for i in self.interfaces)

    # -------------------------------------------------------------- validate

    def validate(self) -> None:
        errors: list[Tuple[str, str]] = []
        errors += self._validate_header()
        errors += self._validate_material()
        errors += self._validate_interfaces()
        errors += self._validate_loads()
        errors += self._validate_envelopes()
        errors += self._validate_mesh_study()
        if errors:
            raise EnvValidationError(errors)

    def _validate_header(self) -> list:
        errors = []
        if not isinstance(self.name, str) or not self.name.strip():
            errors.append(("name", "environment name must be a non-empty string"))
        if not isinstance(self.part_package, str) or not self.part_package.strip():
            errors.append(("part_package", "part_package must be a non-empty .scadpkg path"))
        elif not self.part_package.endswith(".scadpkg"):
            errors.append((
                "part_package",
                f"part_package must reference a .scadpkg product package, got {self.part_package!r}",
            ))
        if not (self.safety_factor_required > 0):
            errors.append((
                "safety_factor_required",
                f"must be > 0, got {self.safety_factor_required}",
            ))
        return errors

    def _validate_material(self) -> list:
        m = self.material
        errors = []
        if not isinstance(m, Material):
            return [("material", f"must be a Material, got {type(m).__name__}")]
        if not m.name.strip():
            errors.append(("material.name", "must be non-empty"))
        if not (m.youngs_modulus_mpa > 0):
            errors.append((
                "material.youngs_modulus_mpa",
                f"must be > 0 MPa, got {m.youngs_modulus_mpa}",
            ))
        if not (0.0 < m.poisson_ratio < 0.5):
            errors.append((
                "material.poisson_ratio",
                f"must lie in (0, 0.5), got {m.poisson_ratio}",
            ))
        if not (m.yield_strength_mpa > 0):
            errors.append((
                "material.yield_strength_mpa",
                f"must be > 0 MPa, got {m.yield_strength_mpa}",
            ))
        if not (m.density_t_per_mm3 > 0):
            errors.append((
                "material.density_t_per_mm3",
                f"must be > 0 t/mm³, got {m.density_t_per_mm3}",
            ))
        return errors

    def _validate_interfaces(self) -> list:
        from .joint_types import validate_spec

        errors = []
        seen = set()
        for idx, iface in enumerate(self.interfaces):
            where = f"interfaces[{idx}]"
            if not isinstance(iface, Interface):
                errors.append((where, f"must be an Interface, got {type(iface).__name__}"))
                continue
            if not _SLUG_RE.match(iface.name or ""):
                errors.append((
                    f"{where}.name",
                    f"{iface.name!r} is not a lowercase slug [a-z0-9_], max 64 chars",
                ))
            elif iface.name in seen:
                errors.append((
                    f"{where}.name",
                    f"duplicate interface name {iface.name!r}",
                ))
            seen.add(iface.name)
            if iface.planarity_tol_mm <= 0:
                errors.append((
                    f"{where}.planarity_tol_mm",
                    f"must be > 0 mm, got {iface.planarity_tol_mm}",
                ))
            if iface.min_area_mm2 is not None and iface.min_area_mm2 <= 0:
                errors.append((
                    f"{where}.min_area_mm2",
                    f"must be > 0 mm², got {iface.min_area_mm2}",
                ))
            if iface.spec is not None:
                for spec_field, message in validate_spec(iface.spec):
                    errors.append((f"{where}.spec.{spec_field}", message))
            if iface.counterpart is not None:
                from .counterpart import validate_counterpart
                for c_field, message in validate_counterpart(iface.counterpart):
                    errors.append((f"{where}.counterpart.{c_field}", message))
        if not any(
            isinstance(i, Interface) and i.method.constrains for i in self.interfaces
        ):
            errors.append((
                "interfaces",
                "no constraining interface: at least one FIXED/BOLTED/WELDED "
                "interface is required, otherwise the part is a free-floating body",
            ))
        return errors

    def _validate_loads(self) -> list:
        errors = []
        declared = {i.name for i in self.interfaces if isinstance(i, Interface)}
        if not self.load_cases:
            errors.append(("load_cases", "at least one load case is required"))
            return errors
        for ci, case in enumerate(self.load_cases):
            cwhere = f"load_cases[{ci}]"
            if not isinstance(case, LoadCase):
                errors.append((cwhere, f"must be a LoadCase, got {type(case).__name__}"))
                continue
            if not (isinstance(case.name, str) and case.name.strip()):
                errors.append((f"{cwhere}.name", "load case name must be non-empty"))
            if not case.loads:
                errors.append((
                    f"{cwhere}.loads",
                    f"load case {case.name!r} is empty: declare at least one load",
                ))
                continue
            for li, load in enumerate(case.loads):
                lwhere = f"{cwhere}.loads[{li}]"
                if isinstance(load, ForceLoad):
                    errors += self._check_vector_load(
                        lwhere, load, load.fx_n, load.fy_n, load.fz_n, declared, "N")
                elif isinstance(load, MomentLoad):
                    errors += self._check_vector_load(
                        lwhere, load, load.mx_nmm, load.my_nmm, load.mz_nmm, declared, "N·mm")
                elif isinstance(load, PressureLoad):
                    if load.interface not in declared:
                        errors.append((
                            f"{lwhere}.interface",
                            f"pressure targets undeclared interface {load.interface!r}",
                        ))
                    if load.magnitude_mpa == 0.0:
                        errors.append((
                            f"{lwhere}.magnitude_mpa",
                            "pressure magnitude must be nonzero",
                        ))
                else:
                    errors.append((
                        lwhere,
                        f"unsupported load type {type(load).__name__}",
                    ))
        return errors

    @staticmethod
    def _check_vector_load(where, load, vx, vy, vz, declared, unit) -> list:
        errors = []
        has_target = load.target is not None
        has_point = load.point_mm is not None
        if has_target and has_point:
            errors.append((
                f"{where}.target",
                "exactly one of target (interface name) or point_mm is allowed, got both",
            ))
        elif not has_target and not has_point:
            errors.append((
                f"{where}.target",
                "load needs a target interface name or a point_mm, got neither",
            ))
        elif has_target and load.target not in declared:
            errors.append((
                f"{where}.target",
                f"references undeclared interface {load.target!r}",
            ))
        if has_point:
            p = load.point_mm
            if len(p) != 3 or not all(_finite(v) for v in p):
                errors.append((
                    f"{where}.point_mm",
                    f"must be three finite coordinates, got {p!r}",
                ))
        if has_target and (has_point is False) and vx == 0.0 and vy == 0.0 and vz == 0.0:
            errors.append((
                where,
                f"all-zero {unit} vector carries no information; declare real magnitudes",
            ))
        return errors

    def _validate_envelopes(self) -> list:
        errors = []
        for ei, env in enumerate(self.envelopes):
            where = f"envelopes[{ei}]"
            if not isinstance(env, KeepOutBox):
                errors.append((where, f"must be a KeepOutBox, got {type(env).__name__}"))
                continue
            if not _SLUG_RE.match(env.name or ""):
                errors.append((f"{where}.name", f"{env.name!r} is not a lowercase slug"))
            lo, hi = env.min_corner_mm, env.max_corner_mm
            if len(lo) != 3 or len(hi) != 3:
                errors.append((where, "corners must have exactly three coordinates"))
                continue
            for axis, (a, b) in enumerate(zip(lo, hi)):
                if not (_finite(a) and _finite(b)):
                    errors.append((where, f"corner coordinates must be finite"))
                    break
                if a >= b:
                    errors.append((
                        where,
                        f"min_corner_mm[{axis}]={a} must be < max_corner_mm[{axis}]={b}",
                    ))
        return errors

    def _validate_mesh_study(self) -> list:
        if self.mesh_study is None:
            return []
        study = self.mesh_study
        errors: list[Tuple[str, str]] = []
        if not isinstance(study, MeshStudy):
            return [("mesh_study", f"must be a MeshStudy, got {type(study).__name__}")]
        sizes = study.sizes_mm
        if len(sizes) < 2:
            errors.append((
                "mesh_study.sizes_mm",
                f"a study needs at least two sizes (three for Richardson/"
                f"GCI), got {len(sizes)}",
            ))
        if any(not _finite(s) or s <= 0.0 for s in sizes):
            errors.append((
                "mesh_study.sizes_mm",
                f"all sizes must be finite and > 0 mm, got {sizes!r}",
            ))
        if len(set(sizes)) != len(sizes):
            errors.append((
                "mesh_study.sizes_mm",
                f"sizes must be distinct, got {sizes!r}",
            ))
        tol = study.qoi_tolerance_pct
        if not (_finite(tol) and 0.0 < tol <= 25.0):
            errors.append((
                "mesh_study.qoi_tolerance_pct",
                f"must lie in (0, 25] percent, got {tol!r}",
            ))
        return errors

    # ------------------------------------------------------------ serialize

    def to_json(self) -> str:
        """Deterministic JSON serialization. Refuses to serialize an invalid env."""
        self.validate()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "units": dict(_UNITS),
            "name": self.name,
            "part_package": self.part_package,
            "safety_factor_required": self.safety_factor_required,
            "material": {
                "name": self.material.name,
                "youngs_modulus_mpa": self.material.youngs_modulus_mpa,
                "poisson_ratio": self.material.poisson_ratio,
                "yield_strength_mpa": self.material.yield_strength_mpa,
                "density_t_per_mm3": self.material.density_t_per_mm3,
            },
            "interfaces": [_interface_to_dict(i) for i in self.interfaces],
            "load_cases": [_load_case_to_dict(c) for c in self.load_cases],
            "envelopes": [_envelope_to_dict(e) for e in self.envelopes],
        }
        if self.mesh_study is not None:
            payload["mesh_study"] = self.mesh_study.to_dict()
        return json.dumps(payload, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "VerificationEnv":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise EnvValidationError([("root", "payload must be a JSON object")])
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise EnvValidationError([
                ("schema_version",
                 f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION!r}"),
            ])
        env = cls(
            name=data["name"],
            part_package=data["part_package"],
            material=Material(
                name=data["material"]["name"],
                youngs_modulus_mpa=data["material"]["youngs_modulus_mpa"],
                poisson_ratio=data["material"]["poisson_ratio"],
                yield_strength_mpa=data["material"]["yield_strength_mpa"],
                density_t_per_mm3=data["material"]["density_t_per_mm3"],
            ),
            interfaces=tuple(_interface_from_dict(d) for d in data["interfaces"]),
            load_cases=tuple(_load_case_from_dict(d) for d in data["load_cases"]),
            envelopes=tuple(_envelope_from_dict(d) for d in data.get("envelopes", [])),
            safety_factor_required=data.get("safety_factor_required", 1.5),
            mesh_study=(MeshStudy.from_dict(data["mesh_study"])
                        if data.get("mesh_study") is not None else None),
        )
        env.validate()
        return env


# ----------------------------------------------------------------- helpers

def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _interface_to_dict(i: Interface) -> dict:
    payload = {
        "name": i.name,
        "method": i.method.value,
        "planarity_tol_mm": i.planarity_tol_mm,
        "min_area_mm2": i.min_area_mm2,
    }
    if i.spec is not None:
        from .joint_types import spec_to_dict
        payload["spec"] = spec_to_dict(i.spec)
    if i.counterpart is not None:
        from .counterpart import counterpart_to_dict
        payload["counterpart"] = counterpart_to_dict(i.counterpart)
    return payload


def _interface_from_dict(d: dict) -> Interface:
    spec = None
    if d.get("spec") is not None:
        from .joint_types import spec_from_dict
        try:
            spec = spec_from_dict(d["spec"])
        except ValueError as exc:
            raise EnvValidationError([("interfaces.spec", str(exc))]) from exc
    counterpart = None
    if d.get("counterpart") is not None:
        from .counterpart import counterpart_from_dict
        try:
            counterpart = counterpart_from_dict(d["counterpart"])
        except ValueError as exc:
            raise EnvValidationError([("interfaces.counterpart", str(exc))]) from exc
    kwargs = dict(
        name=d["name"],
        planarity_tol_mm=d.get("planarity_tol_mm", 0.1),
        min_area_mm2=d.get("min_area_mm2"),
        counterpart=counterpart,
    )
    if spec is not None:
        return Interface(name=kwargs["name"], spec=spec,
                         counterpart=counterpart,
                         planarity_tol_mm=kwargs["planarity_tol_mm"],
                         min_area_mm2=kwargs["min_area_mm2"])
    return Interface(name=kwargs["name"],
                     method=ConnectionMethod(d["method"]),
                     counterpart=counterpart,
                     planarity_tol_mm=kwargs["planarity_tol_mm"],
                     min_area_mm2=kwargs["min_area_mm2"])


def _load_case_to_dict(c: LoadCase) -> dict:
    loads = []
    for load in c.loads:
        if isinstance(load, ForceLoad):
            loads.append({
                "kind": "force",
                "fx_n": load.fx_n, "fy_n": load.fy_n, "fz_n": load.fz_n,
                "target": load.target, "point_mm": list(load.point_mm) if load.point_mm else None,
            })
        elif isinstance(load, MomentLoad):
            loads.append({
                "kind": "moment",
                "mx_nmm": load.mx_nmm, "my_nmm": load.my_nmm, "mz_nmm": load.mz_nmm,
                "target": load.target, "point_mm": list(load.point_mm) if load.point_mm else None,
            })
        elif isinstance(load, PressureLoad):
            loads.append({
                "kind": "pressure",
                "interface": load.interface,
                "magnitude_mpa": load.magnitude_mpa,
            })
    return {"name": c.name, "loads": loads}


def _load_case_from_dict(d: dict) -> LoadCase:
    loads = []
    for ld in d["loads"]:
        kind = ld["kind"]
        if kind == "force":
            loads.append(ForceLoad(
                fx_n=ld["fx_n"], fy_n=ld["fy_n"], fz_n=ld["fz_n"],
                target=ld.get("target"),
                point_mm=tuple(ld["point_mm"]) if ld.get("point_mm") else None))
        elif kind == "moment":
            loads.append(MomentLoad(
                mx_nmm=ld["mx_nmm"], my_nmm=ld["my_nmm"], mz_nmm=ld["mz_nmm"],
                target=ld.get("target"),
                point_mm=tuple(ld["point_mm"]) if ld.get("point_mm") else None))
        elif kind == "pressure":
            loads.append(PressureLoad(
                interface=ld["interface"], magnitude_mpa=ld["magnitude_mpa"]))
    return LoadCase(name=d["name"], loads=tuple(loads))


def _envelope_to_dict(e: KeepOutBox) -> dict:
    return {
        "kind": "box",
        "name": e.name,
        "min_corner_mm": list(e.min_corner_mm),
        "max_corner_mm": list(e.max_corner_mm),
    }


def _envelope_from_dict(d: dict) -> KeepOutBox:
    kind = d.get("kind", "box")
    if kind != "box":
        raise EnvValidationError([("envelopes", f"unknown envelope kind {kind!r}")])
    return KeepOutBox(
        name=d["name"],
        min_corner_mm=tuple(d["min_corner_mm"]),
        max_corner_mm=tuple(d["max_corner_mm"]),
    )
