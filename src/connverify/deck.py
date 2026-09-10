"""FEMaster deck generation (Abaqus-style, one deck per load case).

Verified against FEMaster v2.8.0 (bundled examples + importer source):

- ``*CLOAD`` values are PER NODE — the value is repeated for every node of
  the target set. The generator therefore pre-distributes interface loads by
  tributary-area share and emits exactly one merged row per loaded node, so
  the resultant force is exact.
- ``*SUPPORT`` rows: 0.0 fixes a DOF, NAN leaves it free. Solids carry
  translational DOF only, so rotational components stay NAN.
- ``*SOLIDSECTION`` is one word in this dialect.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

from .env import (
    ForceLoad,
    LoadCase,
    MomentLoad,
    PressureLoad,
    VerificationEnv,
)
from .mesh_model import Mesh


class DeckError(ValueError):
    """The deck cannot be generated from this mesh + environment."""


class UnsupportedLoadError(DeckError):
    """A declared load cannot be represented in the v1 solid-element pipeline."""


def _fmt(value: float) -> str:
    text = f"{float(value):.10g}"
    return text


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text).upper() or "UNNAMED"


def generate_decks(mesh: Mesh, env: VerificationEnv) -> Dict[str, str]:
    """One deck per load case, keyed by case name."""
    return {case.name: generate_deck(mesh, env, case) for case in env.load_cases}


def generate_deck(
    mesh: Mesh,
    env: VerificationEnv,
    case: LoadCase,
    *,
    point_attach_tol_mm: float = 2.0,
) -> str:
    _require_meshed_interfaces(mesh, env)

    model_name = f"{_sanitize_name(env.name)}__{_sanitize_name(case.name)}"
    lines: list[str] = []
    lines.append(f"*MODEL, NAME={model_name}")

    # --- nodes and elements -------------------------------------------------
    lines.append("*NODE, NSET=NALL")
    for node_id in sorted(mesh.nodes):
        x, y, z = mesh.nodes[node_id]
        lines.append(f"{node_id}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}")
    lines.append("*ELEMENT, TYPE=C3D4, ELSET=SOLID")
    for tet in sorted(mesh.tets, key=lambda t: t.element_id):
        n = tet.node_ids
        lines.append(f"{tet.element_id}, {n[0]}, {n[1]}, {n[2]}, {n[3]}")

    # --- interface node sets + supports --------------------------------------
    bc_name = f"BC_{_sanitize_name(case.name)}"
    for iface in env.interfaces:
        node_ids = mesh.interface_nodes(iface.name)
        _emit_nset(lines, f"IF_{_sanitize_name(iface.name)}", node_ids)

    support_rows: list[str] = []
    for iface in env.interfaces:
        if not iface.method.constrains:
            continue
        support_rows.append(
            f"IF_{_sanitize_name(iface.name)}, 0.0, 0.0, 0.0, NAN, NAN, NAN"
        )

    # --- loads: distribute, then merge one row per node ----------------------
    load_name = f"LOAD_{_sanitize_name(case.name)}"
    nodal_forces: Dict[int, Tuple[float, float, float]] = {}
    for load in case.loads:
        if isinstance(load, MomentLoad):
            raise UnsupportedLoadError(
                f"moment loads cannot act on solid-element faces in v1: "
                "linear tets have no rotational DOF and nodal lumping of a pure "
                "moment yields meaningless stress. Replace with an equivalent "
                "force couple: two ForceLoad(point_mm=..., ...) entries acting "
                "in opposite directions along a known arm."
            )
        elif isinstance(load, ForceLoad):
            if load.target is not None:
                _add_interface_force(nodal_forces, mesh, load)
            else:
                _add_point_force(nodal_forces, mesh, load,
                                 point_attach_tol_mm=point_attach_tol_mm)
        elif isinstance(load, PressureLoad):
            _add_pressure(nodal_forces, mesh, load)
        else:  # pragma: no cover - env validation upstream prevents this
            raise DeckError(f"unsupported load object {type(load).__name__}")

    # --- material + section ---------------------------------------------------
    lines.append(f"*MATERIAL, NAME=MAT_{_sanitize_name(env.material.name)}")
    lines.append("*ELASTIC, TYPE=ISOTROPIC")
    lines.append(f"{_fmt(env.material.youngs_modulus_mpa)}, "
                 f"{_fmt(env.material.poisson_ratio)}")
    lines.append("*DENSITY")
    lines.append(_fmt(env.material.density_t_per_mm3))
    lines.append("*SOLIDSECTION, ELSET=SOLID, "
                 f"MATERIAL=MAT_{_sanitize_name(env.material.name)}")

    # --- boundary conditions --------------------------------------------------
    if support_rows:
        lines.append(f"*SUPPORT, SUPPORT_COLLECTOR={bc_name}")
        lines.extend(support_rows)
    if nodal_forces:
        lines.append(f"*CLOAD, LOAD_COLLECTOR={load_name}")
        for node_id in sorted(nodal_forces):
            fx, fy, fz = nodal_forces[node_id]
            lines.append(f"{node_id}, {_fmt(fx)}, {_fmt(fy)}, {_fmt(fz)}, "
                         f"0.0, 0.0, 0.0")

    # --- analysis step ---------------------------------------------------------
    lines.append(f"*LOADCASE, TYPE=LINEARSTATIC, NAME={_sanitize_name(case.name)}")
    lines.append("*SUPPORTS")
    lines.append(bc_name)
    lines.append("*LOADS")
    lines.append(load_name)
    lines.append("*SOLVER, DEVICE=CPU, METHOD=DIRECT")
    lines.append("*CONSTRAINTMETHOD, TYPE=NULLSPACE")
    lines.append("*END")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ helpers

def _emit_nset(lines: list, name: str, node_ids) -> None:
    lines.append(f"*NSET, NAME={name}")
    ids = [str(i) for i in node_ids]
    for start in range(0, len(ids), 8):
        lines.append(", ".join(ids[start:start + 8]))


def _require_meshed_interfaces(mesh: Mesh, env: VerificationEnv) -> None:
    empty = [
        iface.name for iface in env.interfaces
        if not mesh.interface_faces.get(iface.name)
    ]
    if empty:
        raise DeckError(
            f"interfaces have no mesh nodes (face association produced nothing): "
            f"{', '.join(sorted(empty))}. The interface tags exist on the BREP but "
            "no mesh nodes were associated — check that the tagged faces are on "
            "the outer boundary of the solid."
        )


def _add_interface_force(
    nodal_forces: Dict[int, Tuple[float, float, float]], mesh: Mesh, load: ForceLoad
) -> None:
    faces = mesh.interface_faces.get(load.target)
    if faces is None:
        raise DeckError(
            f"force targets interface {load.target!r} which is not in the mesh"
        )
    total = sum(w for f in faces for w in f.tributary_area_mm2.values())
    if total <= 0.0:
        raise DeckError(f"interface {load.target!r} has zero tributary area")
    vec = np.array([load.fx_n, load.fy_n, load.fz_n], dtype=float)
    for face in faces:
        for node_id, weight in face.tributary_area_mm2.items():
            share = vec * (weight / total)
            _accumulate(nodal_forces, node_id, share)


def _add_point_force(
    nodal_forces: Dict[int, Tuple[float, float, float]], mesh: Mesh, load: ForceLoad,
    *, point_attach_tol_mm: float,
) -> None:
    target = np.asarray(load.point_mm, dtype=float)
    best_id, best_dist = None, float("inf")
    for node_id, coords in mesh.nodes.items():
        dist = float(np.linalg.norm(np.asarray(coords, dtype=float) - target))
        if dist < best_dist:
            best_id, best_dist = node_id, dist
    if best_dist > point_attach_tol_mm:
        raise DeckError(
            f"point load at ({', '.join(_fmt(v) for v in load.point_mm)}) is "
            f"{_fmt(best_dist)} mm from the nearest mesh node {best_id} "
            f"(tolerance {_fmt(point_attach_tol_mm)} mm). Give a point on the "
            "part surface, or apply the load through an interface target."
        )
    _accumulate(nodal_forces, best_id, np.array([load.fx_n, load.fy_n, load.fz_n]))


def _add_pressure(
    nodal_forces: Dict[int, Tuple[float, float, float]], mesh: Mesh, load: PressureLoad
) -> None:
    faces = mesh.interface_faces.get(load.interface)
    if faces is None:
        raise DeckError(
            f"pressure targets interface {load.interface!r} which is not in the mesh"
        )
    normal = np.asarray(faces[0].normal, dtype=float)
    for face in faces:
        n = np.asarray(face.normal, dtype=float)
        for node_id, weight in face.tributary_area_mm2.items():
            # positive pressure pushes onto the face: along -outward_normal
            _accumulate(nodal_forces, node_id, -load.magnitude_mpa * weight * n)


def _accumulate(
    nodal_forces: Dict[int, Tuple[float, float, float]],
    node_id: int,
    vec: "np.ndarray",
) -> None:
    current = nodal_forces.get(node_id, (0.0, 0.0, 0.0))
    nodal_forces[node_id] = (
        current[0] + float(vec[0]),
        current[1] + float(vec[1]),
        current[2] + float(vec[2]),
    )
