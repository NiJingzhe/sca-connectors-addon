"""gmsh tetrahedral meshing with exact node-to-BREP-face association.

Pipeline per part:
1. export the materialized solid to BREP bytes and let gmsh's OCC kernel
   mesh it (linear tets, C3D4-compatible);
2. pull nodes/tets from gmsh;
3. for every interface face declared in the package's tag channel, classify
   boundary triangles by testing their centroid's OCC distance to the exact
   trimmed BREP face — association is geometric and exact, never inferred
   from entity ordering;
4. tributary areas (adjacent triangle areas / 3) become the load-distribution
   weights used by the deck generator.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import gmsh

from .brep_geom import (
    make_vertex,
    point_to_face_distance,
    write_brep_to_tempfile,
)
from .mesh_model import Mesh, MeshFace, Tet

_TET_ELEMENT_TYPE = 4        # gmsh: 4-node tetrahedron
_TRIANGLE_ELEMENT_TYPE = 2   # gmsh: 3-node triangle
_INTERFACE_PREFIX = "interface."


def mesh_part(
    loaded,
    *,
    mesh_size_mm: float | None = None,
    association_tol_mm: float = 1e-3,
) -> Mesh:
    """Mesh a LoadedPart and associate interface faces exactly.

    ``mesh_size_mm=None`` picks an automatic target from the bounding box
    (diagonal / 12).
    """
    brep_path = write_brep_to_tempfile(loaded.body)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.open(str(brep_path))

        if mesh_size_mm is None:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            mesh_size_mm = math.dist(
                (xmin, ymin, zmin), (xmax, ymax, zmax)
            ) / 12.0 or 1.0
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_mm)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_mm * 0.25)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.model.mesh.generate(3)

        nodes = _read_nodes()
        tets = _read_tets()
        triangles = _read_boundary_triangles(nodes)
    finally:
        gmsh.clear()
        gmsh.finalize()
        try:
            Path(brep_path).unlink()
        except OSError:
            pass

    interface_faces: Dict[str, Tuple[MeshFace, ...]] = {}
    for full_name, info in loaded.interfaces.items():
        short_name = full_name[len(_INTERFACE_PREFIX):]
        faces = []
        for interface_face in info.faces:
            face = _associate_face(
                interface_face, triangles, nodes,
                association_tol_mm=association_tol_mm,
            )
            faces.append(face)
        interface_faces[short_name] = tuple(faces)

    return Mesh(nodes=nodes, tets=tets, interface_faces=interface_faces)


# ------------------------------------------------------------------ gmsh I/O

def _read_nodes() -> Dict[int, Tuple[float, float, float]]:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes: Dict[int, Tuple[float, float, float]] = {}
    for index, tag in enumerate(tags):
        base = 3 * index
        nodes[int(tag)] = (coords[base], coords[base + 1], coords[base + 2])
    return nodes


def _read_tets() -> Tuple[Tet, ...]:
    elem_tags, node_tags = gmsh.model.mesh.getElementsByType(_TET_ELEMENT_TYPE)
    tets = []
    for index, tag in enumerate(elem_tags):
        base = 4 * index
        tets.append(Tet(
            element_id=int(tag),
            node_ids=tuple(int(t) for t in node_tags[base:base + 4]),
        ))
    return tuple(tets)


def _read_boundary_triangles(nodes) -> List[Tuple[Tuple[int, int, int], Tuple[float, float, float], float]]:
    """All surface triangles as (node_ids, centroid, area)."""
    triangles = []
    for _dim, _tag in gmsh.model.getEntities(2):
        types, elem_tags, node_blocks = gmsh.model.mesh.getElements(2, _tag)
        for etype, elem_block, node_block in zip(types, elem_tags, node_blocks):
            if int(etype) != _TRIANGLE_ELEMENT_TYPE:
                continue
            for index in range(len(elem_block)):
                base = 3 * index
                ids = tuple(int(t) for t in node_block[base:base + 3])
                a = nodes[ids[0]]
                b = nodes[ids[1]]
                c = nodes[ids[2]]
                centroid = tuple((a[i] + b[i] + c[i]) / 3.0 for i in range(3))
                area = _triangle_area(a, b, c)
                triangles.append((ids, centroid, area))
    return triangles


def _triangle_area(a, b, c) -> float:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


# ------------------------------------------------------------- association

def _associate_face(interface_face, triangles, nodes, *, association_tol_mm) -> MeshFace:
    face_shape = interface_face.sdk_face.wrapped
    tributary: Dict[int, float] = {}
    member_nodes = set()
    member_triangles: list[Tuple[int, int, int]] = []
    for node_ids, centroid, area in triangles:
        vertex = make_vertex(*centroid)
        distance = point_to_face_distance(vertex, face_shape)
        if distance <= association_tol_mm:
            member_nodes.update(node_ids)
            member_triangles.append(node_ids)
            for node_id in node_ids:
                tributary[node_id] = tributary.get(node_id, 0.0) + area / 3.0
    normal = interface_face.sdk_face.get_normal_at().to_tuple()
    return MeshFace(
        topo_id=interface_face.topo_id,
        normal=tuple(float(v) for v in normal),
        nodes=tuple(sorted(member_nodes)),
        tributary_area_mm2=tributary,
        triangles=tuple(member_triangles),
    )
