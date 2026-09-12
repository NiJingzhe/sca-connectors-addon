"""gmsh tetrahedral meshing with exact node-to-BREP-face association.

Pipeline per part:
1. export the materialized solid to BREP bytes and let gmsh's OCC kernel
   mesh it (linear tets, C3D4-compatible);
2. pull nodes/tets from gmsh;
3. for every interface face declared in the package's tag channel, classify
   boundary triangles by testing their centroid's OCC distance to the exact
   trimmed BREP face — association is geometric and exact, never inferred
   from entity ordering. Planar faces use the base tolerance; curved faces
   widen it by the chord sagitta h²/(8R) of the linear facets, sampled from
   the surface's maximum curvature, so seats and bores associate without
   admitting neighbouring-face triangles;
4. tributary areas (adjacent triangle areas / 3) become the load-distribution
   weights used by the deck generator.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import gmsh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_SurfaceType

from .brep_geom import (
    make_vertex,
    point_to_face_distance,
    write_brep_to_tempfile,
)
from .mesh_model import Mesh, MeshFace, Tet
from .meshquality import compute_mesh_quality

_TET_ELEMENT_TYPE = 4        # gmsh: 4-node tetrahedron
_TRIANGLE_ELEMENT_TYPE = 2   # gmsh: 3-node triangle
_INTERFACE_PREFIX = "interface."
_SAGITTA_MARGIN = 1.25       # safety factor over the ideal chord sagitta


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
        gmsh.option.setNumber("Mesh.Optimize", 1)  # explicit: quality pass on
        gmsh.model.mesh.generate(3)

        nodes = _read_nodes()
        tets = _read_tets()
        triangles = _read_boundary_triangles(nodes)
        sicn = _read_sicn(tets)
    finally:
        gmsh.clear()
        gmsh.finalize()
        try:
            Path(brep_path).unlink()
        except OSError:
            pass

    quality = compute_mesh_quality(nodes, tets, sicn=sicn)

    interface_faces: Dict[str, Tuple[MeshFace, ...]] = {}
    for full_name, info in loaded.interfaces.items():
        short_name = full_name[len(_INTERFACE_PREFIX):]
        faces = []
        for interface_face in info.faces:
            tolerance = _association_tolerance(
                interface_face.sdk_face, association_tol_mm, mesh_size_mm)
            face = _associate_face(
                interface_face, triangles, nodes,
                association_tol_mm=tolerance,
            )
            faces.append(face)
        interface_faces[short_name] = tuple(faces)

    return Mesh(nodes=nodes, tets=tets, interface_faces=interface_faces,
                quality=quality, target_size_mm=float(mesh_size_mm))


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


def _read_sicn(tets) -> Tuple[float, ...] | None:
    """gmsh minSICN per tet, aligned with ``tets`` order (cross-check metric)."""
    if not tets:
        return None
    tags = [tet.element_id for tet in tets]
    try:
        qualities = gmsh.model.mesh.getElementQualities(tags, "minSICN")
    except Exception:
        return None  # quality gate stays numpy-only; never fail the mesh on this
    return tuple(float(v) for v in qualities)


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

def _association_tolerance(sdk_face, base_tol_mm: float,
                           mesh_size_mm: float) -> float:
    """Base tolerance for planes; base + chord sagitta for curved faces.

    A linear facet of edge h sagging off a surface of minimum curvature
    radius R deviates by up to h²/(8R) at its centroid — with the previous
    flat-only tolerance a cylindrical seat associated zero triangles.
    """
    adaptor = BRepAdaptor_Surface(sdk_face.wrapped)
    if adaptor.GetType() == GeomAbs_SurfaceType.GeomAbs_Plane:
        return base_tol_mm
    radius = _min_curvature_radius_mm(sdk_face)
    if radius is None:
        return base_tol_mm
    sagitta = mesh_size_mm ** 2 / (8.0 * radius)
    return base_tol_mm + _SAGITTA_MARGIN * sagitta


def _min_curvature_radius_mm(sdk_face) -> float | None:
    """1/max|curvature| sampled over the face's UV range (None if flat)."""
    shape = sdk_face.wrapped
    adaptor = BRepAdaptor_Surface(shape)
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(shape)
    samples = 4
    worst_curvature = 0.0
    defined = False
    for i in range(samples):
        for j in range(samples):
            u = umin + (umax - umin) * i / (samples - 1)
            v = vmin + (vmax - vmin) * j / (samples - 1)
            props = BRepLProp_SLProps(adaptor, u, v, 2, 1e-7)
            if not props.IsCurvatureDefined():
                continue
            curvature = max(abs(props.MaxCurvature()),
                            abs(props.MinCurvature()))
            if curvature > 1e-9:
                defined = True
                worst_curvature = max(worst_curvature, curvature)
    if not defined:
        return None
    return 1.0 / worst_curvature


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
