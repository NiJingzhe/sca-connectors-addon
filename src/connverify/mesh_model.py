"""Pure mesh data model shared by meshing, deck generation, and reporting.

The mesh is linear tetrahedra (C3D4). Interface faces carry the BREP face
identity (``topo_id``) they lie on, the outward unit normal, their mesh nodes,
and each node's tributary area — the load-distribution weight that makes nodal
force lumping result-exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Tet:
    element_id: int
    node_ids: Tuple[int, int, int, int]


@dataclass(frozen=True)
class MeshFace:
    """The mesh region lying on one BREP face of one named interface."""

    topo_id: str
    normal: Tuple[float, float, float]      # outward unit normal
    nodes: Tuple[int, ...]
    tributary_area_mm2: Dict[int, float]    # node -> tributary area; sums to face area


@dataclass(frozen=True)
class Mesh:
    nodes: Dict[int, Tuple[float, float, float]]     # id -> (x, y, z) in mm
    tets: Tuple[Tet, ...]
    interface_faces: Dict[str, Tuple[MeshFace, ...]]  # short name (no prefix)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def tet_count(self) -> int:
        return len(self.tets)

    def interface_nodes(self, name: str) -> Tuple[int, ...]:
        """Unique, sorted node ids on all faces of an interface."""
        faces = self.interface_faces.get(name)
        if faces is None:
            raise KeyError(f"interface {name!r} is not present in the mesh")
        return tuple(sorted({n for f in faces for n in f.nodes}))

    def interface_area_mm2(self, name: str) -> float:
        faces = self.interface_faces.get(name)
        if faces is None:
            raise KeyError(f"interface {name!r} is not present in the mesh")
        return sum(w for f in faces for w in f.tributary_area_mm2.values())
