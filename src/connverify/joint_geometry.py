"""Joint-face geometry: exact hole layout facts from the BREP.

- Holes are the face's **inner wires** that are a single closed circular
  edge — exact centers and diameters from the circle geometry, no fitting.
- Through/blind is decided by shooting the hole axis through the solid: a
  clear axis means the fastener passes; any hit means blind material.
- Local plate thickness at the face center = first material hit along the
  inward normal (drives the minimum-fillet-leg rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepTools import BRepTools
from OCP.GeomAbs import GeomAbs_CurveType
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

from .brep_geom import make_vertex

_BIG = 1e6


@dataclass(frozen=True)
class HoleInfo:
    center_mm: Tuple[float, float, float]
    diameter_mm: float
    wire_topo: object  # OCP wire handle for downstream checks

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0


def detect_holes(sdk_face) -> Tuple[HoleInfo, ...]:
    """Circular through/blind openings in a (typically planar) face."""
    shape = sdk_face.wrapped
    wires = _sub_shapes(shape, TopAbs_WIRE, caster=TopoDS.Wire_s)
    outer = BRepTools.OuterWire_s(shape)

    holes = []
    for wire in wires:
        if outer is not None and wire.IsSame(outer):
            continue
        edges = _sub_shapes(wire, TopAbs_EDGE, caster=TopoDS.Edge_s)
        if len(edges) != 1:
            continue  # slot / multi-edge opening: not a round hole in v1
        adaptor = BRepAdaptor_Curve(edges[0])
        if adaptor.GetType() != GeomAbs_CurveType.GeomAbs_Circle:
            continue
        circle = adaptor.Circle()
        center = circle.Location()
        holes.append(HoleInfo(
            center_mm=(center.X(), center.Y(), center.Z()),
            diameter_mm=2.0 * circle.Radius(),
            wire_topo=wire,
        ))
    holes.sort(key=lambda h: (round(h.center_mm[0], 9), round(h.center_mm[1], 9),
                              round(h.center_mm[2], 9)))
    return tuple(holes)


def min_edge_distance_mm(sdk_face, hole: HoleInfo) -> float:
    """Distance from a hole center to the face's outer boundary edges."""
    shape = sdk_face.wrapped
    outer = BRepTools.OuterWire_s(shape)
    if outer is None:
        return float("inf")
    vertex = make_vertex(*hole.center_mm)
    best = float("inf")
    for edge in _sub_shapes(outer, TopAbs_EDGE, caster=TopoDS.Edge_s):
        distance = BRepExtrema_DistShapeShape(vertex, edge).Value()
        best = min(best, float(distance))
    return best


def hole_axis_intersections(body, hole: HoleInfo, sdk_face) -> int:
    """Material hits along the hole axis; 0 means the hole passes through."""
    normal = sdk_face.get_normal_at().to_tuple()
    line = gp_Lin(gp_Pnt(*hole.center_mm), gp_Dir(*normal))
    intersector = _load_intersector(body)
    intersector.Perform(line, -_BIG, _BIG)
    return intersector.NbPnt()


def measure_local_thickness_mm(body, sdk_face) -> Optional[float]:
    """First material depth along the inward normal from the face center."""
    center = sdk_face.get_center()
    normal = sdk_face.get_normal_at().to_tuple()
    inward = tuple(-v for v in normal)
    line = gp_Lin(gp_Pnt(center.x, center.y, center.z), gp_Dir(*inward))
    intersector = _load_intersector(body)
    intersector.Perform(line, 1e-6, _BIG)
    if intersector.NbPnt() == 0:
        return None
    return min(intersector.WParameter(i) for i in range(1, intersector.NbPnt() + 1))


def _load_intersector(body):
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(body.wrapped, 1e-7)
    return intersector


def _sub_shapes(shape, kind, caster=None) -> list:
    mapping = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, mapping)
    items = [mapping.FindKey(i) for i in range(1, mapping.Extent() + 1)]
    if caster is not None:
        items = [caster(item) for item in items]
    return items


def pairwise_pitch_mm(holes) -> dict:
    """{(i, j): distance} between hole centers, i < j."""
    result = {}
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            result[(i, j)] = math.dist(holes[i].center_mm, holes[j].center_mm)
    return result
