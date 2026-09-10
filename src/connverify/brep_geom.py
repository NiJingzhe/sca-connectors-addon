"""Thin OCP helpers: BREP export and exact point-to-face distances."""

from __future__ import annotations

import tempfile
from pathlib import Path

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt

from simplecadapi.artifacts.brep import write_brep_bytes


def write_brep_to_tempfile(solid) -> str:
    payload = write_brep_bytes(solid)
    handle = tempfile.NamedTemporaryFile(
        suffix=".brep", prefix="connverify_", delete=False
    )
    handle.write(payload)
    handle.close()
    return handle.name


def make_vertex(x: float, y: float, z: float):
    return BRepBuilderAPI_MakeVertex(gp_Pnt(float(x), float(y), float(z))).Vertex()


def point_to_face_distance(vertex, face_shape) -> float:
    computer = BRepExtrema_DistShapeShape(vertex, face_shape)
    return float(computer.Value())
