"""Connection-interface geometric verification.

For every declared interface (already resolved to exact BREP faces):
- planarity deviation of the mating faces (0 for exact planes; least-squares
  plane fit over surface samples otherwise);
- total bearing area and face count;
- declared minimum-area expectation;
- outward normal and centroid of the (single-face) interface.

Bolt-hole pattern detection is deliberately v2: v1 verdicts rest on facts the
BREP states exactly.
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
from .package_reader import LoadedPart

_CONSTRaining_METHODS_HINT = "mating face"


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
            f"{_CONSTRaining_METHODS_HINT} not planar: deviation "
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

    normal = centroid = None
    if len(info.faces) == 1:
        normal = tuple(float(v) for v in info.faces[0].sdk_face.get_normal_at().to_tuple())
        c = info.faces[0].sdk_face.get_center()
        centroid = (float(c.x), float(c.y), float(c.z))

    passed = planarity_ok and area_ok
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
    )


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
