"""Installability checks: can a tool actually reach the fastener?

Wrench envelope per bolt (ISO 4014 head geometry):
- head side (outside the mating face): cylinder radius 0.75·s,
  height k + 0.5·d + 5 mm, starting at the face and growing outward;
- nut side: same radius below the local plate thickness, height 0.8·d + 4 mm.

Any boolean intersection with the part is an obstruction: the finding
reports its volume and bounding region so the agent knows WHAT to move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from OCP.Bnd import Bnd_Box
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

from .joint_geometry import HoleInfo, measure_local_thickness_mm
from .joint_types import BoltedThroughSpec

_VOLUME_EPS_MM3 = 1e-6


@dataclass(frozen=True)
class ToolAccessResult:
    passed: bool
    head_obstruction_mm3: float
    nut_obstruction_mm3: float
    head_obstruction_bbox_mm: Optional[Tuple[Tuple[float, float, float],
                                              Tuple[float, float, float]]]
    messages: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        def _bbox(bbox):
            if bbox is None:
                return None
            (xmin, ymin, zmin), (xmax, ymax, zmax) = bbox
            return {"min": [xmin, ymin, zmin], "max": [xmax, ymax, zmax]}
        return {
            "passed": self.passed,
            "head_obstruction_mm3": round(self.head_obstruction_mm3, 6),
            "nut_obstruction_mm3": round(self.nut_obstruction_mm3, 6),
            "head_obstruction_bbox_mm": _bbox(self.head_obstruction_bbox_mm),
            "messages": list(self.messages),
        }


def check_tool_access(
    loaded, sdk_face, hole: HoleInfo, spec: BoltedThroughSpec
) -> ToolAccessResult:
    normal = sdk_face.get_normal_at().to_tuple()
    radius = spec.tool_radius_mm()
    messages = []

    head = _cylinder(
        center=hole.center_mm, direction=normal,
        radius=radius, height=spec.tool_height_mm(),
    )
    head_volume, head_bbox = _intersection(loaded.body, head)

    thickness = measure_local_thickness_mm(loaded.body, sdk_face)
    nut_volume, nut_bbox = 0.0, None
    if thickness is not None:
        inward = tuple(-v for v in normal)
        nut_center = tuple(
            c + n * (thickness + 0.5) for c, n in zip(hole.center_mm, inward))
        nut = _cylinder(
            center=nut_center, direction=inward,
            radius=radius, height=0.8 * spec.nominal_diameter_mm + 4.0,
        )
        nut_volume, nut_bbox = _intersection(loaded.body, nut)

    passed = head_volume <= _VOLUME_EPS_MM3 and nut_volume <= _VOLUME_EPS_MM3
    if head_volume > _VOLUME_EPS_MM3:
        messages.append(
            f"wrench cannot reach the bolt head from outside: obstruction "
            f"{head_volume:.1f} mm³ within the tool envelope (r={radius:.1f} mm, "
            f"h={spec.tool_height_mm():.1f} mm) — move material away or relocate "
            "the bolt"
        )
    if nut_volume > _VOLUME_EPS_MM3:
        messages.append(
            f"nut cannot be fitted under the joint: obstruction "
            f"{nut_volume:.1f} mm³ in the nut envelope"
        )
    return ToolAccessResult(
        passed=passed,
        head_obstruction_mm3=head_volume,
        nut_obstruction_mm3=nut_volume,
        head_obstruction_bbox_mm=head_bbox if head_volume > _VOLUME_EPS_MM3 else None,
        messages=tuple(messages),
    )


def _cylinder(center, direction, radius, height):
    axis = gp_Ax2(
        gp_Pnt(*map(float, center)),
        gp_Dir(*map(float, direction)),
    )
    return BRepPrimAPI_MakeCylinder(axis, float(radius), float(height)).Solid()


def _intersection(body, tool):
    common = BRepAlgoAPI_Common(body.wrapped, tool)
    common.Build()
    if not common.IsDone():
        raise RuntimeError("OCC boolean failed during tool-access check")
    shape = common.Shape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = float(props.Mass())
    if volume <= _VOLUME_EPS_MM3:
        return 0.0, None
    bounds = Bnd_Box()
    BRepBndLib.Add_s(shape, bounds, True)
    xmin, ymin, zmin, xmax, ymax, zmax = bounds.Get()
    return volume, ((xmin, ymin, zmin), (xmax, ymax, zmax))
