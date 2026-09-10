"""Keep-out envelope checks via exact OCC boolean intersection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from OCP.Bnd import Bnd_Box
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Pnt

from .env import KeepOutBox
from .package_reader import LoadedPart

_VOLUME_EPS_MM3 = 1e-6


@dataclass(frozen=True)
class EnvelopeCheckResult:
    name: str
    passed: bool
    intersection_volume_mm3: float
    intersection_bbox_mm: Optional[Tuple[Tuple[float, float, float],
                                          Tuple[float, float, float]]]

    def to_dict(self) -> dict:
        bbox = None
        if self.intersection_bbox_mm is not None:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = self.intersection_bbox_mm
            bbox = {"min": [xmin, ymin, zmin], "max": [xmax, ymax, zmax]}
        return {
            "envelope": self.name,
            "passed": self.passed,
            "intersection_volume_mm3": round(self.intersection_volume_mm3, 6),
            "intersection_bbox_mm": bbox,
        }


def check_envelope(loaded: LoadedPart, keepout: KeepOutBox) -> EnvelopeCheckResult:
    solid = loaded.body.wrapped
    box = BRepPrimAPI_MakeBox(
        gp_Pnt(*map(float, keepout.min_corner_mm)),
        gp_Pnt(*map(float, keepout.max_corner_mm)),
    ).Solid()

    common = BRepAlgoAPI_Common(solid, box)
    common.Build()
    if not common.IsDone():
        raise RuntimeError("OCC boolean intersection failed for keep-out "
                           f"{keepout.name!r}")

    shape = common.Shape()
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume = float(props.Mass())

    if volume <= _VOLUME_EPS_MM3:
        return EnvelopeCheckResult(
            name=keepout.name, passed=True,
            intersection_volume_mm3=0.0, intersection_bbox_mm=None,
        )

    bounds = Bnd_Box()
    BRepBndLib.Add_s(shape, bounds, True)
    xmin, ymin, zmin, xmax, ymax, zmax = bounds.Get()
    return EnvelopeCheckResult(
        name=keepout.name,
        passed=False,
        intersection_volume_mm3=volume,
        intersection_bbox_mm=((xmin, ymin, zmin), (xmax, ymax, zmax)),
    )


def check_envelopes(loaded: LoadedPart, keepouts) -> tuple:
    return tuple(check_envelope(loaded, k) for k in keepouts)
