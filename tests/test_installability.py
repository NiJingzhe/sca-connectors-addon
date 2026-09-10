"""Executable spec for installability: tool access around fasteners.

The wrench envelope is a cylinder around each hole: radius 0.75·s (across
corners/2 of the ISO 4014 hex head plus clearance), height k + 0.5·d + 5 mm
on the head side, and a nut envelope on the far side below the local
thickness. Any boolean intersection with the part means a tool cannot reach
the fastener — the finding reports the obstruction volume and region.
"""

import pytest

from connverify.installability import check_tool_access
from connverify.joint_geometry import detect_holes
from connverify.joint_types import BoltedThroughSpec
from connverify.package_reader import load_part

from tests.test_joint_geometry import flange_pkg, wall_pkg  # noqa: F401


def _loaded(pkg):
    return load_part(pkg)


def _face(loaded):
    return loaded.interfaces["interface.mount_face"].faces[0].sdk_face


class TestWrenchAccess:
    def test_open_flange_hole_has_clear_tool_access(self, flange_pkg):
        loaded = _loaded(flange_pkg)
        face = _face(loaded)
        holes = detect_holes(face)
        clean = [h for h in holes if round(h.center_mm[0]) == 60][0]
        result = check_tool_access(loaded, face, clean,
                                   BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert result.passed
        assert result.head_obstruction_mm3 == pytest.approx(0.0)
        assert result.nut_obstruction_mm3 == pytest.approx(0.0)

    def test_wall_blocks_the_wrench_envelope(self, wall_pkg):
        loaded = _loaded(wall_pkg)
        face = _face(loaded)
        hole = detect_holes(face)[0]
        result = check_tool_access(loaded, face, hole,
                                   BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert not result.passed
        assert result.head_obstruction_mm3 > 0.0
        assert result.head_obstruction_bbox_mm is not None
        (xmin, _ymin, zmin), (xmax, _ymax, zmax) = result.head_obstruction_bbox_mm
        assert xmax == pytest.approx(12.0, abs=1e-3)   # the wall ends at x=12
        assert zmin >= 10.0 - 1e-6                     # obstruction is above the face

    def test_result_names_the_blocked_side(self, wall_pkg):
        loaded = _loaded(wall_pkg)
        face = _face(loaded)
        hole = detect_holes(face)[0]
        result = check_tool_access(loaded, face, hole,
                                   BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert "head" in " ".join(result.messages).lower()
