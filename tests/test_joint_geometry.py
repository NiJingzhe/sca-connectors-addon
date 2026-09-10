"""Executable spec for joint-face geometry detection.

Fixtures: a flanged plate with three deliberately mis-compliant M10 holes
(A: edge violation, B: pitch violation with A, C: clean), a walled flange
whose bolt loses wrench access to the upright, and a blind-hole plate.
"""

import pytest

import simplecadapi as scad
from simplecadapi import capture

from connverify.joint_geometry import (
    HoleInfo,
    detect_holes,
    hole_axis_intersections,
    measure_local_thickness_mm,
)
from connverify.package_reader import load_part


def _tag_top_face(body, z_expected: float, tag: str):
    """QL-enumerate faces, pick the single planar +Z face at z_expected, tag it.

    Kwarg face tags do not reliably survive multi-tool booleans, so the
    interface name is (re)attached to the FINAL geometry — playbook pattern B.
    """
    faces = scad.ql.faces().resolve(body)
    targets = [
        f for f in faces
        if f.get_normal_at().to_tuple()[2] > 0.999
        and abs(f.get_center().z - z_expected) < 1e-6
    ]
    assert len(targets) == 1, (
        f"expected one +Z face at z={z_expected}, got {len(targets)}")
    return scad.apply_tag_rselection(body, targets, tag)


@scad.part(id="flanged_plate")
def build_flanged_plate() -> scad.Part:
    """Plate 80(x) x 40(y) x 10(z). interface.mount_face = z=10 top face.

    Three through-holes (d0=11, M10 medium):
      A (25, 10) — 10 mm from the y=0 edge < 15 mm (1.5d)  -> edge violation
      B (25, 30) — 20 mm pitch to A < 25 mm (2.5d)         -> pitch violation
      C (60, 20) — clean: edges >= 20 mm, pitches >= 36 mm
    """
    plate = scad.make_box_rsolid(
        width=80.0, height=40.0, depth=10.0,
        bottom_face_center=(40.0, 20.0, 0.0),
    )
    tools = [
        scad.make_cylinder_rsolid(radius=5.5, height=30.0,
                                  bottom_face_center=(x, y, -10.0))
        for (x, y) in ((25.0, 10.0), (25.0, 30.0), (60.0, 20.0))
    ]
    result = scad.cut_rsolid(plate, *tools)
    tagged = _tag_top_face(result, 10.0, "interface.mount_face")
    return scad.make_part_rpart(part_id="flanged_plate", body=tagged,
                                name="flanged_plate")


@scad.part(id="walled_flange")
def build_walled_flange() -> scad.Part:
    """Flange 100x60x10 with an upright wall (x<=12, z up to 55).

    interface.mount_face = flange top face; one hole at (20, 30) whose wrench
    envelope (r=12 for M10) necessarily intersects the wall.
    """
    flange = scad.make_box_rsolid(
        width=100.0, height=60.0, depth=10.0,
        bottom_face_center=(50.0, 30.0, 0.0),
    )
    wall = scad.make_box_rsolid(width=12.0, height=60.0, depth=45.0,
                                bottom_face_center=(6.0, 30.0, 10.0))
    body = scad.union_rsolid(flange, wall)
    tool = scad.make_cylinder_rsolid(radius=5.5, height=30.0,
                                     bottom_face_center=(20.0, 30.0, -10.0))
    result = scad.cut_rsolid(body, tool)
    tagged = _tag_top_face(result, 10.0, "interface.mount_face")
    return scad.make_part_rpart(part_id="walled_flange", body=tagged,
                                name="walled_flange")


@scad.part(id="blind_plate")
def build_blind_plate() -> scad.Part:
    """Plate 40x40x10 with one BLIND hole (⌀11, 5 mm deep) in the top face."""
    plate = scad.make_box_rsolid(
        width=40.0, height=40.0, depth=10.0,
        bottom_face_center=(20.0, 20.0, 0.0),
    )
    tool = scad.make_cylinder_rsolid(radius=5.5, height=5.0,
                                     bottom_face_center=(20.0, 20.0, 5.0))
    result = scad.cut_rsolid(plate, tool)
    tagged = _tag_top_face(result, 10.0, "interface.mount_face")
    return scad.make_part_rpart(part_id="blind_plate", body=tagged,
                                name="blind_plate")


@pytest.fixture(scope="session")
def flange_pkg(tmp_path_factory):
    path = tmp_path_factory.mktemp("joint") / "flanged_plate.scadpkg"
    capture(build_flanged_plate(), path, include_scene=False)
    return str(path)


@pytest.fixture(scope="session")
def wall_pkg(tmp_path_factory):
    path = tmp_path_factory.mktemp("joint") / "walled_flange.scadpkg"
    capture(build_walled_flange(), path, include_scene=False)
    return str(path)


@pytest.fixture(scope="session")
def blind_pkg(tmp_path_factory):
    path = tmp_path_factory.mktemp("joint") / "blind_plate.scadpkg"
    capture(build_blind_plate(), path, include_scene=False)
    return str(path)


def _face(pkg):
    loaded = load_part(pkg)
    return loaded, loaded.interfaces["interface.mount_face"].faces[0].sdk_face


class TestHoleDetection:
    def test_three_holes_found_with_exact_diameters(self, flange_pkg):
        _loaded, face = _face(flange_pkg)
        holes = detect_holes(face)
        assert len(holes) == 3
        by_x = {round(h.center_mm[0]): h for h in holes}
        for hole in holes:
            assert hole.diameter_mm == pytest.approx(11.0, abs=1e-6)
        assert set(by_x) == {25, 60}

    def test_hole_centers_are_on_the_face_plane(self, flange_pkg):
        _loaded, face = _face(flange_pkg)
        for hole in detect_holes(face):
            assert hole.center_mm[2] == pytest.approx(10.0, abs=1e-6)

    def test_walled_flange_hole_survives_the_union_and_cut(self, wall_pkg):
        _loaded, face = _face(wall_pkg)
        holes = detect_holes(face)
        assert len(holes) == 1
        assert holes[0].center_mm[:2] == pytest.approx((20.0, 30.0), abs=1e-6)


class TestThroughHole:
    def test_clear_axis_means_through(self, flange_pkg):
        loaded, face = _face(flange_pkg)
        for hole in detect_holes(face):
            assert hole_axis_intersections(loaded.body, hole, face) == 0

    def test_blind_hole_hits_material_on_its_axis(self, blind_pkg):
        loaded, face = _face(blind_pkg)
        holes = detect_holes(face)
        assert len(holes) == 1
        assert hole_axis_intersections(loaded.body, holes[0], face) >= 1


class TestLocalThickness:
    def test_flange_thickness_measured_from_the_face(self, flange_pkg):
        loaded, face = _face(flange_pkg)
        thickness = measure_local_thickness_mm(loaded.body, face)
        assert thickness == pytest.approx(10.0, abs=1e-6)

    def test_walled_flange_top_face_thickness_is_the_flange(self, wall_pkg):
        loaded, face = _face(wall_pkg)
        thickness = measure_local_thickness_mm(loaded.body, face)
        assert thickness == pytest.approx(10.0, abs=1e-6)
