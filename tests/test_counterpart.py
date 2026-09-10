"""Executable spec for counterpart-based assemblability verification.

The connection end face's spec DECLARes the mating counterpart's nominal
geometry (hole pattern, mating plate, snap features) in the face's local
frame. connverify generates the counterpart solid FROM the spec and checks
assemblability computationally — no real mating model is ever built.

Local frame contract (deterministic): origin = face bounding-box center;
n = outward face normal; u = global axis least aligned with n, projected
onto the plane and normalized; v = n × u.

Fixtures (from test_joint_geometry):
- flanged_plate: mount face z=10 spans x[0,80] y[0,40] → local (u,v) =
  (x-40, y-20); holes A(-15,-10) B(-15,10) C(20,0), ⌀11.
- walled_flange: same frame plus an upright wall over u ∈ [-40, -28]
  standing ABOVE the mating plane → a mating plate covering that region
  must report interference (穿模).
"""

import pytest

from connverify.counterpart import (
    BoreCounterpart,
    PlateCounterpart,
    PlaneCounterpart,
    SnapCounterpart,
    fit_interference_range_mm,
    face_local_frame,
    project_point_to_face_local,
)
from connverify.joint_types import (
    BoltedThroughSpec, InterferenceSpec, JointKind, RivetedSpec,
    WeldedFilletSpec,
)
from connverify.package_reader import load_part
from connverify.conncheck import check_interface
from connverify.env import Interface

from tests.test_joint_geometry import (  # noqa: F401  fixtures
    build_flanged_plate, build_walled_flange, flange_pkg, wall_pkg,
)
import simplecadapi as scad
from simplecadapi import capture


@pytest.fixture(scope="session")
def flange_part(flange_pkg):
    return load_part(flange_pkg)


@pytest.fixture(scope="session")
def wall_part(wall_pkg):
    return load_part(wall_pkg)


def _face(part):
    return part.interfaces["interface.mount_face"].faces[0].sdk_face


GOOD_PATTERN = ((-15.0, -10.0, 11.0), (-15.0, 10.0, 11.0), (20.0, 0.0, 11.0))


class TestFaceLocalFrame:
    def test_flange_frame_maps_global_to_local(self, flange_part):
        face = _face(flange_part)
        origin, u, v, n = face_local_frame(face)
        assert origin == pytest.approx((40.0, 20.0, 10.0), abs=1e-6)
        assert n == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
        uv = project_point_to_face_local(face, (25.0, 10.0, 10.0))
        assert uv == pytest.approx((-15.0, -10.0), abs=1e-6)

    def test_frame_is_deterministic(self, flange_part):
        face = _face(flange_part)
        assert face_local_frame(face) == face_local_frame(face)


class TestHolePatternMatch:
    def test_matching_pattern_passes(self, flange_part):
        spec = PlateCounterpart(
            thickness_mm=8.0,
            holes=GOOD_PATTERN,
            window=((-40.0, -20.0), (40.0, 20.0)),
        )
        result = spec.check_against(flange_part, _face(flange_part))
        assert result["pattern"]["passed"]
        assert len(result["pattern"]["holes"]) == 3
        assert all(h["deviation_mm"] < 1e-6 for h in result["pattern"]["holes"])

    def test_shifted_pattern_fails_with_per_hole_deviations(self, flange_part):
        shifted = tuple((u + 3.0, v, d) for (u, v, d) in GOOD_PATTERN)
        spec = PlateCounterpart(thickness_mm=8.0, holes=shifted,
                                window=((-40.0, -20.0), (40.0, 20.0)))
        result = spec.check_against(flange_part, _face(flange_part))
        assert not result["pattern"]["passed"]
        assert all(h["deviation_mm"] == pytest.approx(3.0, abs=1e-6)
                   for h in result["pattern"]["holes"])

    def test_count_mismatch_fails(self, flange_part):
        two = GOOD_PATTERN[:2]
        spec = PlateCounterpart(thickness_mm=8.0, holes=two,
                                window=((-40.0, -20.0), (40.0, 20.0)))
        result = spec.check_against(flange_part, _face(flange_part))
        assert not result["pattern"]["passed"]
        assert any("3" in m for m in result["pattern"]["messages"])


class TestMatingInterference:
    def test_open_window_has_no_interference(self, flange_part):
        spec = PlateCounterpart(
            thickness_mm=8.0, holes=GOOD_PATTERN,
            window=((-40.0, -20.0), (40.0, 20.0)))
        result = spec.check_against(flange_part, _face(flange_part))
        assert result["interference"]["passed"]
        assert result["interference"]["volume_mm3"] == pytest.approx(0.0)

    def test_wall_under_the_mating_plane_reports_interference(self, wall_part):
        # walled flange frame origin sits at x=56 (face bbox center of the
        # notched top face); the upright occupies u in [-56, -44] ABOVE the
        # mating plane — a mating plate over that region must interfere
        spec = PlateCounterpart(
            thickness_mm=8.0, holes=((-36.0, 0.0, 11.0),),
            window=((-50.0, -30.0), (-30.0, 30.0)))
        result = spec.check_against(wall_part, _face(wall_part))
        assert not result["interference"]["passed"]
        assert result["interference"]["volume_mm3"] > 0.0


class TestClampSupport:
    def test_washer_overhanging_the_edge_cannot_clamp(self, flange_part):
        # hole A sits 10 mm from the y=0 edge; an M12 washer (⌀24, r_out
        # 10.8 mm) overhangs that edge, an M10 washer (r_out 9) does not
        spec = PlateCounterpart(thickness_mm=8.0, holes=GOOD_PATTERN,
                                window=((-40.0, -20.0), (40.0, 20.0)))
        result = spec.check_against(
            flange_part, _face(flange_part),
            bolt=BoltedThroughSpec(nominal_diameter_mm=12.0))
        assert not result["clamp"]["passed"]
        joined = " ".join(result["clamp"]["messages"])
        assert "-15" in joined

    def test_m10_washer_on_the_same_hole_does_clamp(self, flange_part):
        spec = PlateCounterpart(thickness_mm=8.0, holes=GOOD_PATTERN,
                                window=((-40.0, -20.0), (40.0, 20.0)))
        result = spec.check_against(
            flange_part, _face(flange_part),
            bolt=BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert result["clamp"]["passed"]

    def test_interior_hole_fully_supported(self, wall_part):
        spec = PlateCounterpart(thickness_mm=8.0, holes=((-36.0, 0.0, 11.0),),
                                window=((-50.0, -30.0), (40.0, 30.0)))
        result = spec.check_against(
            wall_part, _face(wall_part),
            bolt=BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert result["clamp"]["passed"]


class TestStackLength:
    def test_short_bolt_cannot_span_the_stack(self, flange_part):
        # our plate is 10 mm, counterpart 8 mm → grip 18 mm; L=15 too short
        spec = PlateCounterpart(thickness_mm=8.0, holes=GOOD_PATTERN,
                                window=((-40.0, -20.0), (40.0, 20.0)),
                                fastener_length_mm=15.0)
        result = spec.check_against(
            flange_part, _face(flange_part),
            bolt=BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert not result["stack"]["passed"]
        assert result["stack"]["grip_mm"] == pytest.approx(18.0)

    def test_adequate_bolt_passes(self, flange_part):
        spec = PlateCounterpart(thickness_mm=8.0, holes=GOOD_PATTERN,
                                window=((-40.0, -20.0), (40.0, 20.0)),
                                fastener_length_mm=30.0)
        result = spec.check_against(
            flange_part, _face(flange_part),
            bolt=BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert result["stack"]["passed"]


class TestFitBandArithmetic:
    def test_h7_r6_on_20mm_shaft_yields_positive_interference(self):
        result = fit_interference_range_mm(
            shaft_diameter_mm=20.0, shaft_class="r6", bore_class="H7")
        lo, hi = result["interference_range_mm"]
        assert lo > 0.0
        # r6(18-30): +28..+41 µm; H7: 0..+21 µm → range [7, 41] µm
        assert (lo, hi) == pytest.approx((0.007, 0.041), abs=1e-9)

    def test_transition_class_reports_clearance_not_interference(self):
        result = fit_interference_range_mm(
            shaft_diameter_mm=20.0, shaft_class="k6", bore_class="H7")
        lo, hi = result["interference_range_mm"]
        assert hi <= 0.0 or lo < 0.0


@scad.part(id="shaft_pin")
def build_shaft_pin() -> scad.Part:
    """⌀20 x 60 shaft; interface.seat = the cylindrical side face."""
    solid = scad.make_cylinder_rsolid(radius=10.0, height=60.0,
                                      bottom_face_center=(0.0, 0.0, 0.0))
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    faces = scad.ql.faces().resolve(solid)
    targets = [
        f for f in faces
        if BRepAdaptor_Surface(f.wrapped).GetType() == GeomAbs_SurfaceType.GeomAbs_Cylinder
        and abs(BRepAdaptor_Surface(f.wrapped).Cylinder().Radius() - 10.0) < 1e-6
    ]
    assert len(targets) == 1
    tagged = scad.apply_tag_rselection(solid, targets, "interface.seat")
    return scad.make_part_rpart(part_id="shaft_pin", body=tagged, name="shaft_pin")


@pytest.fixture(scope="session")
def shaft_pkg(tmp_path_factory):
    path = tmp_path_factory.mktemp("counterpart") / "shaft_pin.scadpkg"
    capture(build_shaft_pin(), path, include_scene=False)
    return str(path)


@pytest.fixture(scope="session")
def shaft_part(shaft_pkg):
    return load_part(shaft_pkg)


class TestBoreCounterpart:
    def test_measured_shaft_diameter_enters_the_fit_band(self, shaft_part):
        face = shaft_part.interfaces["interface.seat"].faces[0].sdk_face
        spec = BoreCounterpart(bore_diameter_mm=20.0)
        result = spec.check_against(
            shaft_part, face,
            joint=InterferenceSpec(nominal_diameter_mm=20.0, fit="H7/r6"))
        assert result["fit"]["passed"]
        assert result["fit"]["measured_diameter_mm"] == pytest.approx(20.0, abs=1e-6)
        assert result["fit"]["shaft_class"] == "r6"
        assert result["fit"]["interference_range_mm"][0] > 0.0

    def test_undersized_shaft_breaks_interference(self, shaft_part):
        face = shaft_part.interfaces["interface.seat"].faces[0].sdk_face
        spec = BoreCounterpart(bore_diameter_mm=21.0)   # bore ⌀21 vs shaft ⌀20
        result = spec.check_against(
            shaft_part, face,
            joint=InterferenceSpec(nominal_diameter_mm=20.0, fit="H7/r6"))
        assert not result["fit"]["passed"]

    def test_insertion_path_along_axis_is_clear(self, shaft_part):
        face = shaft_part.interfaces["interface.seat"].faces[0].sdk_face
        spec = BoreCounterpart(bore_diameter_mm=20.0, bore_length_mm=25.0)
        result = spec.check_against(
            shaft_part, face,
            joint=InterferenceSpec(nominal_diameter_mm=20.0, fit="H7/r6"))
        assert result["insertion"]["passed"]


class TestPlaneCounterpartWeld:
    def test_weld_seam_land_must_be_continuous(self, wall_part):
        face = wall_part.interfaces["interface.mount_face"].faces[0].sdk_face
        # seam window over the plate; counterpart plate lies on the face
        spec = PlaneCounterpart(thickness_mm=8.0,
                                window=((-30.0, -25.0), (30.0, 25.0)))
        result = spec.check_against(
            wall_part, face, joint=WeldedFilletSpec(design_leg_mm=5.0))
        assert result["interference"]["passed"]
        assert result["contact"]["passed"]   # plate lands on the flange face


class TestSnapCounterpart:
    def test_wall_acting_as_hook_is_found_and_slot_is_clear(self, wall_part):
        # frame origin x=56: the upright occupies u in [-56, -44] above the
        # plane — it plays the hook (卡); the open flange plays the 扣 recess
        face = wall_part.interfaces["interface.mount_face"].faces[0].sdk_face
        spec = SnapCounterpart(catch_uv=(-48.0, 0.0), catch_height_mm=2.0,
                               slot_uv=(30.0, 0.0))
        result = spec.check_against(wall_part, face)
        assert result["hook"]["passed"]
        assert result["hook"]["max_height_mm"] >= 8.0   # wall is 45 tall
        assert result["slot"]["passed"]
        assert result["slot"]["max_height_mm"] == 0.0

    def test_hook_declared_in_empty_region_fails(self, wall_part):
        face = wall_part.interfaces["interface.mount_face"].faces[0].sdk_face
        spec = SnapCounterpart(catch_uv=(30.0, 0.0), catch_height_mm=2.0,
                               slot_uv=(30.0, 25.0))
        result = spec.check_against(wall_part, face)
        assert not result["hook"]["passed"]


class TestDslWiring:
    def test_interface_with_spec_and_counterpart_serializes(self):
        from connverify.env import (
            ForceLoad, Interface, LoadCase, Material, VerificationEnv,
        )

        env = VerificationEnv(
            name="counterpart env",
            part_package="out/p.scadpkg",
            material=Material(name="s355", youngs_modulus_mpa=210000.0,
                              poisson_ratio=0.3, yield_strength_mpa=355.0,
                              density_t_per_mm3=7.85e-9),
            interfaces=[Interface(
                name="mount_face",
                spec=BoltedThroughSpec(nominal_diameter_mm=10.0),
                counterpart=PlateCounterpart(
                    thickness_mm=8.0, holes=GOOD_PATTERN,
                    window=((-40.0, -20.0), (40.0, 20.0)),
                    fastener_length_mm=25.0),
            )],
            load_cases=[LoadCase(name="op", loads=[
                ForceLoad(target="mount_face", fz_n=-100.0)])],
        )
        env.validate()
        clone = VerificationEnv.from_json(env.to_json())
        assert clone.interfaces[0].counterpart == env.interfaces[0].counterpart

    def test_conncheck_runs_counterpart_checks(self, flange_part):
        from connverify.env import Interface

        iface = Interface(
            name="mount_face",
            spec=BoltedThroughSpec(nominal_diameter_mm=10.0),
            counterpart=PlateCounterpart(
                thickness_mm=8.0, holes=GOOD_PATTERN,
                window=((-40.0, -20.0), (40.0, 20.0))),
        )
        result = check_interface(flange_part, iface)
        assert result.joint is not None
        assert "counterpart_checks" in result.joint
        assert result.joint["counterpart_checks"]["pattern"]["passed"]

    def test_wrong_counterpart_family_for_kind_is_rejected(self, flange_part):
        from connverify.env import Interface

        iface = Interface(
            name="mount_face",
            spec=BoltedThroughSpec(nominal_diameter_mm=10.0),
            counterpart=BoreCounterpart(bore_diameter_mm=20.0),
        )
        result = check_interface(flange_part, iface)
        joined = " ".join(result.messages)
        assert "does not match joint kind" in joined


class TestEveryKindHasACounterpartPath:
    def test_registry_covers_all_joint_kinds(self):
        from connverify.counterpart import COUNTERPART_FOR_KIND
        from connverify.joint_types import JointKind as JK
        for kind in JK:
            assert kind in COUNTERPART_FOR_KIND, f"{kind} has no counterpart path"
        assert COUNTERPART_FOR_KIND[JointKind.BOLTED_THROUGH] is PlateCounterpart
        assert COUNTERPART_FOR_KIND[JointKind.RIVETED] is PlateCounterpart
        assert COUNTERPART_FOR_KIND[JointKind.PINNED] is BoreCounterpart
        assert COUNTERPART_FOR_KIND[JointKind.BEARING_SEAT] is BoreCounterpart
        assert COUNTERPART_FOR_KIND[JointKind.WELDED_FILLET] is PlaneCounterpart
        assert COUNTERPART_FOR_KIND[JointKind.ADHESIVE] is PlaneCounterpart
