"""Executable spec for connection checks (interface geometric verification)."""

import pytest

from connverify.conncheck import check_interface, InterfaceCheckResult
from connverify.env import ConnectionMethod, Interface
from connverify.joint_types import (
    BoltedTappedSpec, BoltedThroughSpec, PinnedSpec, WeldedFilletSpec,
)
from connverify.package_reader import load_part

from tests.test_joint_geometry import (  # noqa: F401  (fixture builders)
    blind_pkg, flange_pkg, wall_pkg,
)


@pytest.fixture(scope="module")
def loaded(probe_box_pkg):
    return load_part(probe_box_pkg)


class TestPlanarInterfaces:
    def test_planar_bolted_face_passes(self, loaded):
        iface = Interface(name="mount_face", method=ConnectionMethod.BOLTED)
        result = check_interface(loaded, iface)
        assert isinstance(result, InterfaceCheckResult)
        assert result.passed
        assert result.planarity_deviation_mm == pytest.approx(0.0, abs=1e-9)
        assert result.face_count == 1
        assert result.total_area_mm2 == pytest.approx(60.0 * 30.0, rel=1e-6)
        assert result.normal == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)

    def test_result_names_the_interface_and_method(self, loaded):
        result = check_interface(
            loaded, Interface(name="mount_face", method=ConnectionMethod.WELDED))
        assert result.interface_name == "mount_face"
        assert result.method == ConnectionMethod.WELDED

    def test_min_area_violation_fails_with_numbers(self, loaded):
        iface = Interface(name="mount_face", method=ConnectionMethod.BOLTED,
                          min_area_mm2=5000.0)
        result = check_interface(loaded, iface)
        assert not result.passed
        assert any("5000" in m for m in result.messages)

    def test_min_area_satisfied_passes(self, loaded):
        iface = Interface(name="mount_face", method=ConnectionMethod.BOLTED,
                          min_area_mm2=1000.0)
        assert check_interface(loaded, iface).passed


class TestProvenance:
    def test_face_provenance_maps_topo_ids_to_features(self, loaded):
        prov = loaded.face_provenance
        mount = loaded.interfaces["interface.mount_face"].faces[0].topo_id
        assert mount in prov
        assert prov[mount]["graph_id"] == "probe_box"
        assert prov[mount]["node_id"].startswith("node_")

    def test_all_faces_have_provenance(self, loaded):
        # a box has 6 faces; every BREP face should carry feature_output
        assert len(loaded.face_provenance) == 6


class TestJointSpecDispatch:
    """The connection TYPE drives which geometric rules actually run."""

    def _check(self, pkg, spec, name="mount_face"):
        part = load_part(pkg)
        return check_interface(part, Interface(name=name, spec=spec))

    def test_bolted_layout_violations_cite_numbers_and_standard(self, flange_pkg):
        result = self._check(flange_pkg, BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert not result.passed
        joined = " ".join(result.messages)
        assert "15.0" in joined          # 1.5d edge limit vs measured 10
        assert "25.0" in joined          # 2.5d pitch limit vs measured 20
        assert "EN 1993" in joined or "BS 5950" in joined
        assert result.joint["holes"][0]["edge_mm"] == pytest.approx(10.0, abs=1e-6)
        assert len(result.joint["holes"]) == 3

    def test_bolted_through_required_blind_hole_fails(self, blind_pkg):
        result = self._check(blind_pkg, BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert not result.passed
        assert any("through" in m for m in result.messages)

    def test_bolted_tapped_accepts_a_blind_hole(self, blind_pkg):
        result = self._check(blind_pkg, BoltedTappedSpec(nominal_diameter_mm=10.0))
        assert result.passed

    def test_bolted_expected_count_mismatch_fails(self, flange_pkg):
        spec = BoltedThroughSpec(nominal_diameter_mm=10.0, expected_count=4)
        result = self._check(flange_pkg, spec)
        assert not result.passed
        assert any("4" in m and "3" in m for m in result.messages)

    def test_wrench_blockage_fails_the_interface(self, wall_pkg):
        result = self._check(wall_pkg, BoltedThroughSpec(nominal_diameter_mm=10.0))
        assert not result.passed
        assert any("wrench" in m.lower() for m in result.messages)

    def test_pinned_bore_checked_against_h7_limits(self, flange_pkg):
        result = self._check(flange_pkg, PinnedSpec(pin_diameter_mm=10.0))
        assert not result.passed
        joined = " ".join(result.messages)
        assert "H7" in joined and "10.015" in joined  # measured 11 vs [10, 10.015]

    def test_weld_leg_rule_uses_local_thickness(self, probe_box_pkg):
        # probe box top face: local thickness 30 mm -> min leg 8.5 mm
        too_thin = self._check(probe_box_pkg, WeldedFilletSpec(design_leg_mm=5.0),
                               name="load_pad")
        assert not too_thin.passed
        assert any("8.5" in m for m in too_thin.messages)
        ok = self._check(probe_box_pkg, WeldedFilletSpec(design_leg_mm=10.0),
                         name="load_pad")
        assert ok.passed

    def test_weld_without_declared_leg_gets_advisory_not_failure(self, probe_box_pkg):
        result = self._check(probe_box_pkg, WeldedFilletSpec(), name="load_pad")
        assert result.passed
        assert result.joint["notes"]
