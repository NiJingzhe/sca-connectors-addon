"""Executable spec for connection checks (interface geometric verification)."""

import pytest

from connverify.conncheck import check_interface, InterfaceCheckResult
from connverify.env import ConnectionMethod, Interface
from connverify.package_reader import load_part


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
