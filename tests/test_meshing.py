"""Executable spec for gmsh meshing + exact node-face association.

Association is exact, not heuristic: every boundary node is classified against
the interface's BREP faces with an OCC distance query (point-to-trimmed-face),
so a node belongs to an interface face only when it geometrically lies on it.
"""

import pytest

from connverify.meshing import mesh_part
from connverify.package_reader import load_part


@pytest.fixture(scope="module")
def loaded(probe_box_pkg):
    return load_part(probe_box_pkg)


@pytest.fixture(scope="module")
def mesh(loaded):
    return mesh_part(loaded, mesh_size_mm=10.0)


class TestMeshBasics:
    def test_produces_nodes_and_linear_tets(self, mesh):
        assert mesh.node_count > 10
        assert mesh.tet_count > 10
        for tet in mesh.tets:
            assert len(set(tet.node_ids)) == 4
            for n in tet.node_ids:
                assert n in mesh.nodes

    def test_nodes_lie_within_the_part_bounding_box(self, mesh):
        for x, y, z in mesh.nodes.values():
            assert -1e-6 <= x <= 60.0 + 1e-6
            assert -1e-6 <= y <= 40.0 + 1e-6
            assert -1e-6 <= z <= 30.0 + 1e-6

    def test_mesh_size_controls_resolution(self, loaded, mesh):
        fine = mesh_part(loaded, mesh_size_mm=5.0)
        assert fine.node_count > mesh.node_count


class TestInterfaceAssociation:
    def test_mount_face_nodes_all_lie_on_y0(self, mesh):
        nodes = mesh.interface_nodes("mount_face")
        assert len(nodes) >= 4
        for node_id in nodes:
            assert mesh.nodes[node_id][1] == pytest.approx(0.0, abs=1e-4)

    def test_load_pad_nodes_all_lie_on_z30(self, mesh):
        nodes = mesh.interface_nodes("load_pad")
        assert len(nodes) >= 4
        for node_id in nodes:
            assert mesh.nodes[node_id][2] == pytest.approx(30.0, abs=1e-4)

    def test_interface_areas_match_the_analytic_faces(self, mesh):
        assert mesh.interface_area_mm2("mount_face") == pytest.approx(60.0 * 30.0, rel=0.02)
        assert mesh.interface_area_mm2("load_pad") == pytest.approx(60.0 * 40.0, rel=0.02)

    def test_face_normals_come_from_the_brep(self, mesh):
        mount = mesh.interface_faces["mount_face"][0]
        assert mount.normal == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)
        pad = mesh.interface_faces["load_pad"][0]
        assert pad.normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)

    def test_faces_carry_their_brep_topo_id(self, mesh, loaded):
        mount_brep = loaded.interfaces["interface.mount_face"].faces[0].topo_id
        assert mesh.interface_faces["mount_face"][0].topo_id == mount_brep


class TestDeterminism:
    def test_same_input_same_mesh(self, loaded, mesh):
        again = mesh_part(loaded, mesh_size_mm=10.0)
        assert again.node_count == mesh.node_count
        assert sorted(again.nodes.items()) == sorted(mesh.nodes.items())
        assert again.tet_count == mesh.tet_count


class TestGuard:
    def test_unknown_interface_yields_no_faces(self, mesh):
        assert "ghost" not in mesh.interface_faces
