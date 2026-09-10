"""Mesh data model: pure data consumed by the deck generator, produced by meshing."""

import pytest

from connverify.mesh_model import Mesh, MeshFace, Tet


def single_tet_mesh() -> Mesh:
    """Unit tet with vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1).

    - interface "mount": the z=0 face, nodes {1,2,3}, outward normal (0,0,-1),
      area 0.5 mm², tributary 1/6 mm² per node.
    - interface "load": the y=0 face, nodes {1,2,4}, outward normal (0,-1,0),
      area 0.5 mm², tributary 1/6 mm² per node.
    """
    nodes = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0),
             3: (0.0, 1.0, 0.0), 4: (0.0, 0.0, 1.0)}
    return Mesh(
        nodes=nodes,
        tets=(Tet(element_id=1, node_ids=(1, 2, 3, 4)),),
        interface_faces={
            "mount": (MeshFace(
                topo_id="face_mount", normal=(0.0, 0.0, -1.0),
                nodes=(1, 2, 3),
                tributary_area_mm2={1: 1.0 / 6, 2: 1.0 / 6, 3: 1.0 / 6},
            ),),
            "load": (MeshFace(
                topo_id="face_load", normal=(0.0, -1.0, 0.0),
                nodes=(1, 2, 4),
                tributary_area_mm2={1: 1.0 / 6, 2: 1.0 / 6, 4: 1.0 / 6},
            ),),
        },
    )


class TestMesh:
    def test_interface_nodes_are_unique_and_sorted(self):
        mesh = single_tet_mesh()
        assert mesh.interface_nodes("mount") == (1, 2, 3)
        assert mesh.interface_nodes("load") == (1, 2, 4)

    def test_interface_area_sums_tributaries(self):
        mesh = single_tet_mesh()
        assert mesh.interface_area_mm2("mount") == pytest.approx(0.5)
        assert mesh.interface_area_mm2("load") == pytest.approx(0.5)

    def test_unknown_interface_raises_keyerror(self):
        with pytest.raises(KeyError):
            single_tet_mesh().interface_nodes("ghost")

    def test_node_count_and_tet_count(self):
        mesh = single_tet_mesh()
        assert mesh.node_count == 4
        assert mesh.tet_count == 1
