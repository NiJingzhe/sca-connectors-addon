"""Executable spec for the tag-review renderer.

Purpose: an agent LOOKS at the PNG to confirm that every interface tag was
attached to the intended end face — the highlight comes from the exact
triangles the checker associated, so a misplaced tag shows up as a highlight
on the wrong face.
"""

import os

import pytest

from connverify.meshing import mesh_part
from connverify.package_reader import load_part
from connverify.render import render_tag_review

from tests.test_joint_geometry import flange_pkg  # noqa: F401


@pytest.fixture(scope="module")
def mesh(probe_box_pkg):
    return mesh_part(load_part(probe_box_pkg), mesh_size_mm=10.0)


@pytest.fixture(scope="module")
def flange_mesh(flange_pkg):
    return mesh_part(load_part(flange_pkg), mesh_size_mm=8.0)


class TestMeshFaceTriangles:
    def test_face_carries_its_triangles(self, mesh):
        face = mesh.interface_faces["mount_face"][0]
        assert len(face.triangles) > 0
        assert set(face.nodes) == {n for tri in face.triangles for n in tri}

    def test_mount_face_triangles_lie_on_y0(self, mesh):
        face = mesh.interface_faces["mount_face"][0]
        for a, b, c in face.triangles:
            for n in (a, b, c):
                assert mesh.nodes[n][1] == pytest.approx(0.0, abs=1e-4)

    def test_tributary_area_still_consistent(self, mesh):
        # adding triangles must not disturb the load-distribution weights
        assert mesh.interface_area_mm2("mount_face") == pytest.approx(
            60.0 * 30.0, rel=0.02)


class TestRenderTagReview:
    def test_png_written_with_panel_facts(self, mesh, tmp_path):
        out = tmp_path / "review.png"
        facts = render_tag_review(mesh, out, title="probe box")
        assert out.is_file()
        assert out.stat().st_size > 10_000
        assert facts["png"] == str(out)
        panels = {p["interface"]: p for p in facts["panels"]
                  if p["interface"] is not None}
        assert set(panels) == {"mount_face", "load_pad"}
        for name, panel in panels.items():
            assert panel["triangles"] > 0
            assert panel["tag"] == f"interface.{name}"
            assert panel["color"]

    def test_iso_panel_comes_first(self, mesh, tmp_path):
        out = tmp_path / "review.png"
        facts = render_tag_review(mesh, out)
        assert facts["panels"][0]["interface"] is None  # overview panel

    def test_render_is_headless_and_deterministic(self, mesh, tmp_path):
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        render_tag_review(mesh, a)
        render_tag_review(mesh, b)
        assert a.read_bytes() == b.read_bytes()

    def test_flange_with_holes_renders(self, flange_mesh, tmp_path):
        out = tmp_path / "flange.png"
        facts = render_tag_review(flange_mesh, out)
        assert out.is_file()
        mount = [p for p in facts["panels"] if p["interface"] == "mount_face"][0]
        assert mount["triangles"] > 10


class TestStressContour:
    """FEM result rendering: node von Mises painted on boundary triangles."""

    @staticmethod
    def _synthetic():
        from connverify.frd import FrdBlock, FrdResult
        from connverify.mesh_model import Mesh, Tet

        nodes = {1: (0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0),
                 3: (0.0, 10.0, 0.0), 4: (0.0, 0.0, 10.0)}
        mesh = Mesh(nodes=nodes,
                    tets=(Tet(element_id=1, node_ids=(1, 2, 3, 4)),),
                    interface_faces={})
        stress = FrdBlock(
            name="STRESS",
            components=("SXX", "SYY", "SZZ", "SYZ", "SZX", "SXY"),
            values={
                1: (100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                2: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                3: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                4: (50.0, 50.0, 0.0, 0.0, 0.0, 0.0),
            })
        frd = FrdResult(nodes=nodes, blocks={"STRESS": stress})
        return mesh, frd

    def test_contour_png_written_with_facts(self, tmp_path):
        from connverify.render import render_stress_contour

        mesh, frd = self._synthetic()
        out = tmp_path / "stress.png"
        facts = render_stress_contour(mesh, frd, out)
        assert out.is_file()
        assert out.stat().st_size > 10_000
        assert facts["png"] == str(out)
        assert facts["max_von_mises_mpa"] == pytest.approx(100.0)
        assert facts["hotspot_node"] == 1
        assert facts["triangles"] == 4
        # multi-view: the hotspot view alone can hide geometry behind other
        # bodies, so every contour carries deterministic fallback views
        assert facts["views"] == ("hotspot", "front (X-Z)", "side (Y-Z)",
                                  "top (X-Y)")

    def test_contour_is_headless_and_deterministic(self, tmp_path):
        from connverify.render import render_stress_contour

        mesh, frd = self._synthetic()
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        render_stress_contour(mesh, frd, a)
        render_stress_contour(mesh, frd, b)
        assert a.read_bytes() == b.read_bytes()


class TestRenderConvergence:
    def _cases(self):
        from connverify.convergence import QoiPoint, analyze_convergence
        good = analyze_convergence("press", tuple(QoiPoint(
            size_mm=s, node_count=100 * i, tet_count=400 * i,
            max_von_mises_mpa=v, safety_factor=355.0 / v,
            max_displacement_mm=0.01)
            for i, (s, v) in enumerate(zip((20.0, 10.0, 5.0),
                                           (92.0, 98.0, 99.5)))),
            qoi_tolerance_pct=2.0)
        bad = analyze_convergence("shock", tuple(QoiPoint(
            size_mm=s, node_count=50 * i, tet_count=200 * i,
            max_von_mises_mpa=v, safety_factor=355.0 / v,
            max_displacement_mm=0.02)
            for i, (s, v) in enumerate(zip((20.0, 10.0, 5.0),
                                           (250.0, 290.0, 335.0)))),
            qoi_tolerance_pct=2.0)
        return (good, bad)

    def test_png_written_with_case_facts(self, tmp_path):
        from connverify.render import render_convergence

        out = tmp_path / "conv.png"
        facts = render_convergence(self._cases(), out, title="study",
                                   tolerance_pct=2.0)
        assert out.is_file()
        assert out.stat().st_size > 10_000
        assert facts["png"] == str(out)
        assert facts["tolerance_pct"] == 2.0
        names = {c["name"]: c for c in facts["cases"]}
        assert names["press"]["converged"] is True
        assert names["shock"]["converged"] is False

    def test_render_is_headless_and_deterministic(self, tmp_path):
        from connverify.render import render_convergence

        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        render_convergence(self._cases(), a)
        render_convergence(self._cases(), b)
        assert a.read_bytes() == b.read_bytes()

    def test_case_without_displacement_still_renders(self, tmp_path):
        from connverify.convergence import QoiPoint, analyze_convergence
        from connverify.render import render_convergence

        case = analyze_convergence("press", tuple(QoiPoint(
            size_mm=s, node_count=10, tet_count=40, max_von_mises_mpa=v)
            for s, v in zip((20.0, 10.0), (98.0, 99.0))),
            qoi_tolerance_pct=5.0)
        out = tmp_path / "nodisp.png"
        facts = render_convergence((case,), out)
        assert out.is_file()
        assert facts["cases"][0]["name"] == "press"
