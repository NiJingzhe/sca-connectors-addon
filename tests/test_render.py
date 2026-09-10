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
