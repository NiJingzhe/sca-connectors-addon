"""Shared fixtures: one durable .scadpkg probe part, built once per session."""

import pytest

import simplecadapi as scad
from simplecadapi import capture


@scad.part(id="probe_box")
def build_probe_box() -> scad.Part:
    """A 60(x) x 40(y) x 30(z) box standing on z=0.

    - interface.mount_face: the y=0 side (SDK "left"), area 60*30 = 1800 mm²,
      center (30, 0, 15), outward normal -Y.
    - interface.load_pad: the z=30 top face, area 60*40 = 2400 mm²,
      center (30, 20, 30), outward normal +Z.
    """
    solid = scad.make_box_rsolid(
        width=60.0,
        height=40.0,
        depth=30.0,
        bottom_face_center=(30.0, 20.0, 0.0),
        left_face_tag="interface.mount_face",
        top_face_tag="interface.load_pad",
    )
    return scad.make_part_rpart(part_id="probe_box", body=solid, name="probe_box")


@pytest.fixture(scope="session")
def probe_box_pkg(tmp_path_factory):
    path = tmp_path_factory.mktemp("pkg") / "probe_box.scadpkg"
    result = build_probe_box()
    capture(result, path, include_scene=False)
    return str(path)
