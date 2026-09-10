"""Executable spec for the .scadpkg reader layer.

Consumer contract (from the scadpkg-format spec):
- resolve interface names through the topology snapshot's name_index;
- a required interface name that is absent is a LOUD failure naming every
  missing name — never a guess by geometry;
- the part solid must be rebuildable with tags restored (in-process SDK path).
"""

import pytest

from connverify.package_reader import (
    InterfaceFace,
    LoadedPart,
    MissingInterfaceError,
    UnsupportedPackageError,
    load_part,
    require_interfaces,
)


@pytest.fixture(scope="module")
def loaded(probe_box_pkg):
    return load_part(probe_box_pkg)


class TestLoadPart:
    def test_returns_single_solid_part_with_body(self, loaded):
        assert isinstance(loaded, LoadedPart)
        assert loaded.definition_kind == "single_solid"
        assert loaded.body is not None

    def test_interface_names_come_from_the_tag_channel(self, loaded):
        assert loaded.available_interface_names == frozenset({
            "interface.mount_face", "interface.load_pad",
        })

    def test_interface_face_carries_snapshot_facts(self, loaded):
        info = loaded.interfaces["interface.mount_face"]
        assert len(info.faces) == 1
        face = info.faces[0]
        assert isinstance(face, InterfaceFace)
        assert face.topo_id  # snapshot identity, not a guess
        assert face.geometry_hash.startswith("sha256:")
        assert face.feature_node_id  # provenance for geometry-directed feedback
        assert face.sdk_face is not None  # live geometry for checks/meshing

    def test_face_geometry_is_queryable(self, loaded):
        mount = loaded.interfaces["interface.mount_face"].faces[0]
        area = mount.sdk_face.get_area()
        center = mount.sdk_face.get_center()
        assert area == pytest.approx(60.0 * 30.0, rel=1e-6)
        assert center.x == pytest.approx(30.0, abs=1e-6)
        assert center.y == pytest.approx(0.0, abs=1e-6)
        assert center.z == pytest.approx(15.0, abs=1e-6)
        normal = mount.sdk_face.get_normal_at()
        assert normal.to_tuple() == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)

    def test_multiple_faces_under_one_name_preserve_snapshot_order(self, loaded):
        # name_index entries are ordered deterministically; reader must not reorder
        info = loaded.interfaces["interface.load_pad"]
        assert [f.topo_id for f in info.faces] == [
            e["topo_id"]
            for e in _snapshot_entries(probe_snapshot=loaded, name="interface.load_pad")
        ]


class TestRequireInterfaces:
    def test_all_present_names_resolve(self, loaded):
        resolved = require_interfaces(
            loaded, {"interface.mount_face", "interface.load_pad"})
        assert set(resolved) == {"interface.mount_face", "interface.load_pad"}

    def test_missing_names_fail_loudly_all_at_once(self, loaded):
        with pytest.raises(MissingInterfaceError) as ei:
            require_interfaces(
                loaded,
                {"interface.mount_face", "interface.bolt_flange",
                 "interface.shaft_bore"},
            )
        assert set(ei.value.missing) == {"interface.bolt_flange", "interface.shaft_bore"}
        assert "interface.bolt_flange" in str(ei.value)
        assert "interface.mount_face" not in ei.value.missing


class TestPackageGuards:
    def test_assembly_package_is_rejected_with_a_clear_message(self, monkeypatch, tmp_path):
        from connverify import package_reader as pr

        class _FakeDef:
            definition_kind = "assembly"
            definition_id = "asm"

        class _FakePkg:
            root_definition = _FakeDef()

        monkeypatch.setattr(
            pr, "read_product_package", lambda data: _FakePkg())
        path = tmp_path / "whatever.scadpkg"
        path.touch()
        with pytest.raises(UnsupportedPackageError) as ei:
            pr.load_part(str(path))
        assert "assembly" in str(ei.value)

    def test_not_a_scadpkg_path_is_rejected(self):
        with pytest.raises((UnsupportedPackageError, FileNotFoundError, ValueError)):
            load_part("/nonexistent/thing.scadpkg")


def _snapshot_entries(probe_snapshot: LoadedPart, name: str):
    import json

    return json.loads(probe_snapshot.snapshot_bytes)["name_index"][name]
