"""Exploration: build tagged part -> capture -> read back -> face query paths."""
import json

import simplecadapi as scad
from simplecadapi import capture, read_product_package
from simplecadapi.artifacts.assembly_io import materialize_definition


@scad.part(id="probe_box")
def build_probe_box() -> scad.Part:
    solid = scad.make_box_rsolid(
        width=60.0, height=40.0, depth=30.0,
        bottom_face_center=(30.0, 20.0, 0.0),
        left_face_tag="interface.mount_face",
        top_face_tag="interface.load_pad",
    )
    return scad.make_part_rpart(part_id="probe_box", body=solid, name="probe_box")


result = build_probe_box()
capture(result, "/tmp/probe_box.scadpkg", include_scene=False)
print("captured ok")

package = read_product_package(data="/tmp/probe_box.scadpkg")
root = package.root_definition
print("root kind:", root.definition_kind, "id:", root.definition_id)

snap = json.loads(root.blobs[root.topology_snapshot_ref.path])
print("snapshot keys:", list(snap.keys()))
print("name_index:", json.dumps(snap["name_index"], indent=1)[:600])
ents = snap["entities"]
print("entity keys:", list(ents[0].keys()))
for e in ents:
    tags = [b.get("tag") if isinstance(b, dict) else b for b in (e.get("tag_bindings") or [])]
    tags = [t for t in tags if t]
    if any(str(t).startswith("interface.") for t in tags):
        print("IFACE entity:", e.get("topo_id"), e.get("kind"), tags,
              "prov:", e.get("feature_output"))

part_rt = materialize_definition(root)
body = part_rt.body
faces = scad.ql.faces().resolve(body)
print("face count via ql:", len(faces))
for f in faces:
    attrs = [a for a in dir(f) if not a.startswith("_")]
    print("face attrs sample:", attrs[:25])
    break
for f in faces:
    try:
        tags = list(f._list_tags())
    except Exception as exc:
        tags = f"ERR {exc}"
    print("face topo_id=", getattr(f, "topo_id", None), "tags=", tags)
