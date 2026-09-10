"""Read a `.scadpkg` product package into the verification pipeline.

In-process SDK path (Path A): the package is fully validated by
``read_product_package``; the part solid is materialized with the tag channel
restored, so interface faces resolve by ``topo_id`` — never by geometry
guessing. A required interface name that is absent fails loudly, naming every
missing name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Mapping, Optional

import simplecadapi as scad
from simplecadapi import read_product_package
from simplecadapi.artifacts.assembly_io import materialize_definition


class UnsupportedPackageError(ValueError):
    """The package cannot enter the connector-verification pipeline."""


class MissingInterfaceError(KeyError):
    """Required ``interface.*`` names are absent from the package's tag channel."""

    def __init__(self, missing: Iterable[str]):
        self.missing = sorted(missing)
        names = ", ".join(self.missing)
        super().__init__(
            f"package is missing interface names: {names}. "
            "Model authors create them with apply_tag(shape=..., "
            "tag='interface.<name>'); they are never guessed by geometry."
        )


@dataclass(frozen=True)
class InterfaceFace:
    """One face of a named interface, with snapshot identity and provenance."""

    topo_id: str
    geometry_hash: str
    feature_node_id: str
    sdk_face: object  # live SDK Face (get_area / get_center / get_normal_at / wrapped)


@dataclass(frozen=True)
class InterfaceInfo:
    name: str
    faces: tuple


@dataclass(frozen=True)
class LoadedPart:
    package_path: str
    definition_kind: str
    definition_id: str
    body: object                    # SDK Solid with tags restored
    interfaces: Dict[str, InterfaceInfo]
    snapshot_bytes: bytes
    metadata: Mapping = None
    face_provenance: Mapping = None  # topo_id -> {graph_id, node_id, output_slot}

    @property
    def available_interface_names(self) -> FrozenSet[str]:
        return frozenset(self.interfaces)


def load_part(path: str) -> LoadedPart:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"package not found: {p}")
    if p.suffix != ".scadpkg":
        raise UnsupportedPackageError(
            f"expected a .scadpkg product package, got {p.name!r}"
        )
    package = read_product_package(data=str(p))
    root = package.root_definition
    if root.definition_kind != "single_solid":
        raise UnsupportedPackageError(
            f"connector verification consumes single-part packages; "
            f"root definition {root.definition_id!r} is {root.definition_kind!r}. "
            "Capture each connector part on its own."
        )

    snapshot_bytes = root.blobs[root.topology_snapshot_ref.path]
    snapshot = json.loads(snapshot_bytes)
    part = materialize_definition(root)
    body = part.body

    faces_by_topo = {f.topo_id: f for f in scad.ql.faces().resolve(body)}
    entities_by_topo = {str(e["topo_id"]): e for e in snapshot["entities"]}

    interfaces: Dict[str, InterfaceInfo] = {}
    for name, entries in snapshot["name_index"].items():
        if not name.startswith("interface."):
            continue
        non_face = [e for e in entries if e.get("kind") != "face"]
        if non_face:
            raise UnsupportedPackageError(
                f"interface name {name!r} maps to non-face entities "
                f"({[e.get('kind') for e in non_face]}); only FACE interfaces "
                "are supported"
            )
        faces = []
        for entry in entries:
            topo_id = str(entry["topo_id"])
            sdk_face = faces_by_topo.get(topo_id)
            if sdk_face is None:
                raise UnsupportedPackageError(
                    f"snapshot entity {topo_id} (interface {name!r}) did not "
                    "resolve onto the materialized solid — package is inconsistent"
                )
            entity = entities_by_topo.get(topo_id, {})
            provenance = entity.get("feature_output") or {}
            faces.append(InterfaceFace(
                topo_id=topo_id,
                geometry_hash=str(entry.get("geometry_hash", "")),
                feature_node_id=str(provenance.get("node_id", "")),
                sdk_face=sdk_face,
            ))
        interfaces[name] = InterfaceInfo(name=name, faces=tuple(faces))

    provenance = {}
    for entity in snapshot["entities"]:
        if entity.get("kind") != "face":
            continue
        output = entity.get("feature_output") or {}
        provenance[str(entity["topo_id"])] = {
            "graph_id": str(output.get("graph_id", "")),
            "node_id": str(output.get("node_id", "")),
            "output_slot": int(output.get("output_slot", 0)),
        }

    return LoadedPart(
        package_path=str(p),
        definition_kind=root.definition_kind,
        definition_id=root.definition_id,
        body=body,
        interfaces=interfaces,
        snapshot_bytes=snapshot_bytes,
        metadata=getattr(root, "metadata", {}) or {},
        face_provenance=provenance,
    )


def require_interfaces(
    loaded: LoadedPart, required: Iterable[str]
) -> Dict[str, InterfaceInfo]:
    """Resolve required interface names or fail loudly with ALL missing names."""
    required = set(required)
    missing = required - set(loaded.interfaces)
    if missing:
        raise MissingInterfaceError(missing)
    return {name: loaded.interfaces[name] for name in sorted(required)}
