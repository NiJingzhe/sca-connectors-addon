"""Parse CalculiX ``.frd`` result files written by FEMaster.

Layout (verified against a real v2.8.0 output):

- node block after the ``2C`` header: `` -1 <id(10)> <x(12)> <y(12)> <z(12)>``
- element block after ``3C`` (skipped)
- per result block: ``1PSTEP`` header, `` -4  <NAME> <ncomp>``, `` -5`` component
  lines, then `` -1`` data lines (6 values) with `` -2`` continuation lines,
  closed by `` -3``
- fixed columns: record key [0:3], node id [3:13], then 12-char value fields
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


class FrdParseError(ValueError):
    """The .frd payload is malformed or lacks a required block."""


@dataclass(frozen=True)
class FrdBlock:
    name: str
    components: Tuple[str, ...]
    values: Dict[int, Tuple[float, ...]]

    def component_index(self, component: str) -> int:
        try:
            return self.components.index(component)
        except ValueError:
            raise FrdParseError(
                f"block {self.name!r} has no component {component!r} "
                f"(components: {', '.join(self.components)})"
            ) from None


@dataclass(frozen=True)
class Stress:
    sxx_mpa: float
    syy_mpa: float
    szz_mpa: float
    syz_mpa: float
    szx_mpa: float
    sxy_mpa: float

    @property
    def von_mises_mpa(self) -> float:
        d_xy = self.sxx_mpa - self.syy_mpa
        d_yz = self.syy_mpa - self.szz_mpa
        d_zx = self.szz_mpa - self.sxx_mpa
        return math.sqrt(
            0.5 * (d_xy * d_xy + d_yz * d_yz + d_zx * d_zx)
            + 3.0 * (self.syz_mpa ** 2 + self.szx_mpa ** 2 + self.sxy_mpa ** 2)
        )


@dataclass(frozen=True)
class Displacement:
    dx_mm: float
    dy_mm: float
    dz_mm: float

    @property
    def magnitude_mm(self) -> float:
        return math.sqrt(self.dx_mm ** 2 + self.dy_mm ** 2 + self.dz_mm ** 2)


@dataclass(frozen=True)
class FrdResult:
    nodes: Dict[int, Tuple[float, float, float]]
    blocks: Dict[str, FrdBlock]

    def block(self, name: str) -> FrdBlock:
        try:
            return self.blocks[name]
        except KeyError:
            raise FrdParseError(
                f"result file has no {name!r} block; available: "
                f"{', '.join(sorted(self.blocks)) or '(none)'}"
            ) from None

    def stress_at(self, node_id: int) -> Stress:
        block = self.block("STRESS")
        if node_id not in block.values:
            raise FrdParseError(f"STRESS block has no values for node {node_id}")
        v = block.values[node_id]
        return Stress(
            sxx_mpa=v[block.component_index("SXX")],
            syy_mpa=v[block.component_index("SYY")],
            szz_mpa=v[block.component_index("SZZ")],
            syz_mpa=v[block.component_index("SYZ")],
            szx_mpa=v[block.component_index("SZX")],
            sxy_mpa=v[block.component_index("SXY")],
        )

    def von_mises_at(self, node_id: int) -> float:
        return self.stress_at(node_id).von_mises_mpa

    def max_von_mises(self) -> Tuple[int, float]:
        block = self.block("STRESS")
        best_node, best_value = None, -1.0
        for node_id in block.values:
            value = self.von_mises_at(node_id)
            if value > best_value:
                best_node, best_value = node_id, value
        if best_node is None:
            raise FrdParseError("STRESS block is empty")
        return best_node, best_value

    def displacement_at(self, node_id: int) -> Displacement:
        block = self.block("DISP")
        if node_id not in block.values:
            raise FrdParseError(f"DISP block has no values for node {node_id}")
        v = block.values[node_id]
        return Displacement(
            dx_mm=v[block.component_index("D1")],
            dy_mm=v[block.component_index("D2")],
            dz_mm=v[block.component_index("D3")],
        )


def parse_frd(source) -> FrdResult:
    if isinstance(source, (str, Path)) and Path(source).suffix == ".frd" and Path(source).is_file():
        text = Path(source).read_text()
    elif isinstance(source, (str, Path)):
        text = str(source)
    else:
        text = source.decode() if isinstance(source, bytes) else str(source)

    nodes: Dict[int, Tuple[float, float, float]] = {}
    blocks: Dict[str, FrdBlock] = {}

    state = "preamble"  # preamble | nodes | elements | block
    current_name = None
    current_components: list[str] = []
    current_values: Dict[int, list[float]] = {}
    pending_node: int | None = None

    def _finalize_block() -> None:
        nonlocal current_name, current_components, current_values, pending_node
        if current_name is None:
            return
        ncomp = len(current_components)
        finalized = {}
        for node_id, vals in current_values.items():
            if len(vals) != ncomp:
                raise FrdParseError(
                    f"block {current_name!r}: node {node_id} carries {len(vals)} "
                    f"values but {ncomp} components are declared"
                )
            finalized[node_id] = tuple(vals)
        blocks[current_name] = FrdBlock(
            name=current_name,
            components=tuple(current_components),
            values=finalized,
        )
        current_name, current_components, current_values, pending_node = (
            None, [], {}, None)

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        key = line[:3]

        if line.startswith("    2C") or line.startswith("    2U"):
            state = "nodes"
            continue
        if line.startswith("    3C"):
            state = "elements"
            continue
        if line.startswith("    1PSTEP") or line.startswith("  100CL"):
            state = "block"
            continue
        if line.strip() == "9999":
            break

        if key == " -4" and state == "block":
            _finalize_block()
            tokens = line[3:].split()
            if len(tokens) < 2:
                raise FrdParseError(f"line {lineno}: malformed -4 header")
            current_name = tokens[0]
            state = "block-headers"
            continue
        if key == " -5" and state == "block-headers":
            token = line[3:].split()
            if not token:
                raise FrdParseError(f"line {lineno}: malformed -5 component")
            current_components.append(token[0])
            continue
        if key == " -3":
            if state == "block-headers":
                _finalize_block()
                state = "block"
            else:
                state = "preamble" if state == "nodes" or state == "elements" else state
            continue

        if key == " -1" or key == " -2":
            if state == "nodes" and key == " -1":
                node_id, values = _split_data_line(key, line, lineno)
                if len(values) != 3:
                    raise FrdParseError(
                        f"line {lineno}: node record needs 3 coordinates")
                nodes[node_id] = (values[0], values[1], values[2])
            elif state == "block-headers" and current_name is not None:
                node_id, values = _split_data_line(key, line, lineno)
                if key == " -1":
                    pending_node = node_id
                    current_values[node_id] = list(values)
                else:
                    if pending_node is None:
                        raise FrdParseError(
                            f"line {lineno}: continuation -2 without a -1 data line")
                    current_values[pending_node].extend(values)
            # element-block lines and stray records are skipped

    _finalize_block()

    if not nodes:
        raise FrdParseError("no node block found — not a valid .frd result")
    return FrdResult(nodes=nodes, blocks=blocks)


def _split_data_line(key: str, line: str, lineno: int) -> Tuple[int, list[float]]:
    node_id: int | None = None
    if key == " -1":
        try:
            node_id = int(line[3:13].strip())
        except ValueError:
            raise FrdParseError(f"line {lineno}: cannot read node id from {line[:20]!r}")
    values: list[float] = []
    rest = line[13:]
    for i in range(0, len(rest), 12):
        chunk = rest[i:i + 12].strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            raise FrdParseError(
                f"line {lineno}: bad value field {chunk!r}")
    return node_id, values
