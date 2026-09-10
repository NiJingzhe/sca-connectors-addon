"""Render verification results (deck mesh + .frd fields) as PNG figures.

Usage:
  .venv/bin/python tools/plot_results.py <deck.inp> <result.frd> <report.json> <out_prefix>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from connverify.frd import parse_frd


def read_deck(path: Path):
    nodes: dict[int, tuple[float, float, float]] = {}
    tets: list[tuple[int, int, int, int]] = []
    section = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("*"):
            section = s.upper()
            continue
        if section and section.startswith("*NODE"):
            p = s.split(",")
            nodes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
        elif section and "TYPE=C3D4" in section:
            p = s.split(",")
            tets.append(tuple(int(v) for v in p[1:5]))
    return nodes, tets


def skin_faces(tets):
    faces: dict[tuple[int, ...], int] = {}
    for a, b, c, d in tets:
        for tri in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(tri))
            faces[key] = faces.get(key, 0) + 1
    return [f for f, n in faces.items() if n == 1]


def face_centroid_normal(nodes, tri):
    p = [np.array(nodes[i]) for i in tri]
    centroid = sum(p) / 3.0
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    length = np.linalg.norm(normal)
    return centroid, (normal / length if length > 1e-12 else normal)


def main():
    deck, frd_path, report_path, out_prefix = (Path(sys.argv[i]) for i in range(1, 5))
    report = json.loads(report_path.read_text())
    nodes, tets = read_deck(deck)
    result = parse_frd(frd_path)

    vm = {nid: result.von_mises_at(nid) for nid in result.nodes}
    disp = {nid: result.displacement_at(nid).magnitude_mm for nid in result.nodes}
    faces = skin_faces(tets)

    lc = report["load_cases"][0]
    vm_max = lc["max_von_mises_mpa"]
    sf = lc["safety_factor"]
    sf_req = report["environment"]["safety_factor_required"]
    re = report["environment"]["material"]["yield_strength_mpa"]

    # interface faces from the report (normal + centroid identify them)
    ifaces = {c["interface"]: c for c in report["connections"]}
    def face_on(tri, name):
        iface = ifaces[name]
        n = np.array(iface["normal"])
        c, fn = face_centroid_normal(nodes, tri)
        return abs(float(np.dot(fn, n))) > 0.99 and abs(
            np.dot(c - np.array(iface["centroid_mm"]), n)) < 1e-6

    def panel(ax, title):
        ax.set_title(title, fontsize=11, pad=2)
        ax.set_box_aspect((60, 30, 40))
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]"); ax.set_zlabel("y [mm]")
        # swap y/z so the plate lies flat (y is plate depth, z is height)
        ax.view_init(elev=28, azim=-60)

    def face_xyz(tri):
        pts = [nodes[i] for i in tri]
        return [[p[0], p[2], p[1]] for p in pts]  # (x, z, y)

    fig = plt.figure(figsize=(15, 12), dpi=110)
    fig.suptitle(
        f"connverify — {report['environment']['name']} · verdict: "
        f"{report['verdict'].upper()} · SF {sf:.1f} ≥ {sf_req} required",
        fontsize=14, fontweight="bold")

    # -- A: mesh + interfaces ------------------------------------------------
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    panel(ax, f"mesh: {len(nodes)} nodes / {len(tets)} C3D4 tets")
    mount = [f for f in faces if face_on(f, "mount_face")]
    pad = [f for f in faces if face_on(f, "load_pad")]
    rest = [f for f in faces if f not in mount and f not in pad]
    ax.add_collection3d(Poly3DCollection(
        [face_xyz(f) for f in rest], facecolor="0.85", edgecolor="0.4",
        linewidths=0.15, alpha=0.35))
    ax.add_collection3d(Poly3DCollection(
        [face_xyz(f) for f in mount], facecolor="#d62728", edgecolor="k",
        linewidths=0.3, alpha=0.9))
    ax.add_collection3d(Poly3DCollection(
        [face_xyz(f) for f in pad], facecolor="#1f77b4", edgecolor="k",
        linewidths=0.3, alpha=0.9))
    # load arrow: 1 kN pressing onto load_pad (plotted axes are x, z_world, y_world)
    c = np.array(ifaces["load_pad"]["centroid_mm"])
    ax.quiver(c[0], c[2] + 16, c[1], 0, -12, 0, arrow_length_ratio=0.2,
              color="#1f77b4", linewidth=2.5)
    ax.text(c[0], c[2] + 19, c[1], "F = 1000 N", color="#1f77b4", fontsize=10,
            ha="center")
    ax.legend(handles=[
        Line2D([], [], color="#d62728", lw=4, label="mount_face (bolted, 1800 mm²)"),
        Line2D([], [], color="#1f77b4", lw=4, label="load_pad (contact, 2400 mm²)"),
    ], loc="upper left", fontsize=8)

    # -- B: von Mises on skin ------------------------------------------------
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    panel(ax, f"von Mises [MPa] · max {vm_max:.2f} at mount face top edge")
    norm = Normalize(0, vm_max)
    cmap = plt.cm.turbo
    coll = Poly3DCollection(
        [face_xyz(f) for f in faces],
        facecolors=[cmap(norm(np.mean([vm[i] for i in f]))) for f in faces],
        edgecolor="none")
    ax.add_collection3d(coll)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.6, pad=0.08)
    cbar.set_label("von Mises [MPa]")
    for hs in lc["hotspots"][:2]:
        x, y, z = hs["location_mm"]
        ax.scatter([x], [z], [y], color="k", s=28, marker="x", depthshade=False)

    # -- C: displacement, deformed (scaled) ----------------------------------
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    dmax = max(disp.values())
    panel(ax, f"displacement [mm] · max {dmax:.4f} (deformation ×{2.0 / max(dmax, 1e-12) / 1000:.0f}k)")
    scale = 2.0 / dmax if dmax > 0 else 0.0  # exaggerate to ~2 mm
    dn = {nid: (
        nodes[nid][0] + result.displacement_at(nid).dx_mm * scale,
        nodes[nid][1] + result.displacement_at(nid).dy_mm * scale,
        nodes[nid][2] + result.displacement_at(nid).dz_mm * scale,
    ) for nid in nodes}
    norm = Normalize(0, dmax)
    coll = Poly3DCollection(
        [[(dn[i][0], dn[i][2], dn[i][1]) for i in f] for f in faces],
        facecolors=[cmap(norm(np.mean([disp[i] for i in f]))) for f in faces],
        edgecolor="none")
    ax.add_collection3d(coll)
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.6, pad=0.08)
    cbar.set_label("|u| [mm]")

    # -- D: hotspot attribution ----------------------------------------------
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    panel(ax, "hotspot attribution → feature node_00000001 (box feature)")
    coll = Poly3DCollection(
        [face_xyz(f) for f in faces], facecolor="0.9", edgecolor="0.5",
        linewidths=0.1, alpha=0.3)
    ax.add_collection3d(coll)
    for k, hs in enumerate(lc["hotspots"], 1):
        x, y, z = hs["location_mm"]
        ax.scatter([x], [z], [y], color="#d62728", s=60, depthshade=False)
        ax.text(x + 3, z + 3, y + 4,
                f"#{k}  {hs['von_mises_mpa']:.2f} MPa\n({x:.0f}, {y:.0f}, {z:.0f})",
                color="#b2182b", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          pad=1.5))

    for ax_ in fig.axes:
        if hasattr(ax_, "set_zlim"):
            ax_.set_xlim(0, 60); ax_.set_ylim(0, 40); ax_.set_zlim(0, 40)

    out = out_prefix.with_suffix(".png")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
