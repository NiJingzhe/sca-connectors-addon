"""Tag-review + FEM result rendering.

``render_tag_review`` highlights interface faces for visual verification.
The highlight is drawn from the exact boundary triangles the checker
associated with each ``interface.*`` tag, so the PNG shows precisely what the
pipeline believes. If a tag was attached to the wrong face at modeling time,
its color shows up on the wrong face — one look settles it.

``render_stress_contour`` paints per-node von Mises from a solved .frd onto
the same boundary triangles, with a colorbar and the hotspot node marked —
the visual half of the FEM report.

``render_convergence`` plots the mesh-independence study: quantities of
interest against mesh size (fine to the right), Richardson limit dashed —
the visual proof that the reported numbers are mesh-independent.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

_PALETTE = (
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#66c2a5",
)
_BASE_COLOR = "#cfcfcf"
_BASE_EDGE = "#9a9a9a"


def render_tag_review(mesh, out_path, *, title: str | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    interfaces = sorted(mesh.interface_faces)
    colors = {
        name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(interfaces)
    }

    boundary = _boundary_triangles(mesh.tets)
    panels: List[dict] = []
    n_panels = 1 + len(interfaces)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig = plt.figure(figsize=(4.8 * ncols, 4.4 * nrows))
    fig.suptitle(title or "interface tag review", fontsize=12)

    ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
    counts = _draw_scene(ax, mesh, boundary, colors)
    legend_handles = [
        (colors[name], f"interface.{name}", counts[name])
        for name in interfaces
    ]
    ax.set_title("overview (iso)")
    _apply_box_aspect(ax, mesh)
    panels.append({"interface": None, "view": "iso", "triangles": None,
                   "color": None, "tag": None})

    for k, name in enumerate(interfaces):
        ax = fig.add_subplot(nrows, ncols, 2 + k, projection="3d")
        counts = _draw_scene(ax, mesh, boundary, colors)
        normal = mesh.interface_faces[name][0].normal
        _look_along_normal(ax, normal)
        ax.set_title(f"interface.{name} (face view)", color=colors[name])
        _apply_box_aspect(ax, mesh)
        panels.append({
            "interface": name,
            "tag": f"interface.{name}",
            "view": "normal",
            "triangles": counts[name],
            "color": colors[name],
        })

    legend_labels = [
        f"{tag}  ({count} tris)" for _c, tag, count in legend_handles
    ]
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                   markerfacecolor=color, markeredgecolor="none")
        for color, _tag, _count in legend_handles
    ]
    fig.legend(handles, legend_labels, loc="lower center", ncol=min(3, len(handles)),
               frameon=False)
    fig.tight_layout(rect=(0, 0.06 if handles else 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return {"png": str(out), "panels": panels}


def render_stress_contour(mesh, frd, out_path, *, title: str | None = None) -> dict:
    """von Mises contour in four deterministic views.

    One camera is never enough — a flange hides its own spigot from most
    angles — so the figure is a 2x2 grid: the hotspot-facing view plus
    canonical front/side/top fallbacks, hotspot circled in every panel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    block = frd.block("STRESS")
    von_mises = {node_id: frd.von_mises_at(node_id) for node_id in block.values}
    hotspot_node, max_vm = frd.max_von_mises()

    boundary = _boundary_triangles(mesh.tets)
    cmap = plt.get_cmap("turbo")
    vmin, vmax = 0.0, max_vm if max_vm > 0.0 else 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    polys = [
        tuple(mesh.nodes[n] for n in tri) for tri in boundary
    ]
    facecolors = [
        cmap(norm(sum(von_mises.get(n, 0.0) for n in tri) / 3.0))
        for tri in boundary
    ]

    hotspot_xyz = mesh.nodes.get(hotspot_node) or frd.nodes[hotspot_node]
    views = _contour_views(mesh, hotspot_xyz)

    fig = plt.figure(figsize=(11.0, 9.6))
    fig.suptitle(title or f"von Mises — max {max_vm:.2f} MPa "
                         f"(hotspot node {hotspot_node})", fontsize=12)
    for panel, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, panel + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            polys, facecolors=facecolors, edgecolor="none"))
        ax.scatter([hotspot_xyz[0]], [hotspot_xyz[1]], [hotspot_xyz[2]],
                   color="white", edgecolor="black", linewidths=1.2,
                   s=60, depthshade=False, zorder=10)
        ax.set_title(view_name, fontsize=10)
        _apply_box_aspect(ax, mesh)
        ax.view_init(elev=elev, azim=azim)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    # dedicated axes: a colorbar hung on the 3d subplots lands mid-figure
    # and buries panel titles
    cax = fig.add_axes((0.90, 0.12, 0.022, 0.74))
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label("von Mises (MPa)")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.90, bottom=0.02,
                        wspace=0.04, hspace=0.12)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {
        "png": str(out),
        "max_von_mises_mpa": max_vm,
        "hotspot_node": hotspot_node,
        "triangles": len(polys),
        "views": tuple(name for name, _e, _a in views),
    }


def _contour_views(mesh, hotspot_xyz) -> Tuple[Tuple[str, float, float], ...]:
    """(name, elev, azim) per panel: hotspot-facing + 3 canonical fallbacks."""
    xs = [p[0] for p in mesh.nodes.values()]
    ys = [p[1] for p in mesh.nodes.values()]
    zs = [p[2] for p in mesh.nodes.values()]
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
              (min(zs) + max(zs)) / 2.0)
    dx, dy, dz = (hotspot_xyz[i] - center[i] for i in range(3))
    if math.hypot(dx, dy, dz) < 1e-9:
        hot_azim, hot_elev = -60.0, 22.0       # deterministic default iso
    else:
        hot_azim = math.degrees(math.atan2(dy, dx))
        hot_elev = math.degrees(math.asin(
            max(-1.0, min(1.0, dz / math.hypot(dx, dy, dz)))))
    return (
        ("hotspot", hot_elev, hot_azim),
        ("front (X-Z)", 0.0, 0.0),
        ("side (Y-Z)", 0.0, 90.0),
        ("top (X-Y)", 90.0, -90.0),
    )


def render_convergence(convergence_cases, out_path, *, title: str | None = None,
                       tolerance_pct: float | None = None) -> dict:
    """Mesh-independence study plot: QoI vs mesh size per load case.

    Left panel: max von Mises; right panel: max displacement. The x axis is
    inverted so refinement runs left -> right (the reading direction of a
    convergence curve); the Richardson limit, when reliable, is a dashed
    horizontal line the curve should be flattening toward.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_vm, ax_disp) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    subtitle = title or "mesh independence"
    if tolerance_pct is not None:
        subtitle += f" — tolerance {tolerance_pct:g}% on the finest pair"
    fig.suptitle(subtitle, fontsize=12)

    vm_cases = 0
    for case in convergence_cases:
        sizes = [p.size_mm for p in case.points]
        vm = [p.max_von_mises_mpa for p in case.points]
        disp = [p.max_displacement_mm for p in case.points]
        label = f"{case.name} (ΔQ {case.max_adjacent_delta_pct:.2f}%)"
        if all(v is not None for v in vm):
            ax_vm.plot(sizes, vm, "o-", label=label)
            vm_cases += 1
            if case.richardson_extrapolated_mpa is not None:
                ax_vm.axhline(case.richardson_extrapolated_mpa,
                              linestyle="--", linewidth=1.0, alpha=0.6)
        if all(d is not None for d in disp):
            ax_disp.plot(sizes, disp, "s-", label=case.name)

    for ax, ylabel in ((ax_vm, "max von Mises (MPa)"),
                       (ax_disp, "max displacement (mm)")):
        ax.invert_xaxis()  # fine meshes to the right: the curve flattens ->
        ax.set_xlabel("mesh target size (mm) → finer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    if vm_cases:
        ax_vm.legend(fontsize=9)
    if ax_disp.get_lines():
        ax_disp.legend(fontsize=9)
    else:
        ax_disp.text(0.5, 0.5, "no displacement data",
                     ha="center", va="center", transform=ax_disp.transAxes,
                     color="#888888")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {
        "png": str(out),
        "tolerance_pct": tolerance_pct,
        "cases": [
            {
                "name": c.name,
                "converged": c.converged,
                "max_adjacent_delta_pct": c.max_adjacent_delta_pct,
                "gci_fine_pct": c.gci_fine_pct,
            }
            for c in convergence_cases
        ],
    }


# ------------------------------------------------------------------ helpers

def _draw_scene(ax, mesh, boundary, colors) -> Dict[str, int]:
    """One Poly3DCollection for the whole part, one facecolor per triangle.

    A single collection is essential: matplotlib depth-sorts entire
    collections against each other, so a separate interface collection can
    end up painted UNDER the gray base and vanish (seen on cylindrical
    seats). Per-triangle membership inside one collection sorts correctly.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    membership = {}
    counts = {name: 0 for name in colors}
    for name, faces in mesh.interface_faces.items():
        if name not in colors:
            continue
        for face in faces:
            for tri in face.triangles:
                membership[tuple(sorted(tri))] = colors[name]
                counts[name] += 1
    polys = []
    facecolors = []
    for tri in boundary:
        polys.append(tuple(mesh.nodes[n] for n in tri))
        facecolors.append(membership.get(tri, _BASE_COLOR))
    collection = Poly3DCollection(
        polys, facecolors=facecolors, edgecolor=_BASE_EDGE,
        linewidths=0.15,
    )
    ax.add_collection3d(collection)
    return counts


def _boundary_triangles(tets) -> Tuple[Tuple[int, int, int], ...]:
    """Faces referenced by exactly one tet = the visible outer surface."""
    counts: Dict[Tuple[int, int, int], int] = {}
    for tet in tets:
        n = tet.node_ids
        for face in ((n[0], n[1], n[2]), (n[0], n[1], n[3]),
                     (n[0], n[2], n[3]), (n[1], n[2], n[3])):
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
    boundary = [face for face, count in counts.items() if count == 1]
    boundary.sort()
    return tuple(boundary)


def _apply_box_aspect(ax, mesh) -> None:
    xs = [p[0] for p in mesh.nodes.values()]
    ys = [p[1] for p in mesh.nodes.values()]
    zs = [p[2] for p in mesh.nodes.values()]
    spans = (
        max(xs) - min(xs) or 1.0,
        max(ys) - min(ys) or 1.0,
        max(zs) - min(zs) or 1.0,
    )
    pad = 0.08 * max(spans)
    ax.set_box_aspect(spans)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_zlim(min(zs) - pad, max(zs) + pad)
    ax.set_axis_off()


def _look_along_normal(ax, normal, tilt_deg: float = 30.0) -> None:
    """Face view, tilted off the normal so curved faces are not edge-on.

    Looking exactly down a radial normal compresses a cylindrical seat into
    a line; a fixed 30° tilt toward the global axis least aligned with the
    normal keeps planar faces readable (deterministic camera)."""
    length = math.sqrt(sum(v * v for v in normal)) or 1.0
    n = tuple(v / length for v in normal)
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    raw = min(axes, key=lambda a: abs(sum(x * y for x, y in zip(a, n))))
    dot = sum(x * y for x, y in zip(raw, n))
    tangent = tuple(raw[i] - n[i] * dot for i in range(3))
    t_len = math.sqrt(sum(v * v for v in tangent))
    if t_len < 1e-9:
        direction = n
    else:
        tilt = math.radians(tilt_deg)
        direction = tuple(
            n[i] * math.cos(tilt)
            + (tangent[i] / t_len) * math.sin(tilt)
            for i in range(3)
        )
    dx, dy, dz = direction
    azim = math.degrees(math.atan2(dy, dx))
    elev = math.degrees(math.asin(max(-1.0, min(1.0, dz))))
    ax.view_init(elev=elev, azim=azim)
