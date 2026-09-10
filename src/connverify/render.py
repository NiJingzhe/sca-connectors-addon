"""Tag-review rendering: highlight interface faces for visual verification.

The highlight is drawn from the exact boundary triangles the checker
associated with each ``interface.*`` tag, so the PNG shows precisely what the
pipeline believes. If a tag was attached to the wrong face at modeling time,
its color shows up on the wrong face — one look settles it.

One figure, deterministic layout:
- panel 1: iso overview — whole part in gray, every interface in its color;
- one panel per interface: view looking straight down the face normal.
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
    _draw_part(ax, mesh, boundary)
    legend_handles = []
    for name in interfaces:
        count = _draw_interface(ax, mesh, name, colors[name])
        legend_handles.append((colors[name], f"interface.{name}", count))
    ax.set_title("overview (iso)")
    _apply_box_aspect(ax, mesh)
    panels.append({"interface": None, "view": "iso", "triangles": None,
                   "color": None, "tag": None})

    for k, name in enumerate(interfaces):
        ax = fig.add_subplot(nrows, ncols, 2 + k, projection="3d")
        _draw_part(ax, mesh, boundary)
        count = _draw_interface(ax, mesh, name, colors[name])
        normal = mesh.interface_faces[name][0].normal
        _look_along_normal(ax, normal)
        ax.set_title(f"interface.{name} (face view)", color=colors[name])
        _apply_box_aspect(ax, mesh)
        panels.append({
            "interface": name,
            "tag": f"interface.{name}",
            "view": "normal",
            "triangles": count,
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


# ------------------------------------------------------------------ helpers

def _draw_part(ax, mesh, boundary) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polys = [
        tuple(mesh.nodes[n] for n in tri) for tri in boundary
    ]
    collection = Poly3DCollection(
        polys, facecolor=_BASE_COLOR, edgecolor=_BASE_EDGE,
        linewidths=0.15, alpha=1.0,
    )
    ax.add_collection3d(collection)


def _draw_interface(ax, mesh, name: str, color: str) -> int:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polys = []
    for face in mesh.interface_faces[name]:
        for tri in face.triangles:
            polys.append(tuple(mesh.nodes[n] for n in tri))
    collection = Poly3DCollection(
        polys, facecolor=color, edgecolor="none", alpha=0.96,
    )
    ax.add_collection3d(collection)
    return len(polys)


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


def _look_along_normal(ax, normal) -> None:
    nx, ny, nz = normal
    azim = math.degrees(math.atan2(ny, nx))
    elev = math.degrees(math.asin(max(-1.0, min(1.0, nz))))
    ax.view_init(elev=elev, azim=azim)
