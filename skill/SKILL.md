---
name: sca-connectors-addon
description: Verify whether a mechanical connection end face actually assembles, holds load, and stays inside its keep-out envelope — for static connector parts (brackets, flanges, mounts, plates, shaft seats) captured as SimpleCADAPI .scadpkg packages. Use whenever a part mates with a counterpart through any of these joint kinds — bolted through-holes, screws/bolts into tapped holes, studs, rivets, dowel/parallel pins, keys or splines, press (interference) or transition ISO 286 fits, fillet or butt welds, adhesives, clamped pads, bearing seats (H7), plain contact pads, snap fits (螺栓/螺钉/螺柱/铆/销/键/花键/过盈·过渡配合/焊接/粘接/夹持/轴承座/接触面/卡扣). Consumes a single-part .scadpkg whose mating faces carry interface.* tags, plus a VerificationEnv declaring per-interface JointSpec and counterpart geometry, materials, load cases, keep-out boxes, and an optional MeshStudy. Deterministically checks hole pitch/edge distances (EN 1993-1-8), clearance-hole diameters, through/blind requirement, H7 bore limits, fillet-leg minimums (AWS D1.1), wrench/nut access envelopes, and counterpart assemblability — hole-pattern match, mating-plane interference, clamp-land support, ISO 286 interference range, hub insertion path, snap hook/slot alignment; then gates the gmsh tet mesh on element quality (signed volume / aspect-ratio hard limits, SICN cross-check — inverted meshes are never solved), solves linear-static FEM via FEMaster with safety factors and von Mises hotspots attributed to the owning interface and feature-graph node, and optionally proves mesh independence via a three-mesh h-refinement study (finest-pair ΔQ criterion, observed order, Richardson limit, GCI). Emits report.json + report.md (with mesh-quality, mesh-independence and assumptions sections), a color-highlighted tag_review.png, a von Mises stress_<CASE>.png contour per solved load case, and a convergence.png study plot. Never modifies geometry; every finding points at the interface, feature, or coordinates to change.
license: Apache-2.0
metadata:
  project: sca-connectors-addon
  version: 0.7.0
---

# sca-connectors-addon — static connector verification (python package: connverify)

Turn a natural-language verification request into a **formal, deterministic
verification environment**, run it, and read a **geometry-directed** report.

**Verifiable joint kinds** (the same list routes this skill — see the
frontmatter description): bolted through · bolted into tapped holes · stud ·
riveted · pinned (dowel/parallel) · keyed · splined · interference fit ·
transition fit · fillet weld · butt weld · adhesive · clamped pad ·
bearing seat (H7) · contact pad · snap fit.

```
natural language ──agent──▶ VerificationEnv (Python DSL, validated, JSON)
                                  │
.scadpkg (SimpleCADAPI part) ─────┤
                                  ▼
        interface checks · envelope checks · gmsh mesh (exact face association)
                                  ▼
        FEMaster deck per load case ── solve ── .frd ── hotspot attribution
                                  ▼
        report.json + report.md  (verdict, numbers, WHERE to change geometry)
```

## Pre-check the runtime (once, before first use)

The registry state is a cached install-time probe — ask for the current
truth:

```bash
sca addon check sca-connectors-addon   # re-probe now, refresh registry
```

Run everything through the addon's own environment with `sca addon use`
(the descriptor's `command_prefix` pins the runtime venv in
`{runtime_dir}`, which `sca addon update` never touches; `SCA_ADDON_DIR`
and `SCA_RUNTIME_DIR` are exported). If the runtime is not provisioned
yet:

```bash
sca addon use sca-connectors-addon \
    sh "$SCA_ADDON_DIR/tools/bootstrap_env.sh" /path/to/SimpleCADAPI
```

Solver binary — any of:

```bash
export CONNVERIFY_FEMASTER=/path/to/FEMaster   # or femaster on PATH,
                                               # or $SCA_ADDON_DIR/vendor/FEMaster
```

On failure, name the missing piece and stop — never silently skip the
analysis.

## Consumption contract

- Input geometry: a **single-part** `.scadpkg` (units mm). Assemblies are
  rejected — capture each connector part on its own.
- Connection end faces must be tagged at modeling time:
  `apply_tag(shape=..., tag="interface.<name>")`. An environment referencing a
  name the package does not carry fails loudly listing **all** missing names;
  faces are never guessed by geometry.
- Units are fixed: **mm, N, N·mm, MPa** (density t/mm³).
- This addon **never modifies geometry**. When the report says the geometry
  must change, return to the SimpleCADAPI modeling workflow, repair the
  owning feature/parameter, re-`capture`, and re-verify.

## Workflow

### 1. Formalize the request (natural language → DSL)

Extract from the user's words, making every number explicit:

| User says | DSL object |
| --- | --- |
| "bolted to the wall with 4× M10" | `Interface(spec=BoltedThroughSpec(nominal_diameter_mm=10.0, expected_count=4))` |
| "screwed into the base", "tapped holes" | `Interface(spec=BoltedTappedSpec(nominal_diameter_mm=8.0))` |
| "welded all around", "6 mm fillet weld" | `Interface(spec=WeldedFilletSpec(design_leg_mm=6.0))` |
| "press-fit onto the shaft" | `Interface(spec=InterferenceSpec(nominal_diameter_mm=20.0, fit="H7/r6"))` |
| "located by two dowel pins" | `Interface(spec=PinnedSpec(pin_diameter_mm=6.0))` |
| "the shaft presses here", mating surface | `Interface(spec=ContactPadSpec())` or `BearingSeatSpec(bore_diameter_mm=...)` |
| forces / weights | `ForceLoad(target=<iface>, fx_n=..., fy_n=..., fz_n=...)` or `ForceLoad(point_mm=(x, y, z), ...)` |
| pressure | `PressureLoad(interface=<iface>, magnitude_mpa=...)` (positive pushes onto the face) |
| "must not stick into X" | `KeepOutBox(name=..., min_corner_mm=..., max_corner_mm=...)` |
| material, required safety factor | `Material(...)`, `safety_factor_required=` |

The **joint type is the definition of the connection end face** — it selects
which mechanical rules actually run (see the check matrix below). A bare
`Interface(method=ConnectionMethod.BOLDED)` is accepted for quick FEM-only
runs but performs no joint-geometry verification.

**Connection-type taxonomy** (`connverify.joint_types`):

| Permanence | JointKind | Spec class |
| --- | --- | --- |
| detachable 可拆卸 | `BOLTED_THROUGH` `BOLDED_TAPPED` `STUD` | `BoltedThroughSpec` `BoltedTappedSpec` `StudSpec` |
| detachable | `PINNED` `KEYED` `SPLINED` | `PinnedSpec` `KeyedSpec` `SplinedSpec` |
| semi-permanent | `INTERFERENCE` `TRANSITION` | `InterferenceSpec` `TransitionSpec` (ISO 286) |
| permanent 不可拆 | `WELDED_FILLET` `WELDED_BUTT` `RIVETED` `ADHESIVE` | `WeldedFilletSpec` `WeldedButtSpec` `RivetedSpec` `AdhesiveSpec` |
| load-entry | `CLAMPED` `CONTACT_PAD` `BEARING_SEAT` | `ClampedSpec` `ContactPadSpec` `BearingSeatSpec` |

**Check matrix by joint type** (rules cite their standards in every message):

| Kind | Geometric checks run against the BREP |
| --- | --- |
| bolted / stud / riveted | holes detected exactly (face inner circular wires); pitch ≥ 2.5d, edge ≥ 1.5d (EN 1993-1-8 / BS 5950); hole ⌀ vs d0 ± 0.75 mm (EN 1090-2/ISO 273); **through-hole required** (clear shank axis); wrench + nut envelopes clear (ISO 4014 head geometry, OCC boolean); optional expected_count |
| bolted into tapped holes | same layout rules; **blind hole required**; head-side tool access |
| pinned / interference / transition / bearing seat | bore ⌀ vs ISO 286 **H7 limits** (embedded IT7 table); layout limits where applicable |
| fillet-welded | mating-face planarity; local plate thickness (ray along inward normal); design leg ≥ max(AWS D1.1 T7.7 floor, 1.5·√t) — advisory if no leg declared |
| keyed / splined | load-entry semantics; keyway profile checks arrive in v2 (stated in the report) |
| contact / clamped / adhesive | planarity (+ optional min bearing area) |

**Counterpart verification — the joint spec IS the mating face.** To prove
the end face can actually *assemble*, declare the mating part's nominal
geometry in the interface's face-local frame; connverify GENERATES the
counterpart solid from the spec and checks computationally (no mating model
is ever built):

```python
Interface(
    name="mount_face",
    spec=BoltedThroughSpec(nominal_diameter_mm=10.0),
    counterpart=PlateCounterpart(           # the wall flange it bolts onto
        thickness_mm=8.0,
        holes=((-15.0, -10.0, 11.0), (-15.0, 10.0, 11.0), (20.0, 0.0, 11.0)),
        window=((-40.0, -20.0), (40.0, 20.0)),
        fastener_length_mm=30.0),
)
```

Local frame contract (deterministic): origin = face bounding-box center,
n = outward normal, u = global axis least aligned with n projected onto the
plane, v = n × u. Per-family checks, mapped for **every** JointKind
(`COUNTERPART_FOR_KIND`):

| Counterpart family | Kinds | Assemblability primitives |
| --- | --- | --- |
| `PlateCounterpart` | bolted-through/tapped, stud, riveted | hole-pattern match vs declared pattern (per-hole deviation + tolerance); mating-plate **interference 穿模** (OCC boolean against the generated plate); **clamp land 压紧** (washer annulus probes, ISO 7089-style ⌀2d); fastener **stack length** vs grip + nut |
| `BoreCounterpart` | pin, interference, transition, keyed, splined, bearing seat | nominal-⌀ match; **ISO 286 fit-band arithmetic** (embedded H7 + shaft classes, interference range in µm); hub **insertion path** clearance (annulus sweep along the seat axis) |
| `PlaneCounterpart` | fillet/butt weld, adhesive, clamped, contact pad | mating-plane interference; contact coverage over the declared window |
| `SnapCounterpart` | snap fit | hook (卡) presence + height at the declared catch position; slot (扣) region must stay clear above the mating plane |

Counterpart family must match the joint kind (mismatch is rejected). Keyed/
spline profiles and snap deflection physics are v2 — the report says so.

```python
from connverify.env import (ForceLoad, KeepOutBox, LoadCase, Material,
    VerificationEnv)
from connverify.joint_types import BoltedThroughSpec, ContactPadSpec
from connverify.env import Interface

env = VerificationEnv(
    name="motor bracket static",
    part_package="out/bracket.scadpkg",
    material=Material(name="steel_s355", youngs_modulus_mpa=210000.0,
                      poisson_ratio=0.3, yield_strength_mpa=355.0,
                      density_t_per_mm3=7.85e-9),
    interfaces=[
        Interface(name="mount_face",
                  spec=BoltedThroughSpec(nominal_diameter_mm=10.0,
                                         bolt_grade="8.8", expected_count=4)),
        Interface(name="load_pad", spec=ContactPadSpec()),
    ],
    load_cases=[
        LoadCase(name="operational", loads=[
            ForceLoad(target="load_pad", fz_n=-800.0),
        ]),
        LoadCase(name="shipping", loads=[
            ForceLoad(target="load_pad", fz_n=-400.0, fx_n=300.0),
        ]),
    ],
    envelopes=[KeepOutBox(name="motor_body", min_corner_mm=(0, -10, 40),
                          max_corner_mm=(80, 90, 120))],
    safety_factor_required=2.0,
)
env.validate()                    # raises with ALL violations at once
```

Validation rules you will hit most: interface names are lowercase slugs; at
least one FIXED/BOLTED/WELDED interface (else the part floats); loads need a
target interface or a point; a `.scadpkg` part package path.

`env.to_json()` / `VerificationEnv.from_json()` is the durable, replayable
form (schema 1.0).

### 2. Run the verification (one call)

```python
from connverify.pipeline import verify

report = verify(env, out_dir="out/verify", mesh_size_mm=6.0)
print(report.to_markdown())          # agent-readable
open("out/verify/report.json").read()  # machine contract
```

Artifacts: `decks/<CASE>.inp` (+ `.frd`/`.res`), `stress_<CASE>.png`
(von Mises contour, hotspot circled), `report.json`, `report.md`, and
`tag_review.png`.

To prove the numbers are **mesh-independent** (required before trusting a
peak stress), declare a study instead of a single size — the environment
then owns the mesh family and the run solves every case on every mesh:

```python
env = VerificationEnv(
    ...,
    mesh_study=MeshStudy(sizes_mm=(9.0, 6.0, 4.0),  # geometric, coarse→fine
                         qoi_tolerance_pct=5.0),
)
report = verify(env, out_dir="out/verify")   # no mesh_size_mm — study owns it
```

Artifacts additionally: `decks/h<SIZE>/<CASE>.inp` per size and
`convergence.png` (QoI vs mesh size, Richardson limit dashed). Pick sizes
with a **constant ratio** (e.g. r = 1.5 → 9 / 6 / 4) — only then are
observed order, Richardson limit and GCI computed; and pick sizes small
enough to actually refine the part (verify node counts grow in
`analysis_quality.convergence.cases[].points`).

### 2b. Visually confirm the tags (mandatory before trusting results)

**Look at `tag_review.png` before acting on any report.** The image shows
the whole part in gray and each `interface.*` tag highlighted in its own
color — one iso overview plus one straight-on face view per interface. The
highlights are drawn from the exact triangles the checker associated, so a
tag attached to the WRONG face at modeling time appears as a highlight on
the wrong face, and a lost tag appears as a missing panel.

Judge: does each color sit on the end face the user described? If not, the
geometry's tag is wrong — repair it in the modeling source (re-tag the final
geometry, e.g. `apply_tag_rselection`) and re-verify. Never "fix" the
environment to match a wrong tag.

**`stress_<CASE>.png` is the visual half of the FEM report**: per-node von
Mises painted on the mesh surface with a colorbar, camera facing the
hotspot node (white circle). Confirm the hot region sits where the report
says and where the load path predicts.

Standalone use:

```python
from connverify.meshing import mesh_part
from connverify.render import render_stress_contour, render_tag_review
from connverify.frd import parse_frd

mesh = mesh_part(loaded, mesh_size_mm=8.0)
render_tag_review(mesh, "out/tag_review.png", title="bracket")
render_stress_contour(mesh, parse_frd("out/verify/decks/CASE.frd"),
                      "out/stress_CASE.png")
```

### 3. Read the report like an engineer

- `verdict` — pass only if interfaces, envelopes, every load case, the
  mesh-quality gate, AND (when declared) the mesh-independence study pass.
- Per load case: `max_von_mises_mpa`, `safety_factor = Re / max_vM`,
  `max_displacement_mm`, top-3 hotspots each with `location_mm`,
  `on_interface` (preferentially the constrained one), and
  `owning_feature` — the feature-graph node that **produced the hotspot's
  BREP face** (from the package's topology snapshot; `graph_id` + `node_id`
  map straight back into the modeling feature tree).
- `analysis_quality` — mesh element type/size, full quality stats, the
  convergence dossier (points, ΔQ series, observed order, Richardson
  limit, GCI), and the assumption list. The report also renders
  `## Mesh quality`, `## Mesh independence` and
  `## Analysis quality & assumptions` sections; a single-mesh run states
  plainly that independence was **not verified**.
- `feedback` — ordered, actionable items with numbers and locations.

### 4. Repair loop (geometry changes live in the modeling workflow)

1. Report names interface 'mount_face', feature `node_00000012`, location.
2. Open the part's modeling source; change the owning feature/parameter
   (fillet radius, plate thickness, gusset) — not the verification.
3. Re-`capture` the `.scadpkg`; re-run `verify` with the SAME env JSON.
4. Compare `max_von_mises_mpa` / `safety_factor` across runs — the
   environment is deterministic, so deltas are real geometry effects.
5. After re-tagging or boolean-heavy edits, re-check `tag_review.png` —
   kwarg face tags do not reliably survive multi-tool booleans.

## Mesh quality & mesh independence (built-in evidence)

Two deterministic gates protect every FEM number this addon reports.

**Quality gate (always on, before any solve).** Every tetrahedron is
checked geometrically: signed volume (inverted/zero-volume elements are a
hard FAIL — the mesh is never solved), normalized radius-ratio aspect
ratio (1 = regular tetrahedron; max > 10 is a hard FAIL, > 5 on more than
5% of elements is a warning), plus gmsh's minSICN as a cross-check (< 0.05
warns). Failures name the worst element's coordinates so the geometry or
target size can be repaired. The full statistics land in
`analysis_quality.mesh.quality` and the `## Mesh quality` report section.

**Mesh-independence study (opt-in via `MeshStudy`).** Declares an
h-refinement family and a tolerance; the pipeline meshes + solves every
case on every size, then judges the quantities of interest
(max von Mises, max displacement):

- **Pass criterion**: the finest-pair relative change ΔQ of both QoIs
  ≤ `qoi_tolerance_pct` (the tolerance should come from the decision the
  report feeds, typically 1–5%).
- With three geometrically graded meshes: **observed order** p_obs,
  **Richardson extrapolated limit**, and **GCI** (Fs = 1.25) — the
  estimated remaining discretization uncertainty of the finest mesh.
- An unconverged study FAILS the verdict, and a monotonically *rising*
  peak is flagged as the **singularity signature** (sharp re-entrant
  edge / point load / constrained boundary): the fix is a fillet or a
  path-averaged criterion, not a finer mesh. Oscillatory or
  non-monotone differences are reported as such — never averaged into a
  false plateau.
- The single-mesh default honestly reports `performed: false`; peak
  stresses from such a run are mesh-size sensitive (C3D4) and must not
  anchor a final strength claim without either a study or a stated
  assumption.

## Physics scope (v1)

- Linear static, solid tetrahedra (C3D4), N-mm-MPa unit system.
- FIXED / BOLTED / WELDED interfaces idealize to fully constrained supports
  (full clamp / tie); CONTACT interfaces apply loads only.
- Forces distribute over interface faces by tributary area (resultant exact);
  pressure becomes equivalent inward nodal loads; point loads attach to the
  nearest node (≤ 2 mm by default, loud error otherwise).
- **Moment loads are rejected with guidance**: replace with an equivalent
  force couple (two opposite `ForceLoad(point_mm=...)`).
- Nonlinear contact, bolt preload, fatigue: not claimed; do not infer them.

## What is checked

| Check | Source of truth |
| --- | --- |
| interface existence | package tag channel (`interface.*`) — missing names fail loudly |
| mating-face planarity (bolted/welded/fixed) | exact BREP surface type + sample fit, tolerance `planarity_tol_mm` |
| bearing area ≥ `min_area_mm2` | exact BREP face areas |
| keep-out compliance | OCC boolean intersection (volume + region bbox) |
| mesh element quality | signed volume + radius-ratio aspect ratio (numpy), gmsh minSICN cross-check; inverted/degenerate meshes are never solved |
| mesh independence | optional `MeshStudy`: finest-pair ΔQ vs tolerance; observed order, Richardson limit, GCI on geometric families; singularity signature flagged |
| strength per load case | FEMaster linear-static solve, nodal von Mises vs `Re / SF_req` |

No claims of strength, fatigue, thermal, vibration, tolerance compliance, or
regulatory fitness are made without the corresponding analysis actually run;
this addon runs (only) the linear-static analysis above and every report
number traces to it.
