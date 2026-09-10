---
name: connverify
description: Verify static connector parts (brackets, mounts, flanges) captured as .scadpkg product packages. Use when a connector must be checked for connection-interface validity, load-case strength (linear-static FEM via FEMaster), or keep-out envelope compliance. Consumes finished .scadpkg geometry with interface.* face tags; produces a deterministic JSON + markdown report whose findings point at the owning interfaces, features, and coordinates.
license: Apache-2.0
metadata:
  project: sca-fem-addone
  version: 0.1.0
---

# connverify — static connector verification

Turn a natural-language verification request into a **formal, deterministic
verification environment**, run it, and read a **geometry-directed** report.

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

```bash
python -c "import connverify, gmsh, numpy, simplecadapi"   # addon env
# solver binary — any of:
export CONNVERIFY_FEMASTER=/path/to/FEMaster   # or femaster on PATH,
                                               # or <addon>/vendor/FEMaster
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
| "bolted to the wall", "welded", "clamped" | `Interface(method=ConnectionMethod.BOLTED/WELDED/FIXED)` — constraining supports |
| "the shaft presses here", mating surface | `Interface(method=ConnectionMethod.CONTACT)` — free load-entry face |
| forces / weights | `ForceLoad(target=<iface>, fx_n=..., fy_n=..., fz_n=...)` or `ForceLoad(point_mm=(x, y, z), ...)` |
| pressure | `PressureLoad(interface=<iface>, magnitude_mpa=...)` (positive pushes onto the face) |
| "must not stick into X" | `KeepOutBox(name=..., min_corner_mm=..., max_corner_mm=...)` |
| material, required safety factor | `Material(...)`, `safety_factor_required=` |

```python
from connverify.env import (ConnectionMethod, ForceLoad, Interface,
    KeepOutBox, LoadCase, Material, VerificationEnv)

env = VerificationEnv(
    name="motor bracket static",
    part_package="out/bracket.scadpkg",
    material=Material(name="steel_s355", youngs_modulus_mpa=210000.0,
                      poisson_ratio=0.3, yield_strength_mpa=355.0,
                      density_t_per_mm3=7.85e-9),
    interfaces=[
        Interface(name="mount_face", method=ConnectionMethod.BOLTED),   # wall
        Interface(name="load_pad",   method=ConnectionMethod.CONTACT),  # motor foot
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

Artifacts: `decks/<CASE>.inp` (+ `.frd`/`.res`), `report.json`, `report.md`.

### 3. Read the report like an engineer

- `verdict` — pass only if interfaces, envelopes, AND every load case pass.
- Per load case: `max_von_mises_mpa`, `safety_factor = Re / max_vM`,
  `max_displacement_mm`, top-3 hotspots each with `location_mm`,
  `on_interface` (preferentially the constrained one), and
  `owning_feature` — the feature-graph node that **produced the hotspot's
  BREP face** (from the package's topology snapshot; `graph_id` + `node_id`
  map straight back into the modeling feature tree).
- `feedback` — ordered, actionable items with numbers and locations.

### 4. Repair loop (geometry changes live in the modeling workflow)

1. Report names interface 'mount_face', feature `node_00000012`, location.
2. Open the part's modeling source; change the owning feature/parameter
   (fillet radius, plate thickness, gusset) — not the verification.
3. Re-`capture` the `.scadpkg`; re-run `verify` with the SAME env JSON.
4. Compare `max_von_mises_mpa` / `safety_factor` across runs — the
   environment is deterministic, so deltas are real geometry effects.

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
| strength per load case | FEMaster linear-static solve, nodal von Mises vs `Re / SF_req` |

No claims of strength, fatigue, thermal, vibration, tolerance compliance, or
regulatory fitness are made without the corresponding analysis actually run;
this addon runs (only) the linear-static analysis above and every report
number traces to it.
