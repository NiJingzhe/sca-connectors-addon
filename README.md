# connverify — 静态连接件验证环境 (SimpleCADAPI addon)

Static connector verification environment plugin for SimpleCADAPI, with
FEMaster as the FEM backend.

A connector verification environment is three formal object classes:

1. **Interfaces** — named end faces (`interface.*` tags on the part) plus a
   connection method (FIXED / BOLTED / WELDED / CONTACT).
2. **Load cases** — forces / moments / pressures applied to interfaces or points.
3. **Envelope constraints** — keep-out regions the part must not invade.

Units are millimeters / newtons / megapascals throughout (N-mm-MPa), matching
both `.scadpkg` (mm) and FEMaster's deck convention.

Pipeline: `.scadpkg` + `VerificationEnv` → connection checks → envelope checks
→ gmsh mesh → FEMaster deck → solve → `.frd` parse → geometry-directed
feedback report.

## Layout

- `src/connverify/` — runtime package
- `skill/SKILL.md` — agent-facing skill document (installed as `sca-connverify`)
- `tests/` — pytest suite (TDD)
- `vendor/` — local FEMaster binary (dev only; not committed — see
  `tools/fetch_femaster.sh`)
- `examples/` — end-to-end example scripts

## Dev setup

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ../SimpleCADAPI   # SDK from sibling source
uv pip install -p .venv/bin/python -e ".[dev]"
./tools/fetch_femaster.sh                                # solver binary into vendor/
.venv/bin/python -m pytest
```
