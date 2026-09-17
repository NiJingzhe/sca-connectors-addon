# sca-connectors-addon — 静态连接件验证环境 (SimpleCADAPI addon)

Static connector verification environment plugin for SimpleCADAPI, with
FEMaster as the FEM backend. Addon name, skill name, and repository name
are one identity: `sca-connectors-addon` (the Python distribution /
import package inside the addon is `connverify`).

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
- `skill/SKILL.md` — agent-facing skill document (installed as
  `sca-connectors-addon`)
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
.venv/bin/python examples/plate_mount_demo.py            # full pipeline demo
```

### Runtime provisioning (for `sca` installs)

Runtime state lives OUTSIDE the payload, in the addon's runtime directory
(`~/.sca/runtimes/sca-connectors-addon`): the descriptor's
`command_prefix` pins `{runtime_dir}/.venv`, `sca addon update` replaces
the payload wholesale but never touches that venv, and `sca addon remove`
deletes it with everything else. Provision it once (simplecadapi is not
on public PyPI — pass your local SimpleCADAPI checkout):

```bash
sca addon use sca-connectors-addon \
    sh "$SCA_ADDON_DIR/tools/bootstrap_env.sh" /path/to/SimpleCADAPI
sca addon check sca-connectors-addon     # re-probe now; registry refreshes
```

Until the venv exists the probe falls through to the caller's `PATH`
(a dev checkout works too); a failing probe is a loud warning, never a
silent skip. The probe itself is hermetic once provisioned: it runs the
runtime venv's python with `importlib.util.find_spec`, so it neither
depends on the caller's environment nor pays cold-import time. Point
`CONNVERIFY_FEMASTER` at a FEMaster binary (see `tools/fetch_femaster.sh`).

### Verifying the addon descriptor

```bash
.venv/bin/sca addon add NiJingzhe/sca-connectors-addon   # install from GitHub
.venv/bin/sca addon list
.venv/bin/sca addon use sca-connectors-addon \
    python -c 'import connverify; print(connverify.__version__)'
.venv/bin/sca addon use sca-connectors-addon    # report prefix + runtime dir
.venv/bin/sca addon check sca-connectors-addon  # re-probe the runtime NOW
```

The naming standard (repository == descriptor == skill frontmatter) and
the prefix are validated by `sca addon add` at install time.
