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

The addon declares a `python-env` runtime with a command prefix:

```toml
[runtime]
command_prefix = 'PATH="{addon_dir}/.venv/bin:$PATH"'
```

Every dispatch runs through that prefix — `sca addon use` joins it in
front of your command, and the descriptor's `check_cmd` is probed
through it too. `{addon_dir}` resolves to the installed addon directory
(`~/.sca/addons/sca-connectors-addon` by default). Provision the
interpreter there so the addon owns its environment:

```bash
# inside the installed addon dir — its .venv wins over the caller's PATH
python3 -m venv .venv
.venv/bin/python -m pip install -e . numpy gmsh matplotlib simplecadapi
```

Without that `.venv` the prefix falls through to the caller's `PATH`
gracefully, so a dev checkout works too. `check_cmd` probes
`import connverify, gmsh, numpy, simplecadapi`; a failing probe is a
loud warning, never a silent skip. Point `CONNVERIFY_FEMASTER` at a
FEMaster binary (see `tools/fetch_femaster.sh`).

### Verifying the addon descriptor

```bash
.venv/bin/sca addon add NiJingzhe/sca-connectors-addon   # install from GitHub
.venv/bin/sca addon list
.venv/bin/sca addon use sca-connectors-addon \
    python -c 'import connverify; print(connverify.__version__)'
.venv/bin/sca addon use sca-connectors-addon    # report prefix + addon dir
```

The naming standard (repository == descriptor == skill frontmatter) and
the prefix are validated by `sca addon add` at install time.
