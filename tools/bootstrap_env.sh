#!/bin/sh
# Provision the addon's runtime environment.
#
# Target: the addon's runtime directory ($SCA_RUNTIME_DIR when dispatched
# through `sca addon use`, i.e. ~/.sca/runtimes/sca-connectors-addon) —
# state that lives outside the payload and survives `sca addon update`.
# In a dev checkout (no SCA_RUNTIME_DIR) it falls back to the repo's own
# .venv (the dev-setup workflow in README).
#
# simplecadapi is not on public PyPI — pass a local SimpleCADAPI checkout
# (or set SCA_SDK_PATH).
#
# Usage: tools/bootstrap_env.sh [/path/to/SimpleCADAPI]
set -e
TARGET="${SCA_RUNTIME_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SDK="${1:-${SCA_SDK_PATH:-$(cd "$(dirname "$0")/.." && pwd)/../SimpleCADAPI}}"
if [ ! -f "$SDK/pyproject.toml" ]; then
    echo "SimpleCADAPI checkout not found at: $SDK" >&2
    echo "pass it as the first argument or set SCA_SDK_PATH" >&2
    exit 1
fi
# connverify requires python >=3.10,<3.14; prefer uv (pins 3.10) and
# fall back to the system interpreter when it is in range
if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.10 "$TARGET/.venv"
    UV_PY="uv pip install --python $TARGET/.venv/bin/python"
    $UV_PY -e . numpy "gmsh>=4.15,<5" "matplotlib>=3.7"
    $UV_PY -e "$SDK"
else
    python3 -m venv "$TARGET/.venv"
    "$TARGET/.venv/bin/python" -m pip install --upgrade pip -q
    "$TARGET/.venv/bin/python" -m pip install -e . numpy "gmsh>=4.15,<5" "matplotlib>=3.7" -e "$SDK"
fi
"$TARGET/.venv/bin/python" -c "import connverify, gmsh, numpy, simplecadapi; print('runtime OK')"
