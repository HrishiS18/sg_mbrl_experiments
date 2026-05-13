#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/Users/hrishismac/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"

if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

echo "Using Python: $PY"
"$PY" -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT/requirements-neural.txt"
python -m sg_mbrl_full.scripts.check_dependencies

