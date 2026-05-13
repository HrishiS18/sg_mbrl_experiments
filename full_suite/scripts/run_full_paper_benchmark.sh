#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
STATUS_FILE="$ROOT/results/full_paper_status_${STAMP}.txt"
ACTIVE_HEADLINE_CONFIG="$ROOT/configs/headline_${STAMP}.json"

status() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$STATUS_FILE"
}

source "$ROOT/.venv/bin/activate"
export PYTORCH_ENABLE_MPS_FALLBACK=1

DEVICE="${SG_MBRL_DEVICE:-}"
if [ -z "$DEVICE" ]; then
  DEVICE="$(python - <<'PY'
import torch

if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
fi
status "Using torch device: $DEVICE"

status "Preparing headline config."
python - <<PY
import json
from pathlib import Path

src = Path("configs/headline.json")
dst = Path("$ACTIVE_HEADLINE_CONFIG")
cfg = json.loads(src.read_text())
cfg["run_dir"] = "results/headline_${STAMP}"
if cfg.get("train_defaults") is not None:
    cfg["train_defaults"]["device"] = "$DEVICE"
if cfg.get("cem_defaults") is not None:
    cfg["cem_defaults"]["device"] = "$DEVICE"
dst.write_text(json.dumps(cfg, indent=2))
print(dst)
PY

status "Checking dependencies."
python -m sg_mbrl_full.scripts.check_dependencies | tee -a "$STATUS_FILE"

status "Starting neural headline suite."
python -u -m sg_mbrl_full.scripts.run_headline --config "$ACTIVE_HEADLINE_CONFIG"

status "Aggregating neural headline suite."
python -u -m sg_mbrl_full.scripts.aggregate --run-dir "results/headline_${STAMP}"

status "Starting model-free SAC/PPO/TD3 suite."
python -u -m sg_mbrl_full.scripts.run_model_free --config configs/model_free.json

status "Full paper benchmark completed."
