#!/usr/bin/env bash
# DART on the four domain orders (Table 2), then the O1 ablations (Table 3) and sensitivity runs (Figure 5).
# Usage: bash scripts/run_dart.sh [main|ablations|sensitivity|all]   (one GPU; set CUDA_VISIBLE_DEVICES)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
what="${1:-main}"
if [[ "$what" == main || "$what" == all ]]; then
  for order in O0 O1 O2 O3; do
    python dart/run_dart.py --config dart/configs/dart.yaml --order "$order"
  done
fi
if [[ "$what" == ablations || "$what" == all ]]; then
  for cfg in dart/configs/ablations/*.yaml; do
    python dart/run_dart.py --config "$cfg" --order O1 --tag "$(basename "$cfg" .yaml)"
  done
fi
if [[ "$what" == sensitivity || "$what" == all ]]; then
  for cfg in dart/configs/sensitivity/*.yaml; do
    python dart/run_dart.py --config "$cfg" --order O1 --tag "$(basename "$cfg" .yaml)"
  done
fi
python scripts/summarize.py
