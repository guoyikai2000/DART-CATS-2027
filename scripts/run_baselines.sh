#!/usr/bin/env bash
# Continual baselines on the four domain orders, Joint once, and the fixed detectors (Table 2).
# Usage: bash scripts/run_baselines.sh [continual|fixed|all]   (one GPU unless noted; set CUDA_VISIBLE_DEVICES)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
what="${1:-continual}"
if [[ "$what" == continual || "$what" == all ]]; then
  for order in O0 O1 O2 O3; do
    python baselines/continual/train_lora.py sequential_lora --order "$order"
    for m in er ewc lwf olora ssr pseudo; do
      python baselines/continual/run_easycl.py --config "baselines/continual/configs/$m.yaml" --order "$order"
    done
    python baselines/continual/run_moram.py --config baselines/continual/configs/moram.yaml --order "$order"
  done
  python baselines/continual/train_lora.py joint_lora --order O0
fi
if [[ "$what" == fixed || "$what" == all ]]; then
  # radar / imbd / taste / osm_det run in the main environment; fast_detect_gpt / lastde / binoculars were
  # run in the native environment of baselines/fixed_detectors/requirements-native.txt (Binoculars: 2 GPUs).
  for m in radar imbd taste osm_det fast_detect_gpt lastde binoculars; do
    python baselines/fixed_detectors/score_fixed.py --config "baselines/fixed_detectors/configs/$m.yaml"
  done
fi
python scripts/summarize.py
