#!/usr/bin/env python3
"""Download the public checkpoints into models/ (Hugging Face Hub; some repositories require accepting
their license on the Hub first and `huggingface-cli login` / `hf auth login`).

  python scripts/download_models.py core          # what DART and the continual baselines need
  python scripts/download_models.py fixed         # the fixed detectors that are on the Hub
  python scripts/download_models.py all

The TASTE detector checkpoint (detector_ep3.pt) is not on the Hub; see models/README.md.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = Path(__file__).resolve().parents[1] / "models"
CORE = {
    "Ministral-3-8B-Instruct-2512-BF16": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "sup-simcse-bert-base-uncased": "princeton-nlp/sup-simcse-bert-base-uncased",
}
FIXED = {
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
    "RADAR-Vicuna-7B": "TrustSafeAI/RADAR-Vicuna-7B",
    "ImBD": "xyzhu1225/ImBD",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    "OSM-Det": "tarryzhang/OSM-Det",
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "core"
    todo = {"core": CORE, "fixed": FIXED, "all": {**CORE, **FIXED}}[which]
    for folder, repo in todo.items():
        print(f"{repo} -> models/{folder}", flush=True)
        snapshot_download(repo_id=repo, local_dir=str(MODELS / folder))


if __name__ == "__main__":
    main()
