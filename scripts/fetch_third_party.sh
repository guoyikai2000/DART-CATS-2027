#!/usr/bin/env bash
# Fetch the authors' scoring scripts for Fast-DetectGPT, Binoculars and Lastde (used by
# baselines/fixed_detectors/score_fixed.py) from the Lastde repository, and check them against the
# versions used in the paper. Run from anywhere; files go to third_party/Lastde_Detector.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/Lastde_Detector"
if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 https://github.com/TrustMedia-zju/Lastde_Detector "$DEST"
fi
python - "$DEST/py_scripts/baselines" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {
    "fast_detect_gpt.py": "6ed45ce05bbd6a5dfa971f9806613da71378a81bf256bc46cc7d3795769e9c46",
    "binoculars.py": "fdcef8e6a7359fbfa6d1b55e9e4467657821e5d5d0c9bde1ed0e74849ebcbb83",
    "scoring_methods/based_scoring_baselines.py": "1ac0fdf71a88108cbcdf49b9e241d3adf8870a2f871a839c97f1addc244d9de9",
    "scoring_methods/fastMDE.py": "14b6549616f3a2080d1c1029e0fe0b7e13faf86f2cdc7e641befc094670c155f",
}
ok = True
for name, sha in expected.items():
    path = root / name
    got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    status = "ok" if got == sha else ("MISSING" if got is None else "DIFFERS from the paper's version")
    ok &= got == sha
    print(f"{name:45s} {status}")
sys.exit(0 if ok else 1)
PY
