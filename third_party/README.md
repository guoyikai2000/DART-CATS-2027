# Third-party code (not included)

`bash scripts/fetch_third_party.sh` clones https://github.com/TrustMedia-zju/Lastde_Detector into
`third_party/Lastde_Detector` and checks the SHA-256 of the four files used by
`baselines/fixed_detectors/score_fixed.py` (`py_scripts/baselines/fast_detect_gpt.py`, `binoculars.py`,
`scoring_methods/based_scoring_baselines.py`, `scoring_methods/fastMDE.py`) against the versions used in the paper.
