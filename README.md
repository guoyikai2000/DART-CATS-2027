# Keep the Difference, Not the Data — DART and the CATS benchmark

Code and data for the anonymous submission *"Keep the Difference, Not the Data"*.

* **CATS** (Continual ATtack Stream): 8,100 paired human/AI texts in nine domains. D00 keeps the original AI texts; each of D01–D08 applies one evasion attack (lexical substitution, syntactic paraphrasing, discourse restructuring, local span rewriting, full rewriting, round-trip translation, style transfer, detector-guided rewriting). A detector learns the domains one after another and is evaluated on all of them at 1% FPR.
* **DART**: continual AIGT detection that stores no past texts. For every domain an LLM analyst writes a natural-language *difference memory* (how AI texts differ from paired human texts), verified against held-out real AI texts before the data are discarded. Later stages replay the memories on the *current* human texts: a frozen writer continues the opening of a human text under a memorized pattern, and the detector keeps the candidate it finds hardest.

## Repository layout

```
common/                  shared protocol: CATS loading and verification, detector, metrics, training shell
dart/                    DART (run_dart.py), memory writing, analyst client, writer, configs
  configs/dart.yaml          main setting (Table 2)
  configs/ablations/         Table 3 (w/o replay, w/o verification, random selection, synthetic human)
  configs/sensitivity/       Figure 5 (beta, candidates, revisions) and alpha
  relay/                     optional relay for GPU servers without internet access
baselines/continual/     Sequential / Joint (train_lora.py), ER / EWC / LwF / O-LoRA / SSR / PseudoReplay
                         (run_easycl.py), MoRAM (run_moram.py) and their configs
baselines/fixed_detectors/  Fast-DetectGPT, Binoculars, Lastde, RADAR, ImBD, TASTE, OSM-Det (score_fixed.py)
data/CATS/               the benchmark (train/valid/test x D00-D08) + manifest.json + data card
scripts/                 model download, third-party fetch, run-all scripts, result summary
tests/test_offline.py    CPU-only checks (data, analyst transports, memory writing with fakes)
```

All relative paths in configs and on the command line are resolved against the repository root.

## Setup

```bash
conda create -n dart python=3.11 -y && conda activate dart
pip install -r requirements.txt          # CUDA 12.8 build of torch 2.9.1 was used
python common/data.py                     # verifies all 27 CATS files against data/CATS/manifest.json
python tests/test_offline.py              # CPU-only checks, no weights or API key needed
python scripts/download_models.py core    # Ministral-3-8B-Instruct-2512 (BF16) + sup-SimCSE into models/
```

See `models/README.md` for the expected folders. Fast-DetectGPT, Lastde and Binoculars were scored in a second
environment (`baselines/fixed_detectors/requirements-native.txt`, torch 2.1 / transformers 4.46) with the authors'
scoring functions, fetched by `bash scripts/fetch_third_party.sh`.

**Hardware.** All runs used NVIDIA A100 80GB GPUs, one GPU per run. DART keeps the detector (Ministral-3-8B + LoRA),
the frozen writer (a second copy of the base checkpoint) and SimCSE on one GPU; set `method.writer.device: cuda:1`
to place the writer on a second GPU.

## The analyst LLM

DART calls an LLM analyst only to write and revise memories (about 24 calls per domain order). The paper used
GPT-6 Luna (`method.analyst.model`), through an OpenAI-compatible Chat Completions endpoint with strict JSON-schema
outputs:

```bash
export OPENAI_API_KEY=...                 # read from the environment only
export OPENAI_BASE_URL=https://...        # optional, default https://api.openai.com/v1
```

For GPU servers without internet access, set `method.analyst.backend: queue`: the training process writes
requests to `method.analyst.queue_root`, and `dart/relay/analyst_relay.py`, started on a machine with internet
access, serves them over SSH through `dart/relay/analyst_broker.py` (the key never reaches the server). The writer
and the replay itself run locally; the `no_replay` ablation needs no analyst at all.

## Running

```bash
# DART (Table 2): one order per run; results in outputs/dart/<order>/<tag>_<time>/
python dart/run_dart.py --config dart/configs/dart.yaml --order O1
python dart/run_dart.py --config dart/configs/dart.yaml --order O1 --smoke     # 2 stages, tiny subsets

# ablations (Table 3) and sensitivity (Figure 5), on O1
python dart/run_dart.py --config dart/configs/ablations/synthetic_human.yaml --order O1
python dart/run_dart.py --config dart/configs/sensitivity/candidates_1.yaml --order O1

# continual baselines
python baselines/continual/train_lora.py sequential_lora --order O1
python baselines/continual/train_lora.py joint_lora --order O0
python baselines/continual/run_easycl.py --config baselines/continual/configs/ssr.yaml --order O1
python baselines/continual/run_moram.py  --config baselines/continual/configs/moram.yaml --order O1

# fixed detectors (one scoring pass; AA and pooled do not depend on the order)
python baselines/fixed_detectors/score_fixed.py --config baselines/fixed_detectors/configs/osm_det.yaml

# everything, then the summary table (mean ± std over the orders found)
bash scripts/run_dart.sh all && bash scripts/run_baselines.sh all
python scripts/summarize.py
```

Orders (all start from the unattacked D00; `common/cl_common.py`):

| Order | Sequence |
|---|---|
| O0 | D00 D08 D03 D02 D06 D04 D05 D01 D07 |
| O1 | D00 D07 D01 D05 D04 D06 D02 D03 D08 |
| O2 | D00 D05 D08 D06 D03 D01 D04 D07 D02 |
| O3 | D00 D06 D02 D03 D08 D07 D01 D05 D04 |

Every run writes `continual_metrics.json`; the paper's metrics are `summary.final_AA_empirical_test_TPR_AI_at_FPR`
(AA), `summary.final_global_empirical_test_TPR_AI_at_FPR` (pooled TPR, one threshold over the 2,700 human test
texts) and `summary.average_forgetting_empirical_test_TPR_AI_at_FPR` (FGT). Thresholds are the empirical 1% FPR
point of the human test texts (strict `>`), per domain for AA and pooled for the pooled TPR.

## Main results (TPR at 1% FPR, %; continual methods: mean ± std over O0–O3)

| Method | AA | Pooled | FGT |
|---|---|---|---|
| Fast-DetectGPT | 13.26 | 12.04 | – |
| Binoculars | 12.89 | 5.96 | – |
| Lastde | 2.85 | 1.70 | – |
| RADAR | 1.93 | 0.48 | – |
| ImBD | 16.07 | 6.78 | – |
| TASTE | 8.07 | 8.07 | – |
| OSM-Det | 56.89 | 49.63 | – |
| Sequential | 88.01 ± 3.31 | 81.73 ± 7.83 | 9.99 ± 3.39 |
| EWC | 88.87 ± 1.99 | 85.05 ± 2.81 | 9.49 ± 2.06 |
| LwF | 88.41 ± 2.54 | 83.81 ± 4.11 | 7.66 ± 4.35 |
| O-LoRA | 89.49 ± 1.95 | 84.06 ± 9.43 | 8.51 ± 2.07 |
| MoRAM | 90.71 ± 0.86 | 87.05 ± 3.86 | 7.14 ± 0.89 |
| PseudoReplay | 86.12 ± 2.63 | 73.98 ± 19.34 | 12.57 ± 2.76 |
| SSR | 90.86 ± 1.58 | 86.31 ± 6.51 | 7.08 ± 2.37 |
| ER | 90.48 ± 2.98 | 87.92 ± 4.69 | 7.84 ± 3.33 |
| **DART** | **94.22 ± 0.52** | **93.27 ± 1.36** | **3.13 ± 0.69** |
| Joint (upper bound) | 97.22 | 97.07 | – |

DART per order (AA / pooled / FGT): O0 93.89 / 94.26 / 3.92, O1 94.70 / 91.48 / 2.29, O2 94.63 / 92.93 / 3.38,
O3 93.67 / 94.41 / 2.92. Ablations on O1: w/o replay 92.85 / 88.04 / 4.04; w/o scoring, validation and revision
90.93 / 90.96 / 6.88; w/o hard selection 92.52 / 87.74 / 5.12; w/o real human anchors 92.07 / 78.30 / 2.33.

## Reproducibility notes

* Seeds: root seed 2026, stage i uses 2026 + i; all random choices of DART use their own `numpy` generators
  seeded from the stage seed. The writer samples (temperature 0.9) in batches formed by concurrent requests and
  the analyst is a remote LLM, so DART runs are not bitwise reproducible; expect small deviations from the table.
* The runs of the paper also executed an auxiliary content-extraction call of the analyst for every training
  human text (Section 5.6); its output is not used by the continuation writer, so this release omits it.
* `data/CATS/manifest.json` fixes every file by size, SHA-256 and canonical SHA-256; the loader refuses modified
  files. Keep `.gitattributes` so that git does not rewrite line endings.

## Ethics and license

CATS is derived from MAGA-Bench and MDTA and inherits their licenses and terms of use; the attacked texts are
released for research on detection only. Detector outputs, including DART's, should not be the sole basis for
decisions about individuals. Third-party detector code and weights are not redistributed (see `models/README.md`
and `scripts/fetch_third_party.sh`).
