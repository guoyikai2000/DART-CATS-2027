# Model checkpoints (not included)

`python scripts/download_models.py core|fixed|all` downloads the public checkpoints from the Hugging Face Hub into
the folders below (some repositories require accepting their license on the Hub first).

| Folder under `models/` | Source | Used by |
|---|---|---|
| `Ministral-3-8B-Instruct-2512-BF16` | `mistralai/Ministral-3-8B-Instruct-2512-BF16` (BF16 checkpoint of Ministral 3 8B Instruct) | detector backbone of all continual methods; DART writer; SSR / PseudoReplay generator |
| `sup-simcse-bert-base-uncased` | `princeton-nlp/sup-simcse-bert-base-uncased` | DART fidelity encoder; SSR selection |
| `gpt-j-6b` | `EleutherAI/gpt-j-6b` | Fast-DetectGPT, Lastde |
| `falcon-7b`, `falcon-7b-instruct` | `tiiuae/falcon-7b`, `tiiuae/falcon-7b-instruct` | Binoculars |
| `RADAR-Vicuna-7B` | `TrustSafeAI/RADAR-Vicuna-7B` | RADAR |
| `ImBD`, `gpt-neo-2.7B` | `xyzhu1225/ImBD` (scoring adapter), `EleutherAI/gpt-neo-2.7B` | ImBD |
| `OSM-Det` | `tarryzhang/OSM-Det` | OSM-Det |
| `TASTE/detector` | released TASTE detector of Li et al. (ICLR 2026): `config.json`, tokenizer files, `detector_ep3.pt` | TASTE |

Any other location works as well: pass `--model_name_or_path` or edit the paths in the configs.
