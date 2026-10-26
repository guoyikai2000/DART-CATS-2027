#!/usr/bin/env python3
"""Collect finished runs under outputs/ and print AA / pooled TPR / forgetting (TPR at 1% FPR, %).

  python scripts/summarize.py [outputs]

Continual runs are grouped by method folder; the mean and sample standard deviation are taken over the
domain orders found (the paper reports O0-O3). Smoke runs are skipped.
"""
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "common"))
ORDERS = {
    "O0": ["D00", "D08", "D03", "D02", "D06", "D04", "D05", "D01", "D07"],
    "O1": ["D00", "D07", "D01", "D05", "D04", "D06", "D02", "D03", "D08"],
    "O2": ["D00", "D05", "D08", "D06", "D03", "D01", "D04", "D07", "D02"],
    "O3": ["D00", "D06", "D02", "D03", "D08", "D07", "D01", "D05", "D04"],
}
KEYS = ("final_AA_empirical_test_TPR_AI_at_FPR", "final_global_empirical_test_TPR_AI_at_FPR",
        "average_forgetting_empirical_test_TPR_AI_at_FPR")


def fmt(values):
    values = [100 * v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f}±{statistics.stdev(values):.2f}"


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "outputs"
    groups = {}
    for name in ("continual_metrics.json", "joint_metrics.json", "fixed_metrics.json"):
        for path in sorted(root.rglob(name)):
            rel = path.relative_to(root)
            if any("smoke" in part for part in rel.parts):
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            summary = result.get("summary")
            if not summary:
                continue  # unfinished run
            order = next((o for o, d in ORDERS.items() if d == result.get("domain_order")), None)
            parts = [p for p in rel.parts[:-1] if p not in ORDERS]
            if order is not None and parts and path.parent.parent.name in ORDERS:
                parts = parts[:-1]  # drop the <tag>_<time> folder of DART runs
            method = "/".join(parts) or rel.parts[0]
            groups.setdefault(method, {})[order or "-"] = [summary.get(k) for k in KEYS]
    if not groups:
        print(f"No finished runs under {root}")
        return
    print(f"{'method':40s} {'orders':12s} {'AA':>13s} {'pooled':>13s} {'FGT':>13s}")
    for method, runs in sorted(groups.items()):
        cols = list(zip(*runs.values()))
        print(f"{method:40s} {','.join(sorted(runs)):12s} {fmt(cols[0]):>13s} {fmt(cols[1]):>13s} {fmt(cols[2]):>13s}")


if __name__ == "__main__":
    main()
