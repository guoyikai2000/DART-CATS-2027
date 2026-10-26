"""Binary detection metrics and forgetting; validation thresholds never use test scores."""

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

MATRIX_KEYS = {
    "AUROC_AI_vs_HWT": "auroc_ai_vs_hwt",
    "empirical_test_TPR_AI_at_FPR": "empirical_test_tpr_ai",
    "empirical_test_FPR_HWT": "empirical_test_fpr_hwt",
    "valid_calibrated_test_TPR_AI": "valid_calibrated_test_tpr_ai",
    "valid_calibrated_test_FPR_HWT": "valid_calibrated_test_fpr_hwt",
}


def binary_scores(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2 or not np.isfinite(logits).all():
        raise ValueError("Expected finite binary logits with shape (n, 2).")
    scores = logits[:, 1] - logits[:, 0]
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite binary detection scores.")
    return scores


def checked_labels(labels, size):
    labels = np.asarray(labels)
    if labels.shape != (size,) or set(labels.tolist()) != {0, 1}:
        raise ValueError("Expected aligned labels containing both HWT=0 and AI=1.")
    return labels


def threshold_at_fpr(scores, labels, target_fpr=0.01):
    scores = np.asarray(scores, dtype=np.float64)
    labels = checked_labels(labels, len(scores))
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 <= target_fpr < 1:
        raise ValueError("Invalid scores or target FPR.")
    negatives = np.sort(scores[labels == 0])[::-1]
    # Strict > permits at most floor(FPR * negative_count) false positives, including ties.
    return float(negatives[math.floor(target_fpr * len(negatives) + 1e-12)])


def detection_metrics(logits, labels, target_fpr=0.01, valid_threshold=None):
    scores = binary_scores(logits)
    labels = checked_labels(labels, len(scores))
    hwt, ai = labels == 0, labels == 1
    empirical_threshold = threshold_at_fpr(scores, labels, target_fpr)
    empirical_detected = scores > empirical_threshold
    result = {
        "n": len(labels), "n_hwt": int(hwt.sum()), "n_ai": int(ai.sum()),
        "auroc_ai_vs_hwt": float(roc_auc_score(ai, scores)),
        "target_fpr": target_fpr,
        "empirical_test_threshold_hwt": empirical_threshold,
        "empirical_test_tpr_ai": float(empirical_detected[ai].mean()),
        "empirical_test_fpr_hwt": float(empirical_detected[hwt].mean()),
    }
    if valid_threshold is not None:
        if not np.isfinite(valid_threshold):
            raise ValueError("Non-finite validation threshold.")
        detected = scores > valid_threshold
        result.update({
            "valid_threshold_hwt": float(valid_threshold),
            "valid_calibrated_test_tpr_ai": float(detected[ai].mean()),
            "valid_calibrated_test_fpr_hwt": float(detected[hwt].mean()),
        })
    return result


def evaluate_domains(trainer, data, domains, stage, output_dir: Path, target_fpr):
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics, pooled = {}, {split: ([], []) for split in ("valid", "test")}
    for domain in domains:
        predictions = {}
        for split in ("valid", "test"):
            dataset = data[domain][split]
            result = trainer.predict(dataset.remove_columns(["sample_id"]))
            logits, labels = result.predictions, result.label_ids
            scores = binary_scores(logits)
            checked_labels(labels, len(scores))
            if len(dataset["sample_id"]) != len(scores):
                raise ValueError("Prediction count differs from dataset count.")
            predictions[split] = (logits, labels)
            pooled[split][0].append(logits)
            pooled[split][1].append(labels)
            if trainer.is_world_process_zero():
                with (prediction_dir / f"{stage}_{domain}_{split}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for sample_id, label, values, score in zip(
                        dataset["sample_id"], labels, logits, scores
                    ):
                        handle.write(json.dumps({
                            "sample_id": sample_id, "label": int(label),
                            "logits": values.tolist(), "score_ai_vs_hwt": float(score),
                        }, ensure_ascii=False, allow_nan=False) + "\n")
        valid_logits, valid_labels = predictions["valid"]
        threshold = threshold_at_fpr(binary_scores(valid_logits), valid_labels, target_fpr)
        metrics[domain] = detection_metrics(*predictions["test"], target_fpr, threshold)
        metrics[domain]["n_valid_hwt"] = int((np.asarray(valid_labels) == 0).sum())
    valid_logits, valid_labels = (np.concatenate(values) for values in pooled["valid"])
    test_logits, test_labels = (np.concatenate(values) for values in pooled["test"])
    threshold = threshold_at_fpr(binary_scores(valid_logits), valid_labels, target_fpr)
    metrics["__overall__"] = detection_metrics(test_logits, test_labels, target_fpr, threshold)
    metrics["__overall__"]["n_valid_hwt"] = int((valid_labels == 0).sum())
    return metrics


def summarize(history, domains, sequential, target_fpr):
    if not history or len(set(domains)) != len(domains):
        raise ValueError("Expected nonempty history and unique domains.")
    matrices = {
        name: [[stage["domain_metrics"][domain][key] for domain in domains] for stage in history]
        for name, key in MATRIX_KEYS.items()
    }
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("Non-finite performance matrix.")
    summary = {
        "target_fpr": target_fpr,
        "empirical_test_threshold_source": "same_test_HWT_scores; descriptive ROC working point",
        "valid_threshold_source": "held_out_valid_HWT_scores; test FPR is observed, not guaranteed",
        "positive_class": "D00 original AI; D01-D08 attack_text",
        "per_domain": {domain: {"final": history[-1]["domain_metrics"][domain]}
                       for domain in domains},
    }
    overall = history[-1]["domain_metrics"]["__overall__"]
    for name, key in MATRIX_KEYS.items():
        final = np.asarray(matrices[name])[-1]
        summary[f"final_AA_{name}"] = float(final.mean())
        summary[f"final_global_{name}"] = overall[key]
        summary[f"final_worst_domain_{name}"] = float(final.max() if key.endswith("fpr_hwt") else final.min())
    if sequential:
        learned_at = {stage["train_domains"][0]: index for index, stage in enumerate(history)
                      if len(stage["train_domains"]) == 1}
        if len(history) != len(domains) or set(learned_at) != set(domains):
            raise ValueError("Expected one completed training stage per domain.")
        old_domains = [domain for domain in domains if learned_at[domain] < len(history) - 1]
        for name in ("AUROC_AI_vs_HWT", "empirical_test_TPR_AI_at_FPR",
                     "valid_calibrated_test_TPR_AI"):
            matrix, backward, forgetting = np.asarray(matrices[name]), [], []
            for column, domain in enumerate(domains):
                postlearning = matrix[learned_at[domain]:, column]
                immediate, final, peak = map(float, (postlearning[0], postlearning[-1], postlearning.max()))
                summary["per_domain"][domain][name] = {
                    "immediate": immediate, "final": final, "postlearning_peak": peak,
                    "immediate_to_final_drop": immediate - final,
                    "peak_to_final_forgetting": peak - final,
                    "largest_postlearning_step_drop": float(max(0.0, np.max(postlearning[:-1] - postlearning[1:])))
                    if len(postlearning) > 1 else 0.0,
                }
                if domain in old_domains:
                    backward.append(final - immediate)
                    forgetting.append(peak - final)
            summary[f"BWT_{name}"] = float(np.mean(backward)) if backward else 0.0
            summary[f"average_forgetting_{name}"] = float(np.mean(forgetting)) if forgetting else 0.0
        summary["forgetting_population"] = "all domains except the last learned; prelearning scores excluded"
    return summary, matrices
