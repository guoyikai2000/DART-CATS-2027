"""CATS data: verify the frozen splits and expand every pair into two classification texts.

Each split file holds pairs; a pair becomes two texts, the human text (label 0) followed by its AI
counterpart (label 1). The AI side is `ai_generated_text` in D00 (no attack) and `attack_text` in
D01-D08. `inspect_data` checks every file against data/CATS/manifest.json and fails on any change.

CPU-only check:  python common/data.py --data_dir data/CATS
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

DOMAINS = tuple(f"D{i:02d}" for i in range(9))
LABELS = {"HWT": 0, "AI": 1}
# folder -> (value of the `split` field inside the rows, pairs per domain)
SPLITS = {"train": ("train", 500), "valid": ("dev", 100), "test": ("test", 300)}
MANIFEST_NAME = "manifest.json"
CONTENT_FINGERPRINT = "6e2e01582778587680e7552be182fb9353b2f7fa8471cb0e9445a554b27d198f"
SYSTEM_PROMPT = (
    "You are an AI-generated text detector. Determine whether the following text "
    "is human-written or AI-generated, including AI-generated text that has been "
    "rewritten. Human-written is label 0 and AI-generated is label 1."
)


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".writing.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def format_prompt(text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{text.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def fixed_prompt_encoder(tokenizer, max_length: int):
    """Keep the fixed detector prompt and the tokenizer's outer special tokens; truncate only the body."""
    if type(max_length) is not int or not 1 <= max_length <= 512:
        raise ValueError("Expected an integer detector length from 1 through 512")
    marker = "FIXED_BODY_BOUNDARY_20261013"
    prefix, suffix = format_prompt(marker).split(marker)
    prefix_ids = tokenizer(prefix, add_special_tokens=False,
                           return_token_type_ids=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False,
                           return_token_type_ids=False)["input_ids"]
    decorated_prefix = tokenizer(prefix, add_special_tokens=True,
                                 return_token_type_ids=False)["input_ids"]
    # Fast-tokenizer backend postprocessors need not match the Python build_inputs method.
    locations = [i for i in range(len(decorated_prefix) - len(prefix_ids) + 1)
                 if decorated_prefix[i:i + len(prefix_ids)] == prefix_ids]
    if not prefix_ids or not suffix_ids or len(locations) != 1:
        raise ValueError("Native tokenizer outer-special layout is ambiguous")
    start = locations[0]
    before, after = decorated_prefix[:start], decorated_prefix[start + len(prefix_ids):]
    budget = max_length - len(before) - len(prefix_ids) - len(suffix_ids) - len(after)
    if budget < 1:
        raise ValueError("Fixed detector prompt and special tokens leave no body budget")

    def tokenize(texts):
        bodies = [text.strip() for text in texts]
        if any(not body for body in bodies):
            raise ValueError("Empty detector body")
        body_ids = tokenizer(bodies, add_special_tokens=False,
                             return_token_type_ids=False)["input_ids"]
        result = []
        for body in body_ids:
            retained = body[:budget]
            ids = before + prefix_ids + retained + suffix_ids + after
            suffix_start = len(before) + len(prefix_ids) + len(retained)
            if (not retained or len(ids) > max_length
                    or ids[suffix_start:suffix_start + len(suffix_ids)] != suffix_ids
                    or (after and ids[-len(after):] != after)):
                raise ValueError("Fixed suffix or detector input length differs")
            result.append(ids)
        return {"input_ids": result, "attention_mask": [[1] * len(ids) for ids in result]}

    return tokenize, {"input_protocol": "fixed_detector_segments_body_truncation_v1",
                      "max_length": max_length, "body_token_budget": budget,
                      "fixed_prefix_tokens": len(prefix_ids), "fixed_suffix_tokens": len(suffix_ids),
                      "outer_special_tokens": len(before) + len(after)}


def inspect_data(root: Path) -> tuple[dict, dict]:
    """CPU-only verification against manifest.json; rows are never altered, sampled or repartitioned."""
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found; data_dir must point to the CATS folder")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    expected_files = {f"{split}/{domain}.json" for split in SPLITS for domain in DOMAINS}
    if (manifest.get("pair_count") != 8100 or manifest.get("text_count") != 16200
            or set(manifest.get("files", {})) != expected_files
            or manifest.get("content_fingerprint") != CONTENT_FINGERPRINT):
        raise ValueError("Incomplete or unexpected CATS manifest")
    prepared, bindings, ids, sources, groups = {}, {}, set(), set(), {}
    for domain in DOMAINS:
        prepared[domain] = {}
        for folder, (row_split, quota) in SPLITS.items():
            relative = f"{folder}/{domain}.json"
            raw = (root / relative).read_bytes()
            rows = json.loads(raw)
            entry = manifest["files"][relative]
            if (not isinstance(rows, list) or len(rows) != quota
                    or entry.get("count") != quota or entry.get("bytes") != len(raw)
                    or entry.get("sha256") != sha256(raw)
                    or entry.get("canonical_sha256") != sha256(canonical(rows))):
                raise ValueError(f"Split file differs from the manifest: {relative} "
                                 "(check that git did not convert line endings, see .gitattributes)")
            examples = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"Non-object pair: {relative}")
                pair_id, source, group = row.get("id"), row.get("source_id"), row.get("human_group_id")
                if (type(pair_id) is not int or not 1 <= pair_id <= 8100 or pair_id in ids
                        or not isinstance(source, str) or not source or source in sources
                        or not isinstance(group, str) or not group
                        or row.get("task_domain") != domain or row.get("split") != row_split):
                    raise ValueError(f"Pair identity, domain or split differs: {relative}")
                if group in groups and groups[group] != row_split:
                    raise ValueError(f"Source group crosses splits: {relative}")
                ids.add(pair_id)
                sources.add(source)
                groups[group] = row_split
                positive_field = "ai_generated_text" if domain == "D00" else "attack_text"
                if domain == "D00" and row.get("attack_text") is not None:
                    raise ValueError("D00 must keep its unattacked AI text")
                for field, name in (("original_text", "HWT"), (positive_field, "AI")):
                    text = row.get(field)
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Empty classification text: {relative}/{field}")
                    examples.append({"sample_id": f"{pair_id}:{name}", "text": text,
                                     "labels": LABELS[name]})
            prepared[domain][folder] = examples
            bindings[relative] = {key: entry[key] for key in
                                  ("count", "bytes", "sha256", "canonical_sha256")}
    if ids != set(range(1, 8101)):
        raise ValueError("Pair ids do not cover exactly 1 through 8100")
    content_map = {relative: entry["canonical_sha256"] for relative, entry in bindings.items()}
    fingerprint = sha256(json.dumps(content_map, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))
    if fingerprint != manifest.get("content_fingerprint"):
        raise ValueError("Content fingerprint differs from the manifest")
    audit = {
        "data_dir": str(root), "manifest_sha256": sha256(manifest_raw),
        "content_fingerprint": fingerprint, "files": bindings,
        "pair_count": len(ids), "classification_text_count": 16200,
        "source_group_count": len(groups), "split_text_counts": {
            folder: 9 * quota * 2 for folder, (_, quota) in SPLITS.items()},
        "valid_folder_split_field": "dev", "label_map": LABELS,
        "positive_field_rule": "D00 ai_generated_text; D01-D08 attack_text",
    }
    return prepared, audit


def load_data(root: Path, domains: list[str], tokenizer, max_length: int,
              prepared: dict | None = None) -> dict:
    from datasets import Dataset

    if prepared is None:
        prepared, _ = inspect_data(root)

    encode, _ = fixed_prompt_encoder(tokenizer, max_length)

    def tokenize(batch):
        return encode(batch["text"])

    return {
        domain: {split: Dataset.from_list(prepared[domain][split]).map(
            tokenize, batched=True, remove_columns=["text"], desc=f"{split}/{domain}"
        ) for split in SPLITS} for domain in domains
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU-only check of the CATS splits")
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "CATS")
    args = parser.parse_args()
    _, audit = inspect_data(args.data_dir)
    print(json.dumps({k: v for k, v in audit.items() if k != "files"}, ensure_ascii=False, indent=2))
    print("CATS verified:", audit["pair_count"], "pairs,", audit["classification_text_count"], "texts")
