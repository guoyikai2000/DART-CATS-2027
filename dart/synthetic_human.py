"""Ablation "w/o real human anchors" (Table 3): the human side of every successful replay pair is replaced
by a human-like continuation of the same opening, written by the same frozen writer G.

The AI-side candidates already exist when this runs. The extra generation uses an isolated RNG stream
(fork_rng + fixed reseeding) and restores the Python and NumPy RNG states afterwards, so everything
else in the stage is unchanged.
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import text_utils as T
import textproc as P
from memory import clean_continuation

MAX_ATTEMPTS = 16
PROMPT = '''Write a continuation containing {n} NEW words, counting only the words you add.
The opening is context only: none of its words count toward the continuation length.
Use the voice of an ordinary human author and match the opening's genre and register.
Keep writing until the new text reaches about {n} words, even if this requires another sentence.
Then stop. The continuation should contain between {lo} and {hi} whitespace-separated words.

Opening:
<<<
{opening}
>>>

Output only the newly written continuation. Do not repeat the opening, add a preface,
label, closing note, or refer to AI or language models.'''


def synthesize_one(stage, i):
    pair = stage.pairs[i]
    mcfg, wcfg = stage.mcfg, stage.mcfg["writer"]
    count = pair["h_words"]
    n, lo, hi = T.target_length(count, float(mcfg["mirror_overshoot"]))
    prompt = PROMPT.format(opening=" ".join(pair["h_vis"].split()[:int(mcfg["opening_words"])]), n=n, lo=lo, hi=hi)
    attempts = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        meta = {"domain_step": "S3 synthetic human side", "anchor_pair": pair["pair_id"], "attempt": attempt}
        r = stage.writer.chat(prompt, float(wcfg["temperature"]), int(wcfg["max_new_tokens"]))
        text = clean_continuation((r["content"] or "").strip())
        errors = {}
        if not text:
            errors["empty_output"] = r.get("error") or "empty"
        else:
            if T.boilerplate(text):
                errors["opening_boilerplate"] = T.boilerplate(text)
            if T.shared_ngram(text, pair["h_ngrams"]):
                errors["ngram8_with_anchor"] = T.shared_ngram(text, pair["h_ngrams"])
            lower = mcfg.get("length_check", "lower") == "lower" and .8 * n <= T.words(text) <= 3 * n
            if not T.length_ok(text, n) and not lower:
                errors["length"] = {"words": T.words(text), "target": n}
        if not errors:
            cleaned = P.normalize(P.clean_ai(P.fmt(text)))
            if T.words(cleaned) <= count:
                errors["too_short_for_trim"] = {"words": T.words(cleaned), "anchor_words": count}
            else:
                text = P.cut(cleaned, count)
                if T.shared_ngram(text, pair["h_ngrams"]):
                    errors["ngram8_after_trim"] = T.shared_ngram(text, pair["h_ngrams"])
        attempts.append({"attempt": attempt, "rid": r["rid"], "failed_checks": errors,
                         "target_words": n, "output_words": T.words(text)})
        stage.analyst.log({"purpose": "synthetic_human", "meta": meta, "prompt": prompt, "output": r["content"],
                           "rid": r["rid"], "usage": r["usage"], "finish_reason": r["finish_reason"],
                           "failed_checks": errors})
        if not errors:
            return {"anchor_index": i, "anchor_pair": pair["pair_id"], "text": text,
                    "label": 0, "words": T.words(text), "attempts": attempts}
    raise RuntimeError("Synthetic-human checks exhausted for pair %s: %s" % (pair["pair_id"], attempts))


def replay_humans(stage, replay):
    """Returns {anchor index: feature dict with label 0} for every replay pair."""
    if not replay:
        return {}
    stage.status.set(phase="epoch_boundary_synthetic_humans")
    indices = [i for i, _, _, _ in replay]
    assert len(indices) == len(set(indices))
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            seed = int(stage.stage_seed) * 1009 + 97
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with ThreadPoolExecutor(max_workers=stage.workers) as pool:
                rows = list(pool.map(lambda i: synthesize_one(stage, i), indices))
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    enc = stage.scorer.encode([r["text"] for r in rows])
    features = {r["anchor_index"]: {"input_ids": ids, "attention_mask": mask, "labels": 0}
                for r, ids, mask in zip(rows, enc["input_ids"], enc["attention_mask"])}
    with (stage.stage_dir / "synthetic_human_replay.json").open("w", encoding="utf-8") as f:
        json.dump({"protocol": "frozen_writer_new_continuation_only", "max_attempts": MAX_ATTEMPTS,
                   "n_replay": len(replay), "n_synthetic_humans": len(features),
                   "real_human_fallback": 0, "rng_restored": True, "rows": rows}, f, ensure_ascii=False, indent=2)
    assert len(features) == len(replay) and all(r["words"] == stage.pairs[r["anchor_index"]]["h_words"] for r in rows)
    return features
