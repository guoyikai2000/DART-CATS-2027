"""Text utilities for DART: the detector-visible part of a text, program statistics shown to the
analyst, the checks applied to generated texts, and the 8-gram overlap check."""

import math
import re
import statistics
import string

_PUNCT = string.punctuation + "“”‘’«»—–…·•"
_BLANK = re.compile(r"\n\s*\n")
_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])")
_SENT = re.compile(r"[.!?]")
# a preface at the start of a generated text, or self-reference / placeholders anywhere
_OPENING = re.compile(
    r"^(?:okay|ok|sure|certainly|of course|absolutely|here is|here['’]s|here are|below is|"
    r"as an ai|i['’]d be happy|i would be happy)\b", re.IGNORECASE)
_ANYWHERE = re.compile(r"\bas an ai\b|\blanguage model\b|\[(?:insert|your)\b[^\]]{0,60}\]", re.IGNORECASE)


def words(text: str) -> int:
    """Whitespace-separated word count."""
    return len(text.split())


def visible_text(text: str, tokenizer, budget: int):
    """Characters of text.strip() covered by the first `budget` body tokens (same tokenisation as
    data.fixed_prompt_encoder). Returns (visible_text, exact) where exact tells whether re-encoding
    the prefix gives back exactly the retained body tokens."""
    body = text.strip()
    enc = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) <= budget:
        return body, True
    end = enc["offset_mapping"][budget - 1][1]
    prefix = body[:end]
    again = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    return prefix, again == ids[:budget]


def stats(text: str) -> dict:
    """Program statistics of a visible text, shown to the analyst with every pair."""
    if _BLANK.search(text):
        paragraphs = [p for p in _BLANK.split(text) if p.strip()]
    else:
        paragraphs = [line for line in text.split("\n") if line.strip()]
    lines = text.split("\n")
    lengths = [len(s.split()) for s in _SENT.split(text) if s.split()]
    return {
        "words": words(text),
        "paragraphs": len(paragraphs),
        "list": any(_LIST.match(line) for line in lines),
        "heading": any(line.lstrip().startswith("#") for line in lines),
        "sent_mean": statistics.fmean(lengths) if lengths else 0.0,
        "sent_sd": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
    }


def stats_line(label: str, st: dict) -> str:
    return (f"stats {label}: words={st['words']}, paragraphs={st['paragraphs']}, "
            f"list={'yes' if st['list'] else 'no'}, heading={'yes' if st['heading'] else 'no'}, "
            f"sent_mean={st['sent_mean']:.1f}, sent_sd={st['sent_sd']:.1f}")


def boilerplate(text: str):
    """Template opening or AI self-disclosure. Returns the matched phrase or None."""
    t = text.strip()
    m = _OPENING.match(t) or _ANYWHERE.search(t)
    return m.group(0) if m else None


def norm_tokens(text: str) -> list:
    out = []
    for tok in text.lower().split():
        tok = tok.strip(_PUNCT)
        if tok:
            out.append(tok)
    return out


def ngrams(text: str, n: int = 8) -> set:
    toks = norm_tokens(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def shared_ngram(text: str, reference: set, n: int = 8):
    """First n-gram of `text` that also occurs in `reference` (a set from ngrams()), else None."""
    toks = norm_tokens(text)
    for i in range(len(toks) - n + 1):
        gram = tuple(toks[i:i + n])
        if gram in reference:
            return " ".join(gram)
    return None


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def target_length(anchor_words: int, ratio: float):
    """N = round(anchor words x length ratio); the prompt asks for 0.9N-1.1N words."""
    n = max(1, round_half_up(anchor_words * ratio))
    return n, round_half_up(0.9 * n), round_half_up(1.1 * n)


def length_ok(text: str, n: int) -> bool:
    """Strict length check: fail when the length deviates from the target by more than 20%."""
    return abs(words(text) - n) / n <= 0.2
