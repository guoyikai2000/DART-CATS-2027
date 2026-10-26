# CATS data card

**CATS (Continual ATtack Stream)**: 8,100 pairs = 16,200 texts in nine domains, 500 / 100 / 300 pairs per domain
for train / valid / test. Every pair joins a human text (label 0) with an AI text on the same source item (label 1).

| Domain | Genre (source) | Attack | Attack model | Test words |
|---|---|---|---|---|
| D00 | News, CC News (MAGA-Bench) | none (original AI text) | – | 231 |
| D01 | Wikipedia (MAGA-Bench) | lexical substitution | GLM-4-9B-0414 | 177 |
| D02 | Scientific abstracts, S2ORC (MAGA-Bench) | syntactic paraphrasing | Moonlight-16B-A3B-Instruct | 81 |
| D03 | ELI5 answers (MDTA) | discourse restructuring | Ministral-3-8B-Instruct-2512 | 97 |
| D04 | wikiHow (MAGA-Bench) | local span rewriting | Yi-1.5-9B-Chat | 30 |
| D05 | Amazon reviews (MAGA-Bench) | full rewriting | Llama-3.2-11B-Vision-Instruct (text only) | 62 |
| D06 | Yahoo Answers (MAGA-Bench) | round-trip translation EN→ZH→EN | DeepSeek (deepseek-flash API) | 52 |
| D07 | Trustpilot reviews (MAGA-Bench) | style transfer toward human authors | Gemma-3-12B-it | 55 |
| D08 | Finance QA (MDTA) | detector-guided rewriting (7 candidates, least AI-like by RADAR) | Qwen3.5-9B | 69 |

*Test words*: mean words per test text; both texts of a pair have the same length.

## Files and fields

`{train,valid,test}/D0k.json` is a JSON list of pairs with the fields

| Field | Meaning |
|---|---|
| `id` | pair id, 1–8100, unique |
| `original_text` | human text (label 0) |
| `ai_generated_text` | original AI text of the source dataset (the positive text of D00) |
| `attack_text` | attacked AI text (the positive text of D01–D08; `null` in D00) |
| `attack_type`, `task_domain` | attack family and domain id |
| `domain`, `model`, `temperature` | source genre, source LLM of the AI text and its sampling temperature |
| `source_id`, `human_group_id` | source item and group of shared sources / near-duplicates (never crosses splits) |
| `split` | `train`, `dev` (the valid folder) or `test` |

`manifest.json` records size, SHA-256 and canonical SHA-256 of every file and the content fingerprint
`6e2e0158…d198f`; `python common/data.py` verifies them.

## Construction (Section 3)

Pairs come from pinned releases of MAGA-Bench and MDTA; sources are grouped across domains so that no source
appears twice. Pairs with technical artifacts (self-disclosed AI identity, delivery phrases, output labels,
leaked instructions, chat or reasoning tags, control characters, Web residue or URL placeholders in the human
text) were replaced by unused pairs of the same domain and source LLM; unchanged attacks were regenerated or
replaced. Both texts of a pair receive the same normalization of whitespace, punctuation and Markdown, and are
truncated to W = ⌊0.9 · min(|h|, |a|)⌋ words. On the AI side only, delivery phrases and disclaimers are deleted
and sentences with templated openings, domain stock phrases, words overused by AI or placeholders are rewritten by
the attack model of the same domain (kept only if numbers, length and meaning are preserved). Human texts are never
reworded, and D00 is not rewritten. Splits are stratified by source LLM; shared sources, duplicates and
near-duplicates (word 5-gram Jaccard ≥ 0.5 or containment ≥ 0.8) stay within one split.

## Known limitations

* D04: in 160 of its 900 pairs the truncated window lies outside the rewritten span, so the visible AI text is
  unattacked.
* D03 is attacked with Ministral-3-8B-Instruct, the backbone of the continual detectors and of DART's writer.
* Each domain changes the attack together with the genre and the mix of source LLMs.
* Licenses: CATS inherits the licenses and terms of MAGA-Bench and MDTA; use for research on detection only.
