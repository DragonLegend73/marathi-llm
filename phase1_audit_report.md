# Marathi-LLM Phase 1 — Consolidated Audit Report

Scope: the pasted "Evaluation Complete / Phase 1 Complete" summary, checked against the code, data
artifacts and run metrics you uploaded. Date of review: 2026-11-17.

**Bottom line:** the headline numbers in your report match `metrics.json` and `summary.md`, so the
arithmetic is fine. But several *interpretations* do not hold, the data pipeline has a tokenization bug that
makes decontamination discard most of what it flags for the wrong reason, the decontamination index is much
narrower than documented, and the base-vs-v0 comparison is confounded (different code commits, different
prompt formats, IFEval missing for v0). I would not mark Phase 1 "formally complete" until the gates in
section 10 pass.

---

## 1. What was reviewed

| Source | Reviewed |
|---|---|
| GitHub repo page | Fetched 3 times; always returned the legacy flat layout (6 commits, no `src/`, `docs/`, `results/`). Identical request ID each time, so likely a cached copy. Commits/branches pages are blocked by robots.txt. Superseded by the files below. |
| Docs | `EVAL.md`, `DATA_CARD.md`, `summary.md`, `s0_alpaca.stats.json`, `phase1_review_changes.md`, `plan*.md` |
| Data | `s0_alpaca_12k.jsonl` (12,000), `s0_alpaca_dev500.jsonl` (500) |
| Data code | `load.py`, `clean.py`, `dedup.py`, `decontaminate.py`, `build.py`, `stats.py` |
| Eval code | `generate.py`, `runner.py`, `milu.py`, `flores.py`, `bpc.py`, `auto_metrics.py`, `ifeval.py`, `xquad.py`, `fertility.py`, `stats` (eval), `aggregate.py` (read to ~line 400 of 524; the rest is plotting) |
| Runs | `metrics.json` (base), `metrics_1.json` (v0), legacy `train.py` |

**Not reviewed (please upload):** `src/marathi_llm/train.py` (the new trainer), `config.py`,
`configs/data/*.yaml`, `tests/test_data_pipeline.py`, and the per-run record files
(`flores_en_mr_records.json`, `generations.jsonl`).

---

## 2. Claim-by-claim check of the pasted report

| Claim in report | Status | Notes |
|---|---|---|
| MILU 27.16 → 34.94 (+7.78 pp, CI [4.07, 11.36]) | Numbers verified | Matches `metrics.json`. Interpretation is weak: see 4.5. |
| Flores en→mr 9.34 → 14.40; mr→en 25.87 → 29.19 | Numbers verified | Corpus chrF++ matches. Paired CIs use only 50 bootstrap resamples (4.4). |
| BPC 2.96 → 2.29, Δ −0.67 | Numbers verified, statistic mixed | 2.9586 / 2.2872 are corpus BPC. The Δ CI [−0.69, −0.66] belongs to the sentence-mean Δ of −0.679 (4.3). |
| "Perplexity (on BPC)" 8.46 → 5.21 | Mislabelled | It is token-level perplexity `exp(NLL/token)`, not derived from BPC. |
| Script purity 90.77 → 100.00 | Verified | v0 is 99.999%, not exactly 100 (one Latin char in 997 outputs). Earlier suspicion of a bug is withdrawn. |
| Language-ID 63.39 → 97.29 | Verified, caveat | Lingua is restricted to {Marathi, Hindi, English}, which flatters results. |
| English leakage 8.95% → 0.00% | Verified | |
| EOS rate 37.41 → 67.40 ("learns natural stopping") | Numbers verified, interpretation wrong | On single-sentence translation, 33% of v0 outputs still ran to the 256-token cap (4.6). |
| Repeated-4gram 54.06 → 28.79 | Numbers verified | Still 29% of translations loop. "Significant reduction" is true; "solved" would not be. |
| Mean output length 198.9 → 151.5 | Verified | Task-specific (translation), not general generation. |
| "MILU MCQ bias resolved" | Not supported | v0 predicts A 49%, B 38%: A+B = 87%. Only 47% of probability mass is on option letters (4.5). |
| "Genuine out-of-distribution gains; Flores and MILU completely disjoint" | Unproven | v0 was trained on the full undecontaminated dataset (6). Needs the audit in section 9. |
| "Ranks #1 across all criteria, mean rank 1.00" | Misleading | Only the 3 primary criteria; IFEval is missing for v0; a 2-run ranking is trivial (4.7). |
| Decontamination "indexing 8,038 strings (MILU, Flores-IN, XQuAD-IN, IndicIFEval)" | Not true | The 8,038 are MILU validation + Flores dev only (3.2). |
| "11,378 contaminated rows purged, zero leakage" | Not supported | Most are very likely false positives, and the index is incomplete (3.1, 3.2). |
| Funnel counts (51,973 → 47,731 → 47,730 → 36,352) | Verified | Arithmetic checks out. 3,737 + 889 = 4,626 MILU. |
| "25 tests passed" | Cannot assess | Tests not provided; see 8 for a regression test that would have caught 3.1. |
| "Phase 1 formally complete" | Not yet | See acceptance gates in section 10. |

---

## 3. Data pipeline findings

### 3.1 [P0] Decontamination and dedup fragment Devanagari words
`decontaminate.py` and `dedup.py` both use `re.compile(r"\w+")`. In Python, `\w` does not match Devanagari
vowel signs (matras) or the virama, so one word becomes several fragments:

```
"भारताची राजधानी कोणती आहे"  ->  4 words, but 9 fragments with \w+
```

So an "8-word" n-gram is really about 3–4 words. Measured on your 12k train set (rows sharing an 8-gram with
at least one other row):

| Tokenization | Rows sharing an 8-gram |
|---|---|
| `\w+` (current) | 10,403 / 12,000 (86.7%) |
| Whitespace words | 275 / 12,000 (2.3%) |
| Proposed pattern (section 8.1) | 305 / 12,000 (2.5%) |

Consequences:
- Most of the 11,378 rejections (6,752 "Flores", 4,626 MILU) are very likely false positives. True overlaps
  are still caught, but about 24% of valid data is lost and the DATA_CARD's "8-word" claim is false.
- Dedup shingles (5 fragments ≈ 2 words) are also not what the card says.
- Because the set will change after the fix, S0 (and S1–S3) must be rebuilt with the same fixed pipeline for
  the Stage B equal-budget comparison.

### 3.2 [P0] The decontamination index is far narrower than documented
`8,038 = 810 × 5 + 997 × 4`: MILU validation (questions + 4 options) plus Flores dev (both directions,
source + target). Nothing else is indexed.

- Missing despite DATA_CARD and your report: XQuAD-IN, IndicIFEval, Aya held-outs, MILU test.
- The S0 test/dev ID block in `load_all_eval_targets` reads the JSON and does nothing with it.
- Tag attribution is last-writer-wins: Flores `en_mr` and `mr_en` share the same Marathi sentences, so
  "6,752 flores_mr_en and 0 flores_en_mr" says nothing about direction.

### 3.3 [P0] Eval strings are not normalized like training rows
Training rows go through `normalize_marathi_text` (NFC, zero-width stripped). Eval strings in
`add_eval_text` get only `.strip()`. Marathi text often contains ZWJ/ZWNJ; `\w` splits on them, so the same
sentence can tokenize differently on the two sides. That can produce **missed** true overlaps (false
negatives). Both sides must use one shared normalize + tokenize helper.

### 3.4 [P1] The max-sequence-length filter never ran
`build.py main()` calls `build_dataset_pipeline(...)` without a tokenizer, so `validate_example` skips the
token-length check (`failed_max_seq_len: 0`). The built train set has rows up to 2,569 characters (19 rows
over 1,500; 6 over 2,000). Pass the tokenizer and count tokens **with the training template and EOS
included**, since the Marathi preamble adds tokens the current check ignores.

### 3.5 [P1] Aya held-out data is only half wired
`build.py` saves `native_prompts.jsonl` but never saves `bpc_eval_responses`, although DATA_CARD says 500 BPC
responses are held out and indexed. In practice the BPC evaluation uses Flores, not Aya, so EVAL.md §2/§3.4
and the code disagree.

### 3.6 [P2] Loader and cleaning details
- `load.py` falls back to the English `instruction/output` columns when `translated_text` lacks markers.
  The 674 rows with 0% purity are those; the purity filter catches them, but dropping them at load is cleaner.
- 29 rows are dropped in `load.py` for empty extraction, which explains 51,973 vs the README's 52,002.
- Placeholder inputs survive in the 12k set: 47 × "इनपुट नाही", 23 × "Noinput", 9 × "इनपुट नाही.". Blank them
  in `load.py` so the model does not learn a literal "no input".
- 3 instructions appear twice in the 12k set (dedup is on instruction + response, so this is by design;
  decide whether you want it).
- `build.py` progress bar total is hard-coded to 200,000 (cosmetic).

---

## 4. Evaluation harness findings

### 4.1 Verified fixed (from the earlier review)
- Flores records carry both `output` and `generation`, so the empty-string metric bug is fixed.
- `load_model_and_tokenizer` raises if the adapter fails to attach; `adapter_loaded: true` for v0.
- MILU uses `logits_to_keep=1` with a PEFT-safe fallback.

### 4.2 [P1] The two runs used different code commits
Base: `2e12a67…`. v0: `8a4b758…`. The harness changed between runs, so every delta mixes model and code
changes. Also, only `git rev-parse HEAD` is stored, with no dirty-tree flag.

### 4.3 [P1] BPC: mixed statistic and an odd BOS
- Overview table: corpus BPC with a sentence-mean CI, so 2.9586 sits outside [2.9964, 3.0486].
- Paired table: sentence-mean BPC (3.0221 vs 2.3430), Δ −0.679.
- Report one statistic: corpus BPC with a bootstrap over per-sentence (NLL, chars) sums (code in 8.4).
- Qwen has no `bos_token`, so BPC falls back to `eos_token_id = 151645 = <|im_end|>`. `<|endoftext|>`
  (151643) is the natural document start. Affects both runs, but could affect the adapter differently.

### 4.4 [P1] Paired Flores CIs are coarse and mix statistics
`aggregate.py` uses `min(50, n_bootstraps)` resamples for the corpus-chrF++ CI, so the 2.5/97.5 percentiles
come from 50 samples. The p-value is from a sentence-level test while the CI is corpus-level. "p = 0.0000"
with 1,000 resamples means p < 0.001; print it that way.

### 4.5 [P1] MILU interpretation and confounds
- Base predicts A 98.5% of the time; its 27.16% is essentially the share of A-labelled answers.
- v0: A 49.0, B 38.3, C 7.4, D 5.3. That is still heavily skewed, and only 47% of probability mass is on
  A–D (base 17%). So "balanced" is incorrect; "less skewed" is accurate.
- Base ran under `chat`, v0 under `alpaca_v0`. Model gain and template gain are not separable.
- Per-domain: Health & Medicine is below chance for both models (base 12.5%, v0 17.5%), which is another
  sign the scoring is dominated by letter bias. Per-domain n is not stored.
- Prompt uses `A.` where EVAL.md specifies `A)`.

### 4.6 [P1] Auto-metrics are translation-task statistics
`runner.py` passes only en→mr records to `evaluate_generations`, whereas EVAL.md §3.4 says Aya held-out
prompts. From `metrics.json`: base EOS 37.4% / mean 198.9 tokens, v0 EOS 67.4% / mean 151.5. Working back,
the outputs that stop do so at roughly 100 tokens, so about 63% of base and 33% of v0 translations ran to the
256 cap. Combined with 28.8% of v0 outputs containing a 4-gram repeated ≥ 4 times, a meaningful share of
v0 translations degenerate. chrF++ is penalised by trailing text, so v0's gain is measured on top of that.

Related risks:
- With `--force` and an existing `flores_en_mr_records.json`, old records are loaded and then re-extended
  (double counting). `n_samples` is not saved in `metrics["auto"]`, so it is not visible. It should be 997.
- Token budgets may bite: perplexity and BPC imply about one token per character for Marathi in Qwen, so
  32 new tokens (XQuAD) is about 30 characters, and 256 tokens can truncate long Flores sentences. Confirm
  with `fertility.py`.

### 4.7 [P1] IFEval is missing for v0 and is scored as 0
v0's `metrics.json` has no `instruction_following` key at all (not an error; it was not run, or was skipped).
`aggregate.py` defaults a missing value to `0.0`, so the ranking table shows 0.00% while the overview shows
"—". Also: lm-eval is invoked with `--apply_chat_template` regardless of `prompt_format`, so v0 would be
scored under chat rather than its native Alpaca template, and `HF_HUB_OFFLINE=1` is the default, so a Hub
adapter ID may fail if not cached. Base IFEval for reference: ground strict 8.57%, trans strict 15.10%.

### 4.8 [P2] Smaller
- Language-ID uses a 3-language Lingua detector; use a wider candidate set for a fairer figure.
- `fertility.py` default tokenizers (Qwen2.5, Sarvam-1, Gemma-2) don't match the plan's list, and it reads
  the legacy `s0_dev_ids.json`, not the new dev500.
- `perplexity` is token-level; label it that way and never compare it across tokenizers.

---

## 5. Run metrics (observed values)

| Metric | Base (`chat`) | v0 (`alpaca_v0`) |
|---|---|---|
| git commit | 2e12a67 | 8a4b758 |
| adapter_loaded | false | true |
| MILU acc | 27.16 [24.20, 30.37] | 34.94 [31.85, 38.15] |
| MILU option mass | 0.173 | 0.470 |
| MILU letters A/B/C/D | 98.5 / 0.1 / 0.7 / 0.6 | 49.0 / 38.3 / 7.4 / 5.3 |
| Flores en→mr (corpus / mean-sentence) | 9.34 / 9.34 | 14.40 / 14.61 |
| Flores mr→en (corpus / mean-sentence) | 25.87 / 26.49 | 29.19 / 30.53 |
| BPC (corpus) / BPB / token ppl | 2.9586 / 1.1079 / 8.46 | 2.2872 / 0.8565 / 5.21 |
| script purity | 90.77 | 99.999 |
| lang-ID (mr) | 63.39 | 97.29 |
| repeat-4gram | 54.06 | 28.79 |
| EOS rate | 37.41 | 67.40 |
| mean len (tokens) | 198.9 | 151.5 |
| IFEval ground / trans (prompt strict) | 8.57 / 15.10 | not run |

---

## 6. Training (`train.py`, legacy) and what it means for v0

- Trains on `translated_text + EOS` from the **full raw** `rachittshah/alpaca-marahti` train split. No
  cleaning, no dedup and no decontamination; only a token-length filter (`MAX_SEQ_LEN`).
- Loss is over the whole sequence (preamble + instruction + response), since `SFTTrainer` is used with
  `dataset_text_field` and no completion-only masking. Fine for v0, but check the new `train.py` is
  consistent, otherwise Stage B comparisons are confounded.
- The 5% eval split is random from the same pool, so eval loss is in-distribution only.
- fp16 training; the eval harness picks bf16 when available.
- **Implication:** v0 saw every Alpaca row, including any that genuinely overlap Flores or MILU. If the
  6,752 + 4,626 flagged rows are mostly false positives (3.1) this is harmless, but that must be shown, not
  assumed. Run the audit in section 9 on exactly the rows v0 trained on.
- Also confirm the eval-time preamble in `generate.py` (`ALPACAV0_PROMPT_*`) is character-for-character the
  preamble in the dataset's `translated_text` (add a one-row assertion test).

---

## 7. Corrections to my own earlier remarks

| Earlier statement | Now |
|---|---|
| "v0's 100.00% purity looks too clean" | Withdrawn. It is 99.999% with one Latin character; metrics are computed on real text. |
| "`"output"` vs `"generation"` key bug" | Fixed in the code you uploaded. |
| "Hub adapter ID silently ignored" | Fixed: load now raises, and `adapter_loaded` is recorded. |
| "v0 IFEval probably errored" | No: the key is absent, i.e. not run. |
| "51,973 vs 52,002 discrepancy" | Resolved: 29 rows dropped at load. |

---

## 8. Fixes (copy-paste starting points)

### 8.1 Shared text helper (fixes 3.1 and 3.3)
`src/marathi_llm/data/text.py`:
```python
import re, unicodedata

ZW = re.compile(r"[\u200B-\u200D\uFEFF]")
# word chars + all Devanagari letters/matras/virama/digits, excluding danda U+0964-0965
WORD_PATTERN = re.compile(r"[\w\u0900-\u0963\u0966-\u097F]+")

def normalize_for_match(text: str) -> str:
    return ZW.sub("", unicodedata.normalize("NFC", text)).lower()

def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(normalize_for_match(text))
```
In `decontaminate.py` and `dedup.py`, replace every `WORD_PATTERN.findall(text.lower())` with
`words(text)`, and call `words()` on eval strings in `add_eval_text`. Tested: it returns
`['भारताची','राजधानी','कोणती','आहे']` for the example above and gives 305/12,000 shared-8-gram rows.

### 8.2 Complete the index (fixes 3.2)
Add loaders in `load_all_eval_targets` for: MILU `test` split (questions + options), XQuAD-IN dev and test
(context, question, answer strings), IndicIFEval `trans` and `ground` Marathi prompts, and the Aya held-outs
(`data/eval/native_prompts.jsonl` prompt + reference, plus the saved BPC responses). Index with a per-benchmark
tag and **keep a list of tags per n-gram** instead of one (`Dict[str, Set[str]]`). Update DATA_CARD counts to
the actual indexed total, or trim the card.

### 8.3 Pass a tokenizer in `build.py` (fixes 3.4)
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
build_dataset_pipeline(config_path=args.config, output_dir=args.output_dir,
                       eval_decontaminator=decontaminator, tokenizer=tok)
```
and in `validate_example` count `PREAMBLE + instruction + input + response + tok.eos_token` rather than the
bare triple. Also save `bpc_eval_responses` and, ideally, assert `failed_max_seq_len > 0` in a smoke test.

### 8.4 BPC statistic and BOS (fixes 4.3)
```python
def pick_bos(tok):
    if tok.bos_token_id is not None:
        return tok.bos_token_id
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    return eot if eot not in (None, tok.unk_token_id) else tok.eos_token_id

def bootstrap_corpus_bpc(nll, chars, n=2000, seed=42):
    import numpy as np
    nll, chars = np.asarray(nll), np.asarray(chars)
    idx = np.random.default_rng(seed).integers(0, len(nll), (n, len(nll)))
    boots = nll[idx].sum(1) / (np.log(2) * chars[idx].sum(1))
    return boots.mean(), np.percentile(boots, [2.5, 97.5])
```
For the paired version, reuse the **same** `idx` for both runs and take the difference of the two corpus BPCs.

### 8.5 `aggregate.py` (fixes 4.4 and 4.7)
- Use 1,000 resamples (cache per-sentence results) for paired corpus chrF++.
- Use one test family per row (corpus CI with a corpus-level p-value).
- Print `p < 0.001` instead of `p=0.0000`.
- Treat missing IFEval as `None`: show "—" and skip it in tie-breaks, never as 0.

### 8.6 `runner.py` (fixes 4.2, 4.6)
- Record `git_dirty` (`git status --porcelain`) and refuse to compare runs across commits without a flag.
- Store `n_samples` for the auto metrics, and clear `all_generations` before a forced Flores rerun.
- Generate on the Aya held-out prompts for the general-generation metrics (spec §3.4), and keep the Flores
  translation stats separate.
- Make `prompt_format` apply to IFEval, or document that IFEval always uses the chat template.

### 8.7 Regression tests to add
```python
def test_devanagari_words_not_fragmented():
    from marathi_llm.data.text import words
    assert words("भारताची राजधानी कोणती आहे") == ["भारताची", "राजधानी", "कोणती", "आहे"]

def test_three_shared_words_do_not_trigger_8gram():
    d = BenchmarkDecontaminator()
    d.add_eval_text("भारताची राजधानी कोणती आहे आणि ती कुठे आहे हे सांगा", "t")
    assert d.check_contamination("भारताची राजधानी कोणती आहे पण उत्तर वेगळे आहे")[0] is False

def test_zwj_normalization_matches():
    ...  # same sentence with and without ZWJ/ZWNJ must match
```

---

## 9. Audit of the rows v0 actually trained on

Run this **after** applying 8.1 and 8.2 (needs the Flores and MILU files in the local HF cache). It reports
how many of v0's training rows overlap each benchmark at several n-gram sizes; real overlaps survive larger
n, false positives vanish.

```python
# scripts/audit_v0_contamination.py
import json, collections
from datasets import load_dataset
from marathi_llm.data.decontaminate import BenchmarkDecontaminator, load_all_eval_targets

SIZES = [8, 12, 20]
ds = load_dataset("rachittshah/alpaca-marahti", split="train")
report = {}
for n in SIZES:
    dec = BenchmarkDecontaminator(ngram_size=n)
    load_all_eval_targets(dec)
    hits, samples = collections.Counter(), []
    for i, row in enumerate(ds):
        text = (row.get("translated_text") or "")      # exactly what v0 trained on
        bad, tag = dec.check_contamination(text)
        if bad:
            hits[tag] += 1
            if len(samples) < 20:
                samples.append({"row": i, "tag": tag, "text": text[:200]})
    report[n] = {"total_rows": len(ds), "hits": dict(hits), "samples": samples}
    print(n, sum(hits.values()), dict(hits))
json.dump(report, open("results/audit_v0_contamination.json", "w"), ensure_ascii=False, indent=2)
```
Read the 20 samples per size by hand. Decision rule: if the n=12 and n=20 counts are near zero, v0's
Flores/MILU gains stand as out-of-distribution; if not, re-evaluate those rows excluded and re-report.

---

## 10. Re-run plan and Phase 1 acceptance gates

1. Apply 8.1–8.3, rebuild S0 (and S1–S3), regenerate `*.stats.json`, and update DATA_CARD with the real
   index size and counts.
2. Run the section 9 audit on v0's training rows and record the result in `docs/`.
3. Re-run **base and v0 at the same clean commit** (`git_dirty: false`).
4. Add control runs: base under `alpaca_v0`, v0 under `chat`, so format and model effects are separable.
5. Add a debiased MILU score (option-order permutation) alongside raw accuracy, with per-domain n.
6. Run IFEval for v0 (record which template it used).
7. Switch BPC to corpus BPC with `<|endoftext|>` BOS; report one statistic with a matching CI.
8. Report Flores paired CIs from ≥ 1,000 resamples; format p-values as `< 0.001`.
9. Report generation-quality metrics on the Aya held-out prompts, and show the share of translations that hit
   the token cap.

**Gates to call Phase 1 complete:** decontamination covers all benchmarks listed in the DATA_CARD; the
regression tests above pass; no base-vs-v0 comparison crosses commits or prompt formats without a control;
the v0 contamination audit is recorded; and every number in the report is traceable to one statistic and one
commit.

---

## 11. Open questions

- Does `src/marathi_llm/train.py` mask the prompt (completion-only loss) and match the v0 setup?
- Was v0 trained on all `translated_text` rows ≤ `MAX_SEQ_LEN`, and what is `MAX_SEQ_LEN` in `config.py`?
- Was v0 IFEval intentionally skipped, or did the run use an older runner?
- What do `tests/test_data_pipeline.py` and the `configs/data/*.yaml` files contain (the decontamination test
  almost certainly used text where `\w+` happened to work)?
- Can you share a few raw Flores en→mr records for v0 with `finish_reason: length`, to see what the
  runaway outputs look like?
