# Phase 1 Review: Consolidated Change List

Covers (A) everything from the previous review and (B) new findings from the Phase 1 modules
(`runner.py`, `milu.py`, `ifeval.py`, `xquad.py`, `bpc.py`, `flores.py`, `fertility.py`,
`auto_metrics.py`, `stats.py`). `generate.py` and `train.py` were not provided, so anything that
depends on them is marked "unverified".

Priority: **P0** = changes reported numbers or breaks a run, fix before any number is published.
**P1** = fix before Stage A. **P2** = cleanup and consistency.

---

## 0. Repository status

- [ ] **Confirm the push reached GitHub.** A fresh fetch of `github.com/DragonLegend73/marathi-llm`
      still showed 6 commits, the old README (+10.44 headline, "no validation overfitting"), and only
      the legacy scripts at the root. No `src/`, `docs/`, `tests/`, `data/` or `results/` were visible.
      The commits page could not be fetched (robots disallowed), so this may be stale. Check locally:
      ```
      git status -sb
      git ls-remote origin refs/heads/main
      git branch -r --contains 9acdc5b
      ```
      If the hash is missing from `origin/main`, the push went to another branch or remote.
- [ ] The public README still carries the claims the audit disproves. Update it (see section 5).

---

## 1. P0: fix before any reported number

### A1. Marathi vowel signs are dropped (verified)
Checked in Python: `re.sub(r'[^\w\s]', '', 'भारताची राजधानी')` gives `भरतच रजधन`, and `isalpha()` also skips
Devanagari combining marks (categories Mc and Mn).
- [ ] `audit_baseline.py::normalize_text`: keep characters whose Unicode category starts with L, M or N, plus whitespace.
- [ ] `rescore_v0.py::compute_script_purity` (legacy): same issue (`isalpha`). Legacy purity 95.88 / 93.69 is not comparable with `auto_metrics.py`, which counts the Devanagari block and handles this correctly.
- [ ] Hash the split on `row_id` plus a salt only (not on normalized instruction text), so manifests do not depend on the normalizer. Then regenerate `data/splits/*`, rerun the audit and the re-score, and update `docs/legacy/` and the README numbers.
- [ ] Add tests: pure Marathi with matras scores about 100% purity; normalizer keeps matras.

### B1. Automatic metrics are computed on empty strings
`flores.py` records store the model text under `"generation"`; `auto_metrics.evaluate_generations` reads `item.get("output", "")`.
Script purity, distinct-4-gram ratio and repetition rate in `metrics.json` are therefore computed on empty text.
- [ ] Use one key (`"output"`) in `generate_batch` records and Flores records, or make `evaluate_generations` read both.
- [ ] Add a test that feeds Flores-style records to `evaluate_generations` and asserts non-zero purity.

### B2. Fertility uses English text for the S0 portion
`fertility.py` reads `ds[idx]["output"]` for S0 dev, and the README states `output` is the original English column (Marathi is in `translated_text`).
About 499 of the 1,496 sentences are therefore probably English, which dilutes tokens/word. The 3.73 figure in `results/fertility.json` is not a Marathi fertility measurement.
- [ ] Extract the Marathi response from `translated_text` (as `audit_baseline.extract_instruction_and_response` does), or use Flores only.
- [ ] Report Flores-only and S0-translated fertility separately, and log the dropped sentence (997 + 500 gives 1,497, not 1,496).
- [ ] Rerun for all candidate tokenizers (Qwen2.5, Qwen3, Gemma-3, Llama-3.2; the default list in `fertility.py` does not match the plan's Stage A list). Llama is gated and needs HF auth.
- [ ] `pct_exceeding_max_len` is always 0 on single sentences; compute it on full S0 examples instead.
- **Correction to my earlier review:** I used 3.73 tokens/word to infer that 256 tokens is about 69 words. Do not rely on that until fertility is recomputed on Marathi-only text.

### B3. BPC is not comparable across tokenizers
`bpc.py` tokenizes with `tokenizer(text)`. Qwen adds no BOS, so the first token of each sentence is never scored, while Llama and Gemma add BOS and score it. Every character is still counted in the denominator, so Qwen BPC is biased low relative to BOS models.
- [ ] Prepend a BOS (or `<|endoftext|>` for Qwen) so every text token is scored; count tokens and characters consistently.
- [ ] `load_marathi_reference_texts` catches all exceptions and returns `[]`, which makes `compute_bpc` return 0.0 silently. Raise instead.
- [ ] Use all 997 Flores-mr dev sentences (the plan's BPC set), not the first 500 by default.
- [ ] Note in the report that Flores may appear in some models' pretraining data.

### B4. MILU fast path likely breaks for adapter runs (unverified)
`milu.py` calls `model.model(**inputs).last_hidden_state` and `model.lm_head`. On a `PeftModel`, attribute lookup is forwarded and `.model` can resolve to the wrapped `*ForCausalLM`, whose output has `.logits` and no `.last_hidden_state`. The smoke test probably ran on the base model only.
- [ ] Run the v0 adapter through `evaluate_milu` to confirm.
- [ ] Replace with `model(**inputs, logits_to_keep=1).logits[:, -1, :]` (works with PEFT and keeps memory low), or unwrap with `get_base_model()`.
- [ ] Record the probability mass on the four letter tokens versus the whole vocabulary, to detect prompts where the model does not answer with a letter.

### B5. Prompt format per system (confirmed by `rescore_v0.py`)
The legacy re-score gave the base model the Alpaca template. The new code applies the chat template, which would be wrong for the v0 adapter.
- [ ] Add a per-system `prompt_format` (`chat` or `alpaca_v0`) to `generate_batch`, MILU, Flores and the runner, and record it in `metrics.json`.
- [ ] Re-score base and v0 through the new runner. That replaces the legacy S0-chrF++ row as the base-vs-v0 comparison.

---

## 2. P1: fix before Stage A

### C1. IndicIFEval wrapper (`ifeval.py`)
- [ ] The runner's `prequantized` flag is never passed to `run_indicifeval`. For non-prequantized models, add `load_in_4bit=True` to `--model_args`; otherwise a full-precision load can OOM or silently differ in quantization from the in-process evals.
- [ ] Hard-coded `.venv/bin/lm_eval` breaks outside the repo root and in CI. Use `sys.executable -m lm_eval` or `shutil.which`.
- [ ] Add `--log_samples` so the hand spot-check of harness verdicts is possible.
- [ ] Missing metric keys silently become 0.0 (`.get(..., 0.0)`). Raise instead, and also save stderr and n.
- [ ] Confirm the effective `max_gen_toks` (EVAL.md says 1280, the plan said 512) from the task yaml or the results JSON, and make both documents agree.
- [ ] Seed or confirm `langdetect` determinism.
- [ ] Runtime: batch size 1 with up to 1,280 new tokens on about 970 prompts is likely several hours per system (my estimate). Use `--limit` for selection runs and the full set for finalists, or test a larger batch size and check determinism.
- [ ] Store results under `trans_mr` and `ground_mr` keys to match Appendix B in the plan, or update the plan.

### C2. Runner (`runner.py`)
- [ ] No CLI flag for `prequantized`; `main()` never passes it, so `quant` is always "prequantized". Add `--prequantized/--no-prequantized`.
- [ ] `metrics.json` is written only at the end. Save after each benchmark so an IFEval crash does not lose earlier results.
- [ ] Record `batch_size`, `prompt_format`, `use_chat_template`, decoding settings, dataset revisions, and the IndicIFEval commit.
- [ ] The `auto` block (purity, repetition, EOS, length) is computed on en→mr translation outputs only. EOS and length there are not comparable with the legacy free-form numbers. Wire in the held-out Aya prompts (100) and the native prompt set from the plan.
- [ ] Refuse to overwrite an existing `run_id` directory unless `--force` is given.
- [ ] Write Flores records with `ensure_ascii=False`.

### C3. MILU details
- [ ] `truncation=True, max_length=1024` with the default right truncation side would cut off the end of a long prompt (including the answer cue). Set `truncation_side="left"` and log how many prompts were truncated.
- [ ] EVAL.md shows the prompt as `A)`; the code uses `A.`. Make them agree, and state that the chat template is applied.
- [ ] Domain names in EVAL.md (STEM, Humanities, Social Sciences, Other) should match the actual `domain` values in the data.

### C4. Flores
- [ ] The reported headline is corpus chrF++ but the CI is the bootstrap of mean sentence chrF++. Report one statistic consistently (bootstrap corpus chrF++ by resampling sentences, or report the sentence-level mean), and use the same one in paired comparisons and the Stage A rule.
- [ ] Models that add preamble ("Here is the translation...") are penalized. Decide on, document, and apply one post-processing rule to all systems.

### C5. Selection-rule machinery
- [ ] Implement the mean-rank-with-ties algorithm from EVAL.md in `aggregate.py` (for example: rank by point estimate, group models whose paired CI against the next one includes zero, assign group-average ranks).
- [ ] `mcnemar_test`: use the exact binomial test when discordant pairs are fewer than about 25.
- [ ] Use 10,000 bootstraps for final reports (1,000 is fine for development).

### C6. Dataset and script fixes from the previous review
- [ ] `inspect_phase1_datasets.py`: pass `config="Marathi"` explicitly (the script looks for `marathi`/`mr` and falls back to `configs[0]`).
- [ ] `audit_baseline.py`: `leak_count` uses only the top-ranked match. Use `any(m["split_status"] == "IN_V0_TRAIN" for m in matches)` (prompt 3's top match was a dropped row).
- [ ] Run the bucket check: intersect `s0_test_200` with `v0_train_ids` and `v0_eval_ids` (already computed in `audit_baseline.py`) and report v0 results by bucket.
- [ ] `rescore_v0.py`: the fallback to the `output` column (English) is wrong for a Marathi eval. Log how many rows use it and confirm `test_prompts_count` is 200.
- [ ] `test_canary.py`: it contains the GUID, so `git grep` will find it once the file is committed. Build the string at runtime or exclude the file with a pathspec. Confirm the GUID matches the `canary` field in the real IndicGenBench files.
- [ ] `chat.py`: imports `config`, which may have moved; `reset` has no history to clear. Fix the import, then implement history or remove the command.
- [ ] `profile_hardware.py`: `warmup_steps=0.05` is treated as steps in older Transformers; use `warmup_ratio` in the new `train.py`.

---

## 3. P2: plan and EVAL.md consistency

- [ ] EVAL.md has an English-regression veto but no English benchmark. Add the plan's English check or remove the veto.
- [ ] State that the held-out Aya slice (100 prompts) is excluded from S3 training data and from the splits, and test it.
- [ ] EVAL.md: say whether the IndicIFEval Trans task needed the patch (only the Ground yaml is mentioned) and whether Trans is filtered to the `correct` rows.
- [ ] Reconcile IFEval generation limit between `plan.md` (512) and EVAL.md (1280).
- [ ] After recomputing fertility (B2), update the plan's tokens-per-word assumption and the 256-token implication.
- [ ] `xquad.py` is not called by the runner (fine, it is finalists-only). Add a `--xquad` flag for finalists, use batch size 1 for long contexts, and note that whitespace-token F1 is harsh on inflected Marathi (suffixes attach to words).

---

## 4. Tests to add

- [ ] Devanagari normalization and purity with matras (A1).
- [ ] `evaluate_generations` on Flores-style records (B1).
- [ ] MILU scorer on a tiny synthetic example, including a PEFT-wrapped model if feasible (B4).
- [ ] BPC consistency: the same text scored with and without BOS handling gives the same character accounting (B3).
- [ ] Normalized-instruction overlap between S0 train, dev and test (the audit claims zero text overlap, but the current test checks IDs only).
- [ ] Held-out Aya prompts do not appear in any training source.

---

## 5. README and documentation

- [ ] Replace the headline (+10.44) and remove "validation loss tracked closely, no overfitting" and "base model produces incoherent output / fine-tuned consistently generates structured prose".
- [ ] Lead with a pitch and status line; present the audit as a section; move the roadmap lower.
- [ ] Use the audit wording agreed earlier: 4 of 10 prompts leaked (2 exact, 1 near-match, 1 contained); +19.73 versus +4.25; the 200-prompt re-score is "legacy, possible overlap with v0 training data".
- [ ] Mark planned directories "(planned)" and update the repo tree to match what exists.
- [ ] Add a LICENSE before publishing anything new.

---

## 6. Suggested order

1. A1 (normalizer, purity, row-id hash), regenerate manifests, rerun audit and re-score.
2. B1 (auto-metrics key), B2 (fertility on Marathi text), B3 (BPC).
3. B4 and B5 (MILU adapter path, per-system prompt format), then run base and v0 through the runner.
4. C1 and C2 (IFEval wrapper, runner flags, incremental saves, Aya wiring).
5. Remaining P1 and P2 items, then README and push verification (section 0).

**Done when:** `metrics.json` exists for base and v0 from one command, a rerun reproduces it exactly, the hand spot-check of IFEval verdicts and MILU letters agrees with the code, and the pushed repo shows the new structure.

---

## 7. `generate.py` findings (added after reviewing `generate.py`)

**Confirmed from this file:** B1 (`generate_batch` returns `"output"`, Flores rewrites it to `"generation"`, and the runner passes the rewritten records to `evaluate_generations`) and B5 (there is no non-chat prompt path, so the v0 adapter cannot be evaluated with its Alpaca training template).

**Looks right:** greedy decoding; explicit `repetition_penalty=1.0` (overrides Qwen's generation_config default of 1.05); left padding; deterministic length sort with a stable tie-break; results restored to input order.

### P0
- [ ] **D1. A Hub adapter ID is silently ignored.** `if adapter_path and os.path.exists(adapter_path)` is false for `DragonLegend/marathi-qwen2.5-lora`, so the model loads as the plain base while the runner records the adapter name. A "v0" run would be base-model numbers. Fix: if an adapter is requested and cannot be loaded, raise. Accept Hub IDs (`PeftModel.from_pretrained` handles them). Record `adapter_loaded: true` from `isinstance(model, PeftModel)`. Add a test for this.
- [ ] **D2. Qwen3 thinking mode is not disabled.** The comment says it is, but `apply_chat_template` is called without `enable_thinking=False`, so Qwen3 will spend its 256 tokens on `<think>` output (and break MILU, Flores and XQuAD). Pass `enable_thinking=False` (extra template kwargs are ignored by templates that don't use them) and record it.

### P1
- [ ] **D3. Quantization loading.**
  - `load_in_4bit=True` as a `from_pretrained` kwarg is deprecated and likely rejected in Transformers 5.x (unverified). Use `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=..., bnb_4bit_use_double_quant=...)`, matching how the pre-quantized Unsloth checkpoints were made, or you add a quantization confound.
  - Derive `prequantized` from `AutoConfig.from_pretrained(model_id).quantization_config` instead of a flag defaulting to `True`; a full-precision 1.7B-2B repo loaded as "prequantized" will likely OOM on 4 GB. Record the resolved setting in `metrics.json` and pass it to the IFEval wrapper (C1).
- [ ] **D4. EOS and finish-reason logic.**
  - Only `tokenizer.eos_token_id` is checked. Qwen's `generation_config` stops on several ids (for example `<|im_end|>` and `<|endoftext|>`). Use `model.generation_config.eos_token_id` (list-aware) when locating the stop position.
  - `pad_token_id=eos_id` makes padding indistinguishable from EOS. A sequence that stops on a different EOS id gets `n_new_tokens` inflated up to the next `eos_id`, and can be labelled `"length"` when another sequence in the same batch ran to `max_new_tokens`. This feeds `eos_rate` and mean length.
  - Add a test with a fake model or tokens covering: stop on id A, stop on id B, and hit the length limit, inside one batch.
- [ ] **D5. Silent truncation.** `truncation=True, max_length=2048` with the default right-side truncation would cut the end of a long prompt (including the generation prompt). Set `truncation_side="left"` or raise, and log the number of truncated prompts. The same applies to MILU's 1024 limit (C3).
- [ ] **D6. Add `prompt_format`** (`chat` or `alpaca_v0`) as in B5, applied in `prepare_prompt`, and pass it through MILU, Flores, XQuAD and the runner.

### P2
- [ ] **D7. Smaller items.**
  - `trust_remote_code=True` is hard-coded; default to `False` (the model families in the plan are supported natively) and enable it per model only if needed.
  - `input_lengths` is computed but unused.
  - Sorting by character length of the prompt text: sort by token length to reduce padding.
  - Set `temperature=None, top_p=None, top_k=None` explicitly to avoid greedy-mode warnings from the model's generation_config.
  - Set `torch.manual_seed` and record `batch_size`, `max_new_tokens`, `prompt_format`, `enable_thinking`, and the library versions in every run's `metrics.json` (batched 4-bit greedy decoding can differ slightly with batch composition).
  - The default Qwen system prompt is injected automatically when none is given; this is uniform across Qwen runs but differs across model families, so document it in EVAL.md.
  - Gemma's chat template rejects a `system` role; handle it if a system prompt is ever used.

### Updated order of work
Put D1 first (nothing involving the v0 adapter is trustworthy until it is fixed), then A1, B1, B2, B3, B4, B5/D6, D2, and D3 to D5 before the first Stage A run.
