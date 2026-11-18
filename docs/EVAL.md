# Evaluation Harness Specification (`docs/EVAL.md`)

This document specifies the benchmarks, scoring methodologies, and pre-registered selection protocols for the Marathi LLM research study.

---

## 1. Environment & Locked Toolchain

Evaluations and fine-tuning are executed locally on an **NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)**.

- **OS / Host**: Linux (CachyOS x86_64)
- **Python**: 3.12.15 (managed via `uv`)
- **Key Libraries** (`requirements.lock`):
  - `torch`: `2.14.1+cu132`
  - `unsloth`: `2026.10`
  - `transformers`: `5.17.0`
  - `trl`: `1.15.0`
  - `peft`: `0.21.0`
  - `bitsandbytes`: `0.50.2`
  - `lm-eval`: `0.4.9`
  - `sacrebleu`: `2.6.0`

---

## 2. Benchmark Datasets & Pre-Phase-1 Inspection Findings

| Benchmark | Source / Config | Splits & Item Counts | Schema / Key Fields | Evaluation Modality |
|---|---|---|---|---|
| **MILU (Marathi)** | `ai4bharat/MILU` (config `Marathi`) | `validation`: 810<br>`test`: 6,924 | `question`, `option1`–`option4`, `target` (A–D), `domain`, `subject` | Next-token log-likelihood over options A–D (0 generation tokens) |
| **Flores-IN** | `google/IndicGenBench_flores_in` | `en_mr_dev`: 997<br>`mr_en_dev`: 997 | `source`, `target`, `lang`, `translation_direction` | Greedy generation (`max_new_tokens=256`), chrF++ (`word_order=2`) |
| **XQuAD-IN** | `google/IndicGenBench_xquad_in` | `dev`: 111<br>`test`: 1,190 | `id`, `question`, `context`, `answers` (list of `{text, answer_start}`) | Greedy generation (`max_new_tokens=32`), Token F1 / EM |
| **IndicIFEval** | `ai4bharat/IndicIFEval` (`indicifeval-trans`, `indicifeval-ground`) | `mr`: 490 (Trans)<br>`mr`: 480 (Ground) | `prompt`, `instruction_id_list`, `kwargs`, `resp_lang`, `tags` | `lm-eval` harness (`indicifeval_trans_mr`, `indicifeval_ground_mr`), `max_gen_toks=1280` |
| **Held-Out Aya Prompts** | `CohereForAI/aya_dataset` (Marathi slice) | `eval_heldout`: 100 | `inputs`, `targets` | Script purity, language ID, repetition, EOS rate, mean length |

### Pinned Revisions & Patches
- **IndicIFEval Repository**: Commit `1bb5f1bc4936cb20b8544e185bfd7cbed8f31464`.
- **Harness Verification Patch**: `scripts/patches/indicifeval_verification_mode.patch` applied to `IndicIFEval/lm-evaluation-harness/custom_configs/indicifeval-ground/indicifeval_mr.yaml` to bypass upstream remote split size discrepancies for non-Marathi configs (`kn`, `or`, `pa`).
- **Quantization Rule**: Checkpoints marked `prequantized: true` (e.g. `unsloth/*-bnb-4bit`) already embed 4-bit weights and must **not** be passed `load_in_4bit=True` to `lm_eval`.

---

## 3. Evaluation Protocols

### 3.1 MILU (Marathi MCQ)
- **Scoring**: A single forward pass per question calculates logits for single tokens `A`, `B`, `C`, `D` after the prompt. Argmax over the four option tokens determines prediction.
- **Prompt Format**:
  ```text
  खालील प्रश्नाचे अचूक उत्तर निवडा.
  प्रश्न: {question}
  A) {option1}
  B) {option2}
  C) {option3}
  D) {option4}
  उत्तर:
  ```
- **Reported Metrics**:
  - Overall accuracy on validation (n=810) with 95% bootstrap CI.
  - Per-domain accuracy (`STEM`, `Humanities`, `Social Sciences`, `Other`).
  - Predicted-letter distribution (`% A`, `% B`, `% C`, `% D`) to detect option bias.
  - Chance level: 25.0%.

### 3.2 Flores-IN (Translation chrF++)
- Directional evaluation: English → Marathi (`en_mr`) and Marathi → English (`mr_en`).
- Metric: chrF++ using `sacrebleu` with parameters `word_order=2`, lowercase=False.
- Decoding: Greedy, `max_new_tokens=256`.

### 3.3 IndicIFEval (Instruction Following)
- Harness-based execution using `lm-evaluation-harness`.
- Tasks: `indicifeval_trans_mr` and `indicifeval_ground_mr` reported separately (never pooled).
- Generates up to 1,280 tokens greedy.
- Reports: `prompt_level_strict_acc`, `prompt_level_loose_acc`, `inst_level_strict_acc`, `inst_level_loose_acc`.

### 3.4 Automatic Generation Metrics
Computed on all free-form generations:
- **Devanagari Script Purity**: Percentage of unicode characters in Devanagari block (`[\u0900-\u097F]`) among all non-whitespace, non-punctuation characters.
- **EOS Natural Termination**: Percentage of generations ending with tokenizer EOS before hitting `max_new_tokens`.
- **Repetition Rate**: Ratio of unique 4-grams to total 4-grams; fraction of examples containing degenerate repeated 4-grams (frequency ≥ 4).
- **Mean Output Tokens**: Average number of generated tokens.

### 3.5 Held-Out Bits-Per-Character (BPC)
- Measured on UTF-8 bytes / characters of human-written Marathi (Flores-IN Marathi text).
- Formula: $\text{BPC} = \frac{\text{Total NLL (base } e\text{)}}{\ln(2) \times \text{Total UTF-8 Characters}}$.

---

## 4. Pre-Registered Model Selection Rule (Stage A)

To eliminate post-hoc metric cherry-picking, the following selection rule is committed prior to running Stage A:

1. **Primary Criterion**: Mean rank across:
   - MILU validation accuracy (n=810)
   - Flores-IN `en_mr` dev chrF++ (n=997)
   - Flores-IN `mr_en` dev chrF++ (n=997)
2. **Equivalence & Ties**:
   - Two models whose 95% paired bootstrap confidence interval for their score difference overlaps zero ($0.0 \in \text{CI}_{95}$) are considered tied on that metric.
3. **Tie-Breaker Hierarchy**:
   - Rank on IndicIFEval Ground strict prompt accuracy.
   - Rank on held-out Flores-IN Marathi BPC (lower is better).
   - If still tied, prefer the model with lower parameter count / faster training throughput (samples/second).
4. **Hard Veto Conditions**:
   - Any model that triggers out-of-memory (OOM) at the chosen `max_seq_len` cap under effective batch 8.
   - Any model whose English benchmark performance regresses by more than 10% relative to its base checkpoint.

---

## 5. Pre-Registered Sequence-Length Decision Rule (E4, Section 2.5)

Committed prior to running E2/E3 probes. Candidate training caps: **512** (default), **768**, **1024**.

| Gate | Criterion |
|---|---|
| **Hardware** | No OOM over 100 worst-case steps (rows sized `0.9 × cap` to `cap` tokens) and peak *reserved* VRAM ≤ 3.4 GB (leaving headroom for fragmentation and the idle desktop) |
| **Time** | Projected 12k-example 1-epoch runtime ≤ 150 min for the Stage A control model (`Qwen2.5-1.5B-Instruct`, ≈ 2× the 512 run), computed as `ETA ≈ 12,000 × mean_tokens(s, c) ÷ tokens_per_sec(c)` |
| **Benefit** | Minimum kept % across S1 (`IndicAlign`) and S2 (`SandLogic`) rises by ≥ 15 percentage points over 512 |
| **Decision** | Select the smallest candidate passing all three gates; otherwise **keep 512** |

If padding is the only reason the time gate fails, length-grouped batching is permitted and must be documented as changing batch composition.

### Measured E1 + E2 Results & E4 Decision (`Qwen2.5-1.5B-Instruct`, `b2×a4`)

Full E1 census tables are in `results/length_census/summary.md` and E2 probe telemetry in `results/length_census/e2_probes.json` / `e2_summary.md`.

| Cap | Config | Peak Alloc (GB) | Peak Reserved (GB) | Steps/s | Tokens/s | Temp (°C) | ETA S0 / S1 / S2 (min) | Min Kept (S1/S2) | Δ vs 512 | E4 Verdict |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| **512** | `b2×a4` | 1.386 | 1.422 | 0.198 | 768.8 | 58.5 | 66 / 67 / 78 | 28.6% | +0.0 pp | BASELINE |
| **768** | `b2×a4` | 1.429 | 1.463 | 0.141 | 824.2 | 58.2 | 79 / 83 / 104 | 45.8% | **+17.2 pp** | **PASS (E4 Winner)** |
| **1024** | `b2×a4` | 1.469 | 1.504 | 0.109 | 842.5 | 59.2 | 83 / 93 / 125 | 58.3% | **+29.7 pp** | **PASS** |
| **1536** | `b2×a4` | 1.553 | 1.598 | 0.073 | 846.1 | 58.2 | 86 / 123 / 153 | 70.5% | +41.9 pp | *(time >150m on S2)* |
| **2048** | `b2×a4` | 1.640 | 1.695 | 0.054 | 844.9 | 58.2 | 87 / 150 / 175 | 76.4% | +47.8 pp | *(time >150m on S1/S2)* |

- **E4 Rule Outcome**: Both **768** and **1024** pass all three pre-registered gates (Hardware ≤ 3.4 GB reserved, Time ≤ 150 min, Benefit ≥ +15 pp on `min(S1, S2)`). Under the pre-registered smallest-passing-candidate rule, **768** is selected (pending optional E3 confirmation or explicit selection of 1024).
- **Memory Scaling**: Peak reserved VRAM scales linearly by only ~40–100 MB per cap increment (`1.42 GB` at 512 → `1.70 GB` at 2048) with zero OOMs, confirming Unsloth's gradient offloading + chunked loss avoids fp32 logits blowup on the 4 GB GPU.

### Pre-Registered E3 Paired Mini-Experiment Rule (S1-6k: `cap=768` vs `cap=512`)

Committed prior to training either E3 arm. Both arms are built from the same deduplicated, decontaminated S1 IndicAlign pool using the **chat-template length counter** (prompt + response + EOS), sampled with the same seed (`42`), and trained on `Qwen2.5-1.5B-Instruct` for `750` steps (1 epoch, `batch=2 × accum=4`, `seed=0`, response-only loss). Evaluation uses paired item-level bootstrap 95% CIs on MILU (`n=810`), Flores-IN (`--flores-limit 200`, both directions), and held-out Marathi BPC (`n=997`).

- **Decision Rule**:
  - **Adopt 768** if held-out Marathi BPC improves with a paired 95% CI excluding `0` (upper bound `< 0`), **or** if MILU and Flores both improve and neither regresses beyond their paired 95% CIs.
  - **Otherwise keep 512** and report the E3 result as *"no detectable effect at this sensitivity"*.
- **Sensitivity & Scope Caveats**:
  - With `n=810` MILU questions and `n=200` Flores sentences, only fairly large effects resolve and BPC is the sharpest metric; a null result does not prove cap irrelevance, only that the choice is not driven by benchmark scores.
  - Equal examples (`6,000`) is the primary control: the `768` arm sees ~32% more tokens (~2.04M vs ~1.55M) and a shifted S1 sub-corpus mix (`S1_Wiki_Conv` is ~36% of eligible S1 at 512 vs ~27% at 768; `S1_WikiHow` is <1% at both). Any observed gain reflects *"longer rows allowed, plus more tokens and sub-corpus shift"*, not sequence length alone. Each arm's exact sub-config counts are logged in `results/length_census/e3_s1_data_stats.json`.
  - **Length-counter note**: Earlier `pipeline_version: 2.1` build funnels (`64% / 38% / 23% / 86%` kept at 512) counted length with the legacy Alpaca preamble (`clean.py`), whereas the E1 census (`75.3% / 46.7% / 28.6% / 87.1%`) and `train.py` count with the Qwen2.5 chat template. `A_qwen25_1p5b` remains valid under 512 (every row is ≤512 under either counter).

### Measured E3 Results & Final Cap Decision (`ADOPT 768`)

Full report in `results/length_census/e3_comparison.md` and `e3_comparison.json`.

| Metric | `E3_s1_6k_cap512` | `E3_s1_6k_cap768` | Paired Δ (768 − 512) | Paired 95% CI | Crosses 0? |
|---|---:|---:|---:|---|---|
| **MILU val acc (%)** *(n=810)* | 27.41 | 27.41 | +0.00 pp | `[−1.48, +1.48]` | Yes *(McNemar p=1.0)* |
| **Flores en→mr chrF++** *(n=200)* | 9.85 | 9.92 | +0.07 | `[−0.63, +0.77]` | Yes |
| **Flores mr→en chrF++** *(n=200)* | 21.67 | **24.76** | **+3.09** | `[+0.76, +5.23]` | **No** |
| **Held-out Marathi BPC** *(n=997, lower better)* | 2.3245 | **2.2879** | **−0.0366** | `[−0.0408, −0.0323]` | **No** |

- **Training cost**: `cap=512` ran in `38.4 min` (`1.484 GB` reserved VRAM); `cap=768` ran in `45.0 min` (`1.480 GB` reserved VRAM).
- **MILU discordance check**: `E3_s1_6k_cap512` and `E3_s1_6k_cap768` predicted different letters on `64 / 810` questions (`19` wins for `cap=768`, `19` wins for `cap=512`, exact tie at `222 / 810 = 27.41%`, McNemar `p=1.0`, `mean_option_letters_prob_mass` `0.294` vs `0.209`).
- **Verdict & Honest Interpretation**: Under the pre-registered E3 rule (BPC paired 95% CI `[−0.0408, −0.0323]` strictly below zero, alongside a `+3.09` chrF++ paired gain on `mr→en` and no regressions), **adopt `max_seq_len = 768`**. Because the `768` arm saw 31% more tokens (`1.92M` vs `1.47M`) and a shifted sub-corpus mix, the BPC gain is expected from more training alone; E3 shows `768` does not hurt and probably helps, but it does not isolate the effect of length. The primary case for `768` is data coverage (`S2` rises from `28.6%` to `45.8%` kept and `S1` from `46.7%` to `61.1%` at essentially the same VRAM).
- **Sampling & Screening Policy**: S1 uses **uniform sampling** (not stratified) across its eligible pool so Stage B preserves what "S1" means; both example share and token share per sub-config are reported in `docs/DATA_CARD.md`. Stage B screening runs use `--flores-limit 200` (~30 min per eval), with full 997-sentence Flores evaluation run on finalists and the control.

---

## 6. Stage A Control & Stage B Baseline (`A_qwen25_1p5b_768`)

Following the adoption of `max_seq_len = 768` (Section 5) and dataset rebuilds under `pipeline_version: "2.2"`, the control adapter was retrained from scratch on the newly built `s0_alpaca_12k.jsonl` (12,000 examples, mean length 324.5 tokens, 3.89M total tokens) under response-only loss, effective batch 8 (`batch=2 × accum=4`), 1 epoch (1,500 steps), and seed 0.

### Benchmark Results (Dev Set, Full Flores n=997)

| Metric | Base (Zero-shot `chat`) | Control 512 (`A_qwen25_1p5b`) | **Control 768 (`A_qwen25_1p5b_768`)** | Δ vs Base | Δ vs Control 512 |
|---|---:|---:|---:|---:|---:|
| **MILU dev acc (%)** *(n=810)* | 27.16 `[24.20, 30.37]` | 28.64 `[25.68, 31.98]` | **29.14** `[25.93, 32.22]` | +1.98 pp | +0.50 pp |
| **Flores en→mr chrF++** *(n=997)* | 10.37 `[10.05, 10.70]` | 10.70 `[10.40, 11.03]` | **11.20** `[10.86, 11.56]` | **+0.83** | **+0.50** |
| **Flores mr→en chrF++** *(n=997)* | 25.47 `[24.58, 26.39]` | 28.26 `[27.48, 29.09]` | **28.86** `[27.91, 29.78]` | **+3.39** | **+0.60** |
| **Held-out Marathi BPC** *(n=997)* | 2.8584 | 2.2342 | **2.1724** | **−0.6860** | **−0.0618** |
| Perplexity (held-out) | 8.87 | 5.02 | **4.80** | −4.07 | −0.22 |
| Script Purity (%) | 99.89 | 100.00 | **100.00** | +0.11 pp | +0.00 pp |
| Marathi Lang-ID (%) | 96.5 | 97.49 | **96.39** | −0.11 pp | −1.10 pp |
| 4-gram Repetition (%) | 71.5 | 60.48 | **55.37** | **−16.13 pp** | **−5.11 pp** |
| EOS Completion Rate (%) | 23.0 | 37.11 | **42.43** | +19.43 pp | +5.32 pp |
| Mean Gen Tokens (cap=256) | 218.8 | 193.7 | **186.8** | −32.0 | −6.9 |

- **Training Resource**: Ran 1,500 steps in 92.9 min (`1.422 GB` allocated, `1.484 GB` peak reserved VRAM, temperature 58.0°C).
- **Interpretation**: Retraining under `max_seq_len = 768` strictly improves upon the `cap=512` control across translation quality (en→mr `+0.50` chrF++, mr→en `+0.60` chrF++), held-out bits-per-character (`2.2342 → 2.1724`), and generation fluency (repetition rate drops from `60.48%` to `55.37%` and clean EOS termination increases to `42.43%`). The 768 control serves as the valid, fair comparator (`B1`) for Stage B screening.






