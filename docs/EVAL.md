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
   - Any model that triggers out-of-memory (OOM) at `max_seq_len=512` under batch size 2 with gradient accumulation 4.
   - Any model whose English benchmark performance regresses by more than 10% relative to its base checkpoint.
