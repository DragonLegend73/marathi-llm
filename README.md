# मराठी LLM Fine-Tuning — Marathi Language Model

Fine-tuning and evaluating Marathi language models under a **4 GB VRAM budget** (RTX 3050 Ti Laptop GPU) using **QLoRA + Unsloth**.

Marathi is spoken by over 83 million people but remains under-represented in open-weights LLMs. This repository implements a reproducible evaluation harness and audited Marathi instruction data pipeline to benchmark a pilot QLoRA adapter (`DragonLegend/marathi-qwen2.5-lora` on `Qwen2.5-1.5B-Instruct`) against the unadapted base model under matched prompt formats.

---

## Benchmark Results (Base vs. Pilot v0 Adapter)

All four runs were executed from commit `1cd6b0f` with greedy decoding (`max_new_tokens=256`, 4-bit NF4 base weights). Neither format pairing is fully fair on its own (`alpaca_v0` is native to `v0` and off-template for `base`; `chat` is the reverse), so the most conservative headline compares each model in its best prompt format:

| Metric | Base (best format) | v0 (best format) | Δ |
|---|---|---|---|
| **MILU-Marathi val acc (%)** | 27.16 *(chat)* | **35.06** *(alpaca)* | **+7.90 pp** |
| **Flores en→mr chrF++** | 10.59 *(alpaca)* | **14.44** *(alpaca)* | **+3.85** |
| **Flores mr→en chrF++** | 25.47 *(chat)* | **29.69** *(chat)* | **+4.22** |
| **Held-out Marathi BPC** *(lower is better)* | 2.858 | **2.194** | **−0.664** |

### Full 2×2 Matrix (All Four Runs)

| Metric | Base (chat) | Base (alpaca) | v0 (alpaca) | v0 (chat) |
|---|---|---|---|---|
| **MILU-Marathi val acc (%)** | 27.16 [24.20, 30.37] | 26.54 [23.58, 29.63] | **35.06** [31.85, 38.40] | 33.33 [30.12, 36.54] |
| **MILU predicted-letter spread (A/B/C/D %)** | 98.5 / 0.1 / 0.7 / 0.6 | 99.5 / 0.1 / 0.1 / 0.2 | 48.9 / 38.4 / 7.5 / 5.2 | 58.0 / 33.2 / 5.3 / 3.5 |
| **Flores en→mr chrF++** | 9.45 [9.16, 9.73] | 10.59 [10.42, 10.76] | **14.44** [14.05, 14.84] | 14.39 [13.98, 14.78] |
| **Flores mr→en chrF++** | 25.47 [24.71, 26.15] | 12.48 [12.11, 12.85] | 29.33 [28.34, 30.29] | **29.69** [28.71, 30.70] |
| **Held-out Marathi BPC** *(lower is better)* | 2.858 | 2.858 | **2.194** | **2.194** |

*Footnotes:*
- BPC does not depend on prompt format, so the duplicate columns are expected.
- Runs used batch size 4 (`base_chat` was resumed mid-run after changing batch size, so its `metrics.json` still records batch size 2); repeated runs differ by about ±1 MILU question and ±0.1–0.4 chrF++, which is small against the gaps.

*Summary of findings:*
- **MILU-Marathi**: Letter bias reduced (`99.5%` A → `48.9%` A / `38.4%` B under `alpaca_v0`; `98.5%` A → `58.0%` A / `33.2%` B under `chat`), lifting zero-shot validation accuracy by `+7.90` pp on the best-of-format comparison (`27.16%` → `35.06%`). Base puts only `4%` (`alpaca`) to `17%` (`chat`) of its probability mass on the letters A–D (vs. `29%`–`47%` for v0), so its near-chance score reflects an answer-format failure as much as missing knowledge.
- **Flores-IN & BPC**: Gains on Flores and BPC, with 95% bootstrap CIs (`+3.85` en→mr chrF++ and `+4.22` mr→en chrF++ on best-of-format). `Base (alpaca)` drops to `12.48` on mr→en because it breaks off its native template (`0.0%` EOS rate), so the `chat` figure (`+4.22`) is reported rather than the alpaca gap. Flores and matched-format MILU intervals do not overlap (for MILU under `chat`, the intervals nearly touch: `[24.20, 30.37]` vs. `[30.12, 36.54]`). Held-out Marathi BPC improves from `2.858` to `2.194` (`−0.664`; token-level perplexity `7.87` → `4.87`, BOS `<|endoftext|>`).
- **Generation stats on translation outputs**: Devanagari script purity rises from `91.9%` (`Base chat`) / `93.8%` (`Base alpaca`) to `100.0%` (`v0`), 4-gram repetition drops from `55.7%` / `64.8%` to `29.9%` (`v0 alpaca`) / `27.5%` (`v0 chat`), and EOS termination rate rises from `36.6%` (`Base chat`) / `0.0%` (`Base alpaca`) to `67.7%` (`v0 alpaca`) / `69.2%` (`v0 chat`), though roughly one-third of v0 translations still hit the 256-token cap.

### Evaluation Notes & Caveats

> v0 is a pilot QLoRA adapter (`Qwen2.5-1.5B-Instruct`, trained on Alpaca-Marathi, ~52k machine-translated examples, template matched to training data). Evaluation is zero-shot with greedy decoding: MILU (810 validation questions, next-token scoring over A–D), Flores-IN translation (chrF++, 997 sentences per direction), and held-out Marathi BPC. Intervals are bootstrap 95% CIs. MILU scores are close to chance and strongly affected by option-letter bias, so the predicted-letter distribution is reported alongside accuracy. Generation-quality statistics (script purity, repetition, EOS rate) were measured on translation outputs only. Contamination checks found no overlap detected between the training data and Flores or MILU.

Full per-run outputs are saved in [`results/runs/`](results/runs/):
- [`results/runs/base_chat/metrics.json`](results/runs/base_chat/metrics.json)
- [`results/runs/base_alpaca/metrics.json`](results/runs/base_alpaca/metrics.json)
- [`results/runs/v0_alpaca/metrics.json`](results/runs/v0_alpaca/metrics.json)
- [`results/runs/v0_chat/metrics.json`](results/runs/v0_chat/metrics.json)

---

## Data Pipeline Audit & Fix

> While auditing the data pipeline I found that word tokenization used `\w+`, which splits Devanagari words at vowel signs, so the "8-word" overlap check actually matched ~3-word spans and rejected 11,378 rows (~22%) as false benchmark contamination. I fixed tokenization (shared [`text.py`](src/marathi_llm/data/text.py) helper excluding danda punctuation `।`/`॥`), widened the decontamination index to MILU val/test, Flores, XQuAD-IN, IndicIFEval, and native prompts, and enforced sequence length bounds. Re-auditing the v0 training data with the fixed matcher found 1 overlapping row out of 52,002 (an IndicIFEval riddle prompt at n=8, no overlap detected with Flores or MILU). Regression tests cover the tokenizer bug and prompt-template parity. The initial 512-token cleaned build (30,509 pool after filtering 51,973 ingested rows) is archived in [`data/processed/archive_cap512/`](data/processed/archive_cap512/); the active Phase 3/4 pipeline uses `pipeline_version: 2.2` with `max_seq_len = 768` and the chat-template counter (44,644 clean pool for S0; built into [`data/processed/s0_alpaca_12k.jsonl`](data/processed/s0_alpaca_12k.jsonl)).

Audit artifacts:
- [`results/audit_v0_contamination.json`](results/audit_v0_contamination.json)
- [`data/processed/s0_alpaca.stats.json`](data/processed/s0_alpaca.stats.json)

---

## Initial Leakage Audit (Why the Harness Was Rebuilt)

The original v0 report claimed **+10.44 chrF++** on 10 hand-picked prompts. An audit revealed that **four of them were in the training data**:
- Prompts 1 and 2 were **Rows 0 and 1** of the training split (100% exact match).
- Prompt 3 was a **near-match** in train (Row 10004: `"पाण्याचे महत्त्व सांगणारी काही वाक्ये लिहा"`).
- Prompt 4 was **contained in a training instruction** (Row 37982: `"एका प्रश्नाचे उत्तर द्या - भारताची राजधानी कोणती आहे?"`).

The largest gains (`+33.07` and `+25.06`) came directly from those memorized rows. The **4 contaminated prompts averaged +19.73 chrF++**, whereas the **6 unseen prompts averaged +4.25** (under the original protocol: sampling at temperature 0.7, one sample per prompt, scored against hand-written references). Furthermore, at least three of the ten reference answers traced directly to training responses.

For the full methodology and individual prompt matches, see [`docs/legacy/leakage_audit.md`](docs/legacy/leakage_audit.md). The original v0 documentation is preserved in [`docs/legacy/v0_baseline.md`](docs/legacy/v0_baseline.md).

---

## Hardware Profile (RTX 3050 Ti Laptop GPU)

A 100-step training profile of the baseline QLoRA configuration (batch size 1, gradient accumulation 8, `max_seq_len=512`) measured:

- **Peak VRAM Allocated**: **1.38 GB** (1.45 GB reserved, ~36% of 4 GB capacity)
- **Throughput**: **0.249 steps/s** (~1.99 samples/s)
- **Thermals & Power**: 59 °C at the end of a 100-step profile; 33.2 W measured power draw
- **Full 12k Run Estimate**: ~100.5 minutes (~1.67 hours)

Detailed telemetry is logged in [`results/hw_profile.json`](results/hw_profile.json).

---

## Sequence-Length Study & Rebuild (`max_seq_len = 768`)

Before screening models and data sources, we conducted a sequence-length study (E1–E4) to avoid truncating long conversational rows:
- **E1 Census**: Under the chat template, `cap=512` rejected 53% of S1 (IndicAlign) and 71% of S2 (SandLogic).
- **E2 Worst-Case Probes**: Tested 100 worst-case steps (`batch=2 × accum=4`) on the RTX 3050 Ti Laptop GPU (4 GB VRAM) across sequence caps `{512, 768, 1024, 1536, 2048}`. Peak reserved VRAM grew by only ~40–100 MB per step (`1.42 GB` at 512 → `1.70 GB` at 2048) with zero OOMs.
- **E3 Paired Mini-Experiment (S1-6k)**: Paired item-level evaluation showed held-out Marathi BPC improved significantly (`2.3245 → 2.2879`, paired 95% CI `[−0.0408, −0.0323]`), Flores mr→en gained `+3.09` chrF++ (`[+0.76, +5.23]`), and MILU had zero regressions.
- **Pre-Registered Decision**: **Adopted `max_seq_len = 768`**. S0–S3 datasets were rebuilt under `pipeline_version: "2.2"` using chat-template length counting and uniform sampling for S1.

### Stage A Control & Baseline Retraining (`A_qwen25_1p5b_768`)

The baseline control was retrained from scratch at `max_seq_len = 768` (12k S0, 1,500 steps, response-only loss, `batch=2 × accum=4`, seed 0) and fully evaluated on dev (`n=810` MILU, full `n=997` Flores, `n=997` BPC):

| Metric | Base (Zero-shot `chat`) | Control 512 (`A_qwen25_1p5b`) | **Control 768 (`A_qwen25_1p5b_768`)** | Δ vs Base | Δ vs Control 512 |
|---|---:|---:|---:|---:|---:|
| **MILU dev acc (%)** *(n=810)* | 27.16 `[24.20, 30.37]` | 28.64 `[25.68, 31.98]` | **29.14** `[25.93, 32.22]` | +1.98 pp | +0.49 pp |
| **Flores en→mr chrF++** *(n=997)* | 9.45 `[9.16, 9.73]` | 10.70 `[10.40, 11.03]` | **11.20** `[10.86, 11.56]` | **+1.75** | **+0.50** |
| **Flores mr→en chrF++** *(n=997)* | 25.47 `[24.71, 26.15]` | 28.26 `[27.48, 29.09]` | **28.86** `[27.91, 29.78]` | **+3.38** | **+0.60** |
| **Held-out Marathi BPC** *(n=997)* | 2.8584 | 2.2342 | **2.1724** | **-0.6859** | **-0.0618** |
| 4-gram Repetition (%) | 55.67 | 60.48 | **55.37** | **-0.30 pp** | **-5.12 pp** |
| EOS Completion Rate (%) | 36.61 | 37.11 | **42.43** | **+5.82 pp** | **+5.32 pp** |

---

## Repository Structure

```
marathi-llm/
├── configs/             # Model, data, and training YAML configs
├── data/
│   ├── processed/       # Cleaned 768-token datasets & stats (pipeline_version 2.2, S0–S3)
│   │   └── archive_cap512/ # Archived 512-token datasets
│   ├── splits/          # Committed split manifests (s0_test_ids.json, etc.)
│   └── eval/            # Evaluation benchmarks (Flores, MILU, native prompts, BPC responses)
├── docs/                # EVAL.md, DATA_CARD.md, replication docs
│   └── legacy/          # Original v0 notes and leakage audit
├── results/
│   ├── figures/         # Summary metric plots
│   ├── runs/            # Evaluation run outputs (active runs + archive/)
│   ├── tables/          # Markdown tables generated by make_tables.py
│   ├── audit_v0_contamination.json
│   └── hw_profile.json  # VRAM, throughput, and thermal profile
├── scripts/             # Audit, profiling, and helper scripts
├── src/marathi_llm/     # Package source (data/, eval/, generate.py, aggregate.py)
├── tests/               # Unit and regression tests (tokenizer, template parity, data pipeline)
└── requirements.lock    # Pinned dependency lockfile
```

---

## Setup & Reproduction

### 1. Environment Setup

```bash
# In bash/zsh:
source .venv/bin/activate

# In fish:
source .venv/bin/activate.fish
```

### 2. Run Test Suite

```bash
.venv/bin/pytest
```

### 3. Reproduce the 4-Way Evaluation

```bash
.venv/bin/python -m marathi_llm.eval.runner --run-id base_chat   --prompt-format chat      --skip-ifeval
.venv/bin/python -m marathi_llm.eval.runner --run-id base_alpaca --prompt-format alpaca_v0 --skip-ifeval
.venv/bin/python -m marathi_llm.eval.runner --run-id v0_alpaca   --prompt-format alpaca_v0 --skip-ifeval --adapter DragonLegend/marathi-qwen2.5-lora
.venv/bin/python -m marathi_llm.eval.runner --run-id v0_chat     --prompt-format chat      --skip-ifeval --adapter DragonLegend/marathi-qwen2.5-lora
```

### 4. Rebuild S0 Dataset & Run Contamination Audit

```bash
.venv/bin/python -m marathi_llm.data.build --config configs/data/s0_alpaca.yaml
.venv/bin/python scripts/audit_v0_contamination.py
```
