# मराठी LLM Fine-Tuning — Marathi Language Model

Fine-tuning and evaluating Marathi language models under a **4 GB VRAM budget** (RTX 3050 Ti Laptop GPU) using **QLoRA + Unsloth**.

Marathi is spoken by over 83 million people but remains significantly under-represented in open-weights LLMs. This project is a systematic study comparing base model architectures and instruction data sources for Marathi under strict consumer hardware constraints, using standardized benchmark suites, judge-free quality metrics, and error analysis.

---

## What I found when I audited my own evaluation

The original report claimed **+10.44 chrF++** on 10 hand-picked prompts. An audit revealed that **four of them were in the training data**:
- Prompts 1 and 2 were **Rows 0 and 1** of the training split ($100\%$ exact match).
- Prompt 3 was a **near-match** in train (Row 10004: `"पाण्याचे महत्त्व सांगणारी काही वाक्ये लिहा"`).
- Prompt 4 was **contained in a training instruction** (Row 37982: `"एका प्रश्नाचे उत्तर द्या - भारताची राजधानी कोणती आहे?"`).

The largest gains (+33.07 and +25.06) came directly from those memorized rows. The **4 contaminated prompts averaged +19.73 chrF++**, whereas the **6 unseen prompts averaged +4.25** (evaluated under the original protocol: sampling at temperature 0.7, one sample per prompt, scored against hand-written references). Furthermore, at least three of the ten reference answers traced directly to training responses.

A re-score on 200 prompts from new hash-split manifests gave:

| Metric | Base Model (`Qwen2.5-1.5B`) | Fine-Tuned v0 Adapter | Measured Delta (Δ) | Original v0 Claim |
|---|---|---|---|---|
| **Corpus chrF++** | 17.75 | **22.37** | **+4.62** | *+10.44 (inflated by leakage)* |
| **EOS Rate (%)** | 1.00% | **81.50%** | **+80.50 pp** | *Not measured* |
| **Script Purity (%)** | **95.88%** | 93.69% | **-2.20 pp** | *Not measured* |
| **Mean Output Length** | 255.6 tokens | **123.6 tokens** | -131.9 tokens | *Not measured* |

*Notes on the 200-prompt re-score:*
- Prompts were drawn from new hash-split manifests; because v0 originally trained on an unmanifested random 95% split, some prompts may overlap v0's original training data.
- **chrF++ against machine-translated reference answers rose about 4.6 points on this set (n=200, no confidence interval). Fluency and correctness are not yet measured; that is Phase 1.**
- Under the v0 prompt template, the fine-tuned model ended answers naturally 81.5% of the time versus 1.0% for the base (+80.5 pp). This will be re-measured with the native chat template in Phase 1.

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

## Upgrade Plan & Roadmap

We are upgrading this project from a single-dataset fine-tune into an end-to-end reproducible study:

- [x] **Phase 0 — Baseline Audit & Clean Splits**: Audit train/test leakage, document findings, commit deterministic split manifests (`data/splits/`), profile hardware on the 4 GB GPU.
- [ ] **Phase 1 — Rigorous Evaluation Harness**: Native MCQ benchmark (MILU - Marathi), standard generation benchmarks (IndicGenBench Flores-IN translation & XQuAD-IN QA), verifiable instruction-following (IndicIFEval), held-out bits-per-character (BPC), and English regression checks.
- [ ] **Phase 2 — Data Pipeline**: Ingest and clean diverse Marathi sources (S0: Alpaca, S1: IndicAlign, S2: SandLogic, S3: Aya) with strict decontamination against all eval sets.
- [ ] **Phase 3 — Model Selection (Stage A)**: Benchmark candidate base models (Qwen2.5-1.5B, Qwen3-1.7B, Gemma-3-1B, Llama-3.2-1B) under equal training budgets.
- [ ] **Phase 4 — Data Selection (Stage B)**: Compare data sources and construct data scaling curves.
- [ ] **Phase 5 — Final Training & Ablations (Stage C)**: Measure noise-floor across seeds, ablate response-only loss masking, and release final adapter + GGUF.

---

## Repository Structure

```
marathi-llm/
├── configs/             # Model, data, and training YAML configs
├── data/
│   ├── splits/          # Committed split manifests (s0_test_ids.json, etc.)
│   └── eval/            # Eval prompt suites
├── docs/
│   └── legacy/          # v0 baseline numbers and leakage audit
│       ├── v0_baseline.md
│       └── leakage_audit.md
├── results/
│   ├── hw_profile.json  # VRAM, throughput, and thermal profile
│   └── v0_rescore.json  # Re-evaluation on 200 test prompts
├── scripts/             # Audit, profiling, and evaluation runners
├── src/marathi_llm/     # Package source (data, eval, train, aggregate)
├── requirements.lock    # Pinned dependency lockfile
└── plan.md              # Upgrade plan
```

---

## Setup & Quickstart

### 1. Environment Setup (using `uv`)

```bash
# In fish:
source .venv/bin/activate.fish

# In bash/zsh:
source .venv/bin/activate
```

### 2. Verify GPU & Environment

```bash
python -c "import torch, unsloth; print(torch.cuda.get_device_name(0), 'ready!')"
```

### 3. Re-run Audit & Profiling

```bash
python scripts/audit_baseline.py
python scripts/rescore_v0.py
python scripts/profile_hardware.py
```
