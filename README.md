# मराठी LLM Fine-Tuning — Marathi Language Model

> Fine-tuning and evaluating Marathi language models under a **4 GB VRAM budget** (RTX 3050 Ti Laptop GPU) using **QLoRA + Unsloth**.

---

## ⚠️ Baseline Audit & Revised Results

> [!IMPORTANT]
> **Evaluation Audit Notice (October 2026):**
> An audit of the initial (v0) release revealed that **4 of the 10 original test prompts were present in the training split** (including Row 0 and Row 1 of the dataset). The previously reported headline improvement of **+10.44 chrF++ points** was inflated by direct memorization of these training examples.
>
> Full audit findings and methodology are documented in [`docs/legacy/leakage_audit.md`](docs/legacy/leakage_audit.md). The original v0 documentation is preserved in [`docs/legacy/v0_baseline.md`](docs/legacy/v0_baseline.md).

### Honest Re-Evaluation on 200 Clean Held-Out Prompts

Both the base model (`Qwen2.5-1.5B-Instruct-bnb-4bit`) and the published v0 adapter (`DragonLegend/marathi-qwen2.5-lora`) were re-evaluated on a **strictly held-out test split of 200 unseen Marathi prompts** (`data/splits/s0_test_200.json`) with deterministic greedy decoding:

| Metric | Base Model (`Qwen2.5-1.5B`) | Fine-Tuned v0 Adapter | Measured Delta (Δ) | Original v0 Claim |
|---|---|---|---|---|
| **Corpus chrF++** | 17.75 | **22.37** | **+4.62** | *+10.44 (inflated by leakage)* |
| **EOS Rate (%)** | 1.00% | **81.50%** | **+80.50%** | *Not measured* |
| **Script Purity (%)** | **95.88%** | 93.69% | -2.20% | *Not measured* |
| **Mean Output Length** | 255.6 tokens | **123.6 tokens** | -131.9 tokens | *Not measured* |

### What the Audit Revealed
1. **The chrF++ gain is real but halved (+4.62 vs +10.44):** The model does produce better Marathi prose than base Qwen, but the massive +33.07 and +25.06 deltas in v0 were due to testing on Row 0, Row 1, and Row 37982 of the training data.
2. **The real win is conversational termination (+80.5% EOS Rate):** Base Qwen almost never stopped in Marathi (rambling to the 256-token limit 99% of the time). Fine-tuning taught the model how and when to end a Marathi response naturally.
3. **Hardware validation (4 GB VRAM):** 100-step training profiling verified peak allocated VRAM of **1.38 GB** (1.45 GB reserved) at **0.249 steps/s (~1.99 samples/s)** on the RTX 3050 Ti laptop GPU with stable 59 °C thermals (`results/hw_profile.json`).

---

## Hardware & Environment

| Component | Details |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM) |
| Host OS | CachyOS / Linux |
| Python Environment | Python 3.12 managed via `uv` (`.venv`) |
| Core Stack | PyTorch 2.14.1+cu132, Unsloth 2026.10, Transformers 5.17, TRL 1.15, PEFT 0.21 |
| Public Adapter | [DragonLegend/marathi-qwen2.5-lora](https://huggingface.co/DragonLegend/marathi-qwen2.5-lora) |

---

## Upgrade Plan & Roadmap

We are upgrading this project from a single-dataset fine-tune into an end-to-end reproducible research study:

- [x] **Phase 0 — Baseline Audit & Clean Splits**: Audit train/test contamination, document leakage, generate deterministic hash split manifests (`data/splits/`), profile hardware.
- [ ] **Phase 1 — Rigorous Evaluation Harness**: Native MCQ benchmark (MILU - Marathi), standard generation benchmarks (IndicGenBench Flores-IN translation & XQuAD-IN QA), verifiable instruction-following (IndicIFEval), held-out bits-per-character (BPC), and English regression checks.
- [ ] **Phase 2 — Data Pipeline**: Ingest and clean diverse Marathi sources (S0: Alpaca, S1: IndicAlign, S2: SandLogic, S3: Aya) with strict decontamination against all eval sets.
- [ ] **Phase 3 — Model Selection (Stage A)**: Benchmark candidate base models (Qwen2.5-1.5B, Qwen3-1.7B, Gemma-3-1B, Llama-3.2-1B) under equal training budgets.
- [ ] **Phase 4 — Data Selection (Stage B)**: Compare data sources and construct data scaling curves.
- [ ] **Phase 5 — Final Training & Ablations (Stage C)**: Noise-floor measurement across random seeds, response-only loss masking ablations, and final model release.

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
│   └── v0_rescore.json  # Re-evaluation on 200 unseen prompts
├── scripts/             # Audit, profiling, and evaluation runners
├── src/marathi_llm/     # Package source (data, eval, train, aggregate)
├── requirements.lock    # Pinned dependency lockfile
└── plan.md              # Detailed upgrade plan
```

---

## Setup & Quickstart

### 1. Environment Setup (using `uv`)

```bash
# Activate virtual environment
# In fish:
source .venv/bin/activate.fish

# In bash/zsh:
source .venv/bin/activate
```

### 2. Verify GPU & Stack

```bash
python -c "import torch, unsloth; print(torch.cuda.get_device_name(0), 'ready!')"
```

### 3. Re-run Audit & Profiling

```bash
python scripts/audit_baseline.py
python scripts/rescore_v0.py
python scripts/profile_hardware.py
```
