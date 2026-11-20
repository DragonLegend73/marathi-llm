# Dataset Card: Marathi Instruction Datasets (`docs/DATA_CARD.md`)

This document records the provenance, licensing, cleaning protocols, decontamination procedures, and fixed-budget builds (`pipeline_version: 2.2`) for the Marathi LLM comparative study.

---

## 1. Source Inventory

| ID | Dataset | Hugging Face Repository | Origin | License | Modality / Format | Intended Role |
|---|---|---|---|---|---|---|
| **S0** | **Alpaca-Marathi** | `rachittshah/alpaca-marahti` | English Stanford Alpaca machine-translated into Marathi | Non-commercial (Alpaca-derived) | Instruction–Response (Alpaca format) | Baseline control |
| **S1** | **IndicAlign-Instruct** | `ai4bharat/indic-align` | Translated Dolly, OpenAssistant, WikiHow, Anudesh, Wiki-Conv, Wiki-Chat, HHRLHF (`per_config_limit: 15,000`) | CC-BY-4.0 | Multi-turn chat (flattened to single-turn) | Diverse task coverage |
| **S2** | **Indic Chat Dataset** | `SandLogicTechnologies/Indic_Chat_Dataset` (`split=marathi`, first `100,000` streamed rows) | Curated conversational Indic chat | Apache-2.0 | Turn-based chat messages | Conversational Marathi |
| **S3** | **Aya Dataset** | `CohereForAI/aya_dataset` (`split=train`, `language_code=mar`) | Human-written crowdsourced multilingual instructions | Apache-2.0 | Natural prompt–completion pairs | Native quality & held-out eval |

---

## 2. Cleaning & Quality Funnel

Every raw source passes through the same deterministic filtering pipeline implemented in `src/marathi_llm/data/clean.py`:

1. **Unicode NFC Normalization & Artifact Stripping:**
   - Unicode normalized to `NFC`.
   - Zero-width spaces, joiners, and BOMs (`\u200B`–`\u200D`, `\uFEFF`) stripped.
   - HTML tags (`<[^>]+>`) removed.
   - Residual template tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, raw prompt delimiters) rejected.

2. **Script Purity, Language ID & Latin Leakage:**
   - **Devanagari Script Purity ≥ 60%**: Evaluates Unicode letters and combining marks (`L` and `M` categories) belonging to the Devanagari block (`U+0900`–`U+097F`).
   - **Latin Leak Ratio ≤ 15%**: Rejects rows where English/Latin characters exceed 15% of letters.
   - **Marathi Language ID** (`lingua-language-detector`, enabled on S1/S2/S3): Filters out Hindi and non-Marathi Devanagari rows.

3. **Degenerate Repetition:**
   - Rejects rows exhibiting repetitive 4-gram loops (frequency ≥ 4).

4. **Length & Structure:**
   - Bounded by `max_seq_len = 768` tokens (`pipeline_version: 2.2`, adopted per Section 2.5 E1–E4 study; measured using the `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` **chat template** matching `marathi_llm.train`). Earlier `pipeline_version: 2.1` (`512`-token, Alpaca-preamble) builds are archived in `data/processed/archive_cap512/`.
   - Responses `< 10` characters rejected.

---

## 3. Deduplication (Exact + Near-Duplicate)

Implemented in `src/marathi_llm/data/dedup.py` using shared Devanagari word tokenization (`src/marathi_llm/data/text.py`, excluding danda punctuation `।`/`॥`):
- **Exact Hash Dedup**: SHA-256 hash over normalized `instruction + response`.
- **Near-Duplicate MinHash LSH**: 128 permutations over 5-word shingles with Jaccard similarity threshold `0.80`.

---

## 4. Benchmark Decontamination

Implemented in `src/marathi_llm/data/decontaminate.py`:
Every candidate training example is checked against an index of **52,149 evaluation strings**:
- **MILU Validation & Test**: Questions and all four answer options.
- **Flores-IN (dev + test)**: English and Marathi source/target sentences for `en_mr` and `mr_en`.
- **XQuAD-IN (dev + test)**: Marathi contexts, questions, and ground-truth answers.
- **IndicIFEval (`ground` + `trans`)**: Marathi constraint prompts.
- **Held-out Aya Sets**: `100` clean held-out native evaluation prompts + references (`data/eval/native_prompts.jsonl`) and `500` held-out BPC responses (`data/eval/bpc_responses.jsonl`), carved out after cleaning, deduplication, and decontamination of S3.

**Rejection Criteria:**
- **Verbatim 8-word overlap**: Any candidate sharing ≥ 8 consecutive Devanagari/word tokens with any indexed evaluation string is rejected.
- **MinHash overlap**: Any candidate with 5-shingle MinHash Jaccard similarity ≥ 0.70 against any indexed evaluation string is rejected.

---

## 5. Measured Build Funnels (`pipeline_version: 2.2`, `max_seq_len = 768`, Chat-Template Counter)

All active funnel counts below are recorded in committed `data/processed/*.stats.json` files (uniform sampling, `seed = 42`, frozen `data/eval/` held-outs):

| Source | Raw Ingested | Held-Out Removed | Failed Length (>768 tok) | Failed Other Clean | Passed Clean | Failed Dedup | Failed Decontam | Final Pool | Built Train / Dev | Mean Tok (Train) | Total Train Tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S0 (`s0_alpaca`)** | 51,973 | 0 | 3,084 | 4,242 | 44,647 | 2 | 1 | 44,644 | 12,000 / 500 | 324.5 | 3.89M |
| **S1 (`s1_indicalign`)** | 90,000 | 0 | 32,402 | 6,709 | 50,889 | 8,511 | 15 | 42,363 | 12,000 / 500 | 321.1 | 3.85M |
| **S2 (`s2_sandlogic`)** | 100,000 | 0 | 49,760 | 8,131 | 42,109 | 2,371 | 2 | 39,736 | 12,000 / 500 | 415.4 | 4.99M |
| **S3 (`s3_aya`)** | 3,545 | 600 *(frozen eval)* | 282 | 1,033 *(413 Hindi, 482 short)* | 1,630 | 203 | 91 *(benchmark overlap)* | 1,336 | 1,203 / 133 | 191.8 | 0.23M |

### S1 IndicAlign (`s1_indicalign_12k.jsonl`) Sub-Config Composition (Uniform Sampling, `max_seq_len = 768`)

| Sub-Config | Train Count | Example Share (%) | Mean Tokens | Total Tokens | Token Share (%) |
|---|---:|---:|---:|---:|---:|
| `S1_HHRLHF_T` | 2,648 | 22.07% | 505.6 | 1,338,867 | **34.75%** |
| `S1_Dolly_T` | 3,184 | 26.53% | 315.4 | 1,004,336 | **26.06%** |
| `S1_OpenAssistant_T` | 1,589 | 13.24% | 386.6 | 614,292 | **15.94%** |
| `S1_Wiki_Conv` | 3,744 | **31.20%** | 128.9 | 482,588 | **12.52%** |
| `S1_Wiki_Chat` | 764 | 6.37% | 481.7 | 368,053 | **9.55%** |
| `S1_WikiHow` | 71 | 0.59% | 634.9 | 45,076 | **1.17%** |

*Note:* Although `S1_Wiki_Conv` accounts for `31.2%` of examples under uniform sampling, its rows are short (`mean = 128.9` tokens), so it represents only `12.5%` of S1 training tokens.

### Archived `pipeline_version: 2.1` Funnels (`max_seq_len = 512`, Legacy Alpaca-Preamble Counter)

Archived in `data/processed/archive_cap512/` for sensitivity reference (counted `512` tokens using the legacy Alpaca preamble, ~85 tokens overhead vs. ~25 tokens for chat template):

| Source | Raw Ingested | Failed Length (>512 tok) | Failed Other Clean | Passed Clean | Failed Dedup | Failed Decontam | Final Pool | Built Train / Dev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **S0 (`s0_alpaca`)** | 51,973 | 17,220 | 4,242 | 30,511 | 2 | 0 | 30,509 | 12,000 / 500 |
| **S1 (`s1_indicalign`)** | 90,000 | 51,660 | 6,709 | 31,631 | 3,457 | 6 | 28,168 | 12,000 / 500 |
| **S2 (`s2_sandlogic`)** | 100,000 | 70,362 | 8,131 | 21,507 | 161 | 1 | 21,345 | 12,000 / 500 |
| **S3 (`s3_aya`)** | 3,545 | 351 | 1,033 | 2,161 | 262 | 0 | 1,899 *(−600 held-out)* | 1,170 / 129 |

---

## 6. Sequence-Length Census (E1 — Qwen2.5 Chat Template + EOS)

Measured across all rows passing quality filters (excluding length, dedup, and decontamination; full JSON in `results/length_census/`):

| Source | Pass Clean | p50 | p90 | p99 | ≤384 | ≤512 | ≤768 | ≤1024 | ≤1536 | ≤2048 | Mean@512 | Mean@768 | Mean@1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S0 (`s0_alpaca`)** | 47,731 | 307 | 685 | 1234 | 59.5% | 75.3% | 93.5% | 97.8% | 99.6% | 99.9% | 255.2 | 325.5 | 349.2 |
| **S1 (`s1_indicalign`)** | 83,291 | 548 | 2505 | 9272 | 33.6% | 46.7% | 61.1% | 67.2% | 78.5% | 86.5% | 257.6 | 340.8 | 390.4 |
| **S2 (`s2_sandlogic`)** | 91,869 | 842 | 2305 | 2529 | 20.2% | 28.6% | 45.8% | 58.3% | 70.5% | 76.4% | 298.0 | 426.7 | 524.9 |
| **S3 (`s3_aya`)** | 2,512 | 156 | 1091 | 3227 | 84.2% | 87.1% | 88.8% | 89.7% | 91.6% | 94.8% | 179.4 | 187.7 | 194.5 |

### S1 IndicAlign Kept % by Sub-Config

| Sub-Config | Pass Clean | p50 | p90 | ≤512 | ≤768 | ≤1024 | ≤1536 | ≤2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `S1_Dolly_T` | 13,202 | 317 | 918 | 72.7% | 86.0% | 91.9% | 96.3% | 98.1% |
| `S1_HHRLHF_T` | 14,997 | 500 | 656 | 54.8% | 97.2% | 99.6% | 99.9% | 100.0% |
| `S1_OpenAssistant_T` | 13,995 | 643 | 1750 | 40.3% | 57.1% | 68.4% | 85.2% | 94.0% |
| `S1_WikiHow` | 13,532 | 2460 | 8100 | 0.2% | 1.5% | 8.2% | 34.1% | 46.5% |
| `S1_Wiki_Chat` | 13,678 | 1470 | 2393 | 11.3% | 21.1% | 31.4% | 53.3% | 79.1% |
| `S1_Wiki_Conv` | 13,887 | 120 | 177 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

