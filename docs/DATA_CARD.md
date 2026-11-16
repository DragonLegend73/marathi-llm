# Dataset Card: Marathi Instruction Datasets (`docs/DATA_CARD.md`)

This document records the provenance, licensing, cleaning protocols, decontamination procedures, and fixed-budget builds for the Marathi LLM comparative study.

---

## 1. Source Inventory

| ID | Dataset | Hugging Face Repository | Origin | License | Modality / Format | Intended Role |
|---|---|---|---|---|---|---|
| **S0** | **Alpaca-Marathi** | `rachittshah/alpaca-marahti` | English Stanford Alpaca machine-translated into Marathi | Non-commercial (Alpaca-derived) | Instruction–Response (Alpaca format) | Baseline control |
| **S1** | **IndicAlign-Instruct** | `ai4bharat/indic-align` | Translated Dolly, OpenAssistant, WikiHow, Anudesh, Wiki-Chat, HHRLHF | CC-BY-4.0 | Multi-turn chat (flattened to single-turn) | Diverse task coverage |
| **S2** | **Indic Chat Dataset** | `SandLogicTechnologies/Indic_Chat_Dataset` (split `marathi`) | Curated conversational Indic chat | Apache-2.0 | Turn-based chat messages | Conversational Marathi |
| **S3** | **Aya Dataset** | `CohereForAI/aya_dataset` (split `train`, `language_code=mar`) | Human-written crowdsourced multilingual instructions | Apache-2.0 | Natural prompt–completion pairs | Native quality & held-out eval |

---

## 2. Cleaning & Quality Funnel

Every raw source passes through an identical, deterministic filtering pipeline implemented in `src/marathi_llm/data/clean.py`:

1. **Unicode NFC Normalization & Artifact Stripping:**
   - Unicode normalized to `NFC`.
   - Zero-width spaces, joiners, and BOMs (`\u200B`–`\u200D`, `\uFEFF`) stripped.
   - HTML tags (`<[^>]+>`) removed.
   - Residual template tokens (`<|im_start|>`, `<|endoftext|>`, raw prompt delimiters) rejected.

2. **Script Purity & Latin Leakage:**
   - **Devanagari Script Purity $\ge 60\%$**: Evaluates Unicode letters and combining marks (`L` and `M` categories) belonging to the Devanagari block (`U+0900`–`U+097F`).
   - **Latin Leak Ratio $\le 15\%$**: Rejects rows where English/Latin characters dominate non-code responses.

3. **Degenerate Repetition:**
   - Rejects rows exhibiting repetitive n-gram loops (4-gram repetition frequency $\ge 4$).

4. **Length & Structure:**
   - Bounded by maximum sequence length ($\le 512$ tokens).
   - Responses $< 10$ characters rejected.

---

## 3. Deduplication (Exact + Near-Duplicate)

Implemented in `src/marathi_llm/data/dedup.py`:
- **Exact Hash Dedup**: SHA-256 hash over normalized `instruction + response`.
- **Near-Duplicate MinHash LSH**: 128 permutations over word shingles with a Jaccard similarity threshold of $0.80$. Rejects near-paraphrased twins within and across sources.

---

## 4. Benchmark Decontamination

Implemented in `src/marathi_llm/data/decontaminate.py`:
To ensure zero evaluation contamination, every candidate training example is checked against an index of **8,000+ benchmark evaluation strings**:
- **MILU Validation & Test**: Questions and all candidate answer options.
- **Flores-IN**: English and Marathi source/target sentences for `en_mr` and `mr_en`.
- **XQuAD-IN**: Marathi contexts, questions, and ground-truth answer strings.
- **IndicIFEval**: Marathi constraint instructions for both `trans` and `ground` suites.
- **Held-out Aya Prompts**: 100 held-out native evaluation prompts and 500 BPC responses.

**Rejection Criteria:**
- **Verbatim 8-word overlap**: Any candidate sharing $\ge 8$ consecutive words with any evaluation text is immediately rejected.
- **MinHash overlap**: Any candidate with MinHash Jaccard similarity $\ge 0.70$ against an evaluation text is rejected.

---

## 5. Fixed-Budget Builds

To enable rigorous, confound-free comparisons in Stage B, all data sources are sampled under **equal budgets**:
- **Training Set**: Exactly **12,000 examples** (`*_12k.jsonl`), shuffled with seed `42`.
- **Development Set**: Exactly **500 examples** (`*_dev500.jsonl`) held out for validation loss tracking.
- Output artifacts are stored in `data/processed/` accompanied by committed funnel counts in `*.stats.json`.
