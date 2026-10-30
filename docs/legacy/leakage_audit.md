# Baseline Audit (v0) — Leakage, References, and Validation Loss

**Date:** October 30, 2026  
**Audited commit/tag:** `v0-baseline`  
**Dataset audited:** `rachittshah/alpaca-marahti` (52,002 rows raw, 34,499 rows kept after 512-token filtering)

---

## 1. Executive Summary

An audit of the v0 baseline reveals significant evaluation leakage and explains the inflated headline results:

1. **Training-Contaminated Prompts (4 of 10 prompts leaked):**
   Under a strict definition of leakage (any prompt that appeared in the training split either verbatim, as a near-duplicate, or as a direct semantic substring), **4 of the 10 evaluation prompts were seen during training**:
   - **Prompt 1** is an exact match for **Row 0** of the training split.
   - **Prompt 2** is an exact match for **Row 1** of the training split.
   - **Prompt 3** is a near-match to **Row 10004** in the training split (`"पाण्याचे महत्त्व सांगणारी काही वाक्ये लिहा"`).
   - **Prompt 4** is contained directly within the instruction of **Row 37982** in the training split (`"एका प्रश्नाचे उत्तर द्या - भारताची राजधानी कोणती आहे?"`).

2. **Deltas Driven by Memorization:**
   - The **4 contaminated prompts** averaged an improvement of **+19.73 chrF++ points** (dominated by Prompt 2 at $\Delta = +33.07$ and Prompt 4 at $\Delta = +25.06$).
   - The **6 genuinely unseen prompts** averaged an improvement of only **+4.25 chrF++ points**.
   - The original headline average of $+10.44$ was heavily skewed by memorization of training examples.

3. **Reference Strings Audit:**
   The references in `evaluate.py` were not official benchmark references or independent human translations. They were 10 hardcoded strings manually written into the script, two of which mirrored the training dataset's translated responses verbatim.

4. **Validation Loss Definition:**
   The reported validation loss of `0.4479` was computed on a random 5% slice (1,725 examples) of the filtered dataset using standard sequence loss without prompt/completion masking (the model was evaluated on predicting both the system prompt/instruction tokens and completion tokens).

---

## 2. Detailed Leakage Breakdown (Task 0.2)

Each of the 10 prompts from `infer.py` was checked against the 32,774 examples in the v0 training split (filtered to $\le 512$ tokens, 95% split, seed 42):

| # | Prompt | Matched Row ID | Split Status | Leakage Classification | Dataset Response / Notes |
|---|---|---|---|---|---|
| 1 | निरोगी राहण्यासाठी तीन टिप्स द्या. | **Row 0** | **IN_V0_TRAIN** | **Leaked** (100% exact match) | Verbatim Row 0 instruction. Translated English Alpaca row 0. |
| 2 | तीन मुख्य रंग कोणते आहेत? | **Row 1** | **IN_V0_TRAIN** | **Leaked** (100% exact match) | Verbatim Row 1 instruction. Translated English Alpaca row 1. |
| 3 | पाण्याचे महत्त्व सांगा. | **Row 10004** | **IN_V0_TRAIN** | **Leaked** (Near-match in train) | Instruction: `"पाण्याचे महत्त्व सांगणारी काही वाक्ये लिहा"`. (Row 18278 was dropped by length). |
| 4 | भारताची राजधानी कोणती आहे? | **Row 37982** | **IN_V0_TRAIN** | **Leaked** (Contained in instruction) | Instruction: `"एका प्रश्नाचे उत्तर द्या - भारताची राजधानी कोणती आहे?"`. Ground truth: `"भारताची राजधानी नवी दिल्ली आहे."` |
| 5 | सूर्यमाला म्हणजे काय? | None | — | **Unseen** | Highest partial match: Row 12645 (Swiss roll definition, unrelated). |
| 6 | मित्राला पत्र कसे लिहावे? | None | — | **Unseen** | No match in dataset. Original $\Delta = +2.69$. |
| 7 | झाडे लावणे का महत्त्वाचे आहे? | None | — | **Unseen** | No match in dataset. Original $\Delta = +8.60$. |
| 8 | संगणक म्हणजे काय आणि त्याचा उपयोग काय? | None | — | **Unseen** | No match in dataset. Original $\Delta = +2.81$. |
| 9 | वेळेचे व्यवस्थापन कसे करावे? | Row 46607 | Dropped (>512) | **Unseen** | Near-match existed in raw data, but was dropped during length filtering (>512 tokens). Never seen in training. Original $\Delta = +3.72$. |
| 10 | चांगल्या विद्यार्थ्याचे गुण कोणते असतात? | None | — | **Unseen** | No match in dataset. Original $\Delta = +3.40$. |

### Breakdown by Category

- **All 10 prompts average Δ:** **+10.44** chrF++
- **4 contaminated prompts (1, 2, 3, 4) average Δ:** **+19.73** chrF++
- **6 genuinely unseen prompts (5, 6, 7, 8, 9, 10) average Δ:** **+4.25** chrF++

---

## 3. Reference String Audit (Task 0.3)

In `evaluate.py`, ground-truth references are hardcoded in lines 16–27:

```python
REFERENCES = [
    "1. संतुलित आहार घ्या आणि भरपूर फळे आणि भाज्या खा. 2. नियमित व्यायाम करा. 3. पुरेशी झोप घ्या.",
    "लाल, निळा आणि पिवळा असे तीन प्राथमिक रंग आहेत.",
    "पाणी जीवनासाठी अत्यंत आवश्यक आहे. ते शरीरातील पेशींना कार्यक्षम ठेवते आणि पचनक्रिया सुधारते.",
    "भारताची राजधानी नवी दिल्ली आहे.",
    ...
]
```

- **Prompt 2 reference** is word-for-word identical to the dataset's `translated_text` response for Row 1: `"लाल, निळा आणि पिवळा असे तीन प्राथमिक रंग आहेत."`
- **Prompt 4 reference** is word-for-word identical to Row 37982's `translated_text` response: `"भारताची राजधानी नवी दिल्ली आहे."`
- The remaining references are single-sentence human summaries crafted by the author. Scoring open-ended answers against a single hand-written reference sentence using character n-gram overlap (chrF++) rewards verbatim recall and heavily penalizes valid alternative phrasings.

---

## 4. Validation Loss Audit (Task 0.7)

In `train.py`:
- `dataset = load_dataset("rachittshah/alpaca-marahti", split="train")`
- `dataset = dataset.filter(filter_length, ...)` (512 token ceiling: 34,499 kept)
- `split_dataset = dataset.train_test_split(test_size=0.05, seed=42)`
- `dataset_text_field = "text"`

**Findings:**
1. The validation loss (`0.4479`) was computed over an arbitrary 5% random split of the same translated Alpaca dataset.
2. Loss was computed on all tokens in the sequence (`dataset_text_field="text"`), including the repeated Devanagari prompt wrapper (`खाली एक सूचना आहे... ### सूचना:... ### प्रतिसाद:`). Because prompt template tokens are repetitive and easy to predict, sequence loss appears artificially low.
3. No response-only loss masking (`DataCollatorForCompletionOnlyLM`) was used in v0.

---

## 5. Clean Split Manifests Created (Task 0.4)

To prevent future data leakage, `rachittshah/alpaca-marahti` (S0) was deduplicated and deterministically partitioned:
- Total raw rows: 52,002
- Near/exact duplicates by normalized instruction removed: 43 rows
- Unique rows remaining: 51,959
- Deterministic SHA-256 hash split:
  - `data/splits/s0_test_ids.json`: 1,000 examples (strictly held out)
  - `data/splits/s0_dev_ids.json`: 500 examples (for hyperparameter/dev loss)
  - `data/splits/s0_train_ids.json`: 50,459 examples (pool for training)
  - `data/splits/s0_test_200.json`: First 200 examples of the clean test split for fast baseline re-scoring (Task 0.5)

Zero ID or text overlap exists between train, dev, and test sets.

---

## 6. Honest v0 Re-evaluation on 200 Unseen Test Prompts (Task 0.5)

To establish the ground truth for v0 without train-set contamination or cherry-picked prompts, both the base model (`unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`) and the published v0 adapter (`DragonLegend/marathi-qwen2.5-lora`) were evaluated on the first 200 prompts of `data/splits/s0_test_ids.json`.

Decoding was kept strictly uniform and deterministic across both models: greedy decoding, `max_new_tokens=256`, `do_sample=False`.

### Summary Comparison Table

| Metric | Base Model (`Qwen2.5-1.5B-Instruct`) | Fine-Tuned v0 (`marathi-qwen2.5-lora`) | Measured Delta (Δ) | Original v0 Claim |
|---|---|---|---|---|
| **Corpus chrF++** | **17.75** | **22.37** | **+4.62** | *+10.44* |
| **EOS Rate (%)** | 1.00% | 81.50% | **+80.50%** | *Not measured* |
| **Script Purity (%)** | 95.88% | 93.69% | **-2.20%** | *Not measured* |
| **Mean Output Length** | 255.6 tokens | 123.6 tokens | **-131.9 tokens** | *Not measured* |

### Analysis of the Re-evaluated Baseline

1. **The chrF++ gain is real, but cut in half (+4.62 vs +10.44):**
   When tested on 200 genuine unseen Marathi prompts, fine-tuning improves chrF++ by **+4.62 points**. The original claim of **+10.44 points** was an artifact of testing on contaminated rows.
2. **The primary improvement is conversational termination (EOS Rate 1% → 81.5%):**
   The base model fails to recognize when to stop in Marathi; in 99% of cases it rambles on until hitting the hard ceiling of 256 tokens (averaging 255.6 tokens). The fine-tuned model terminates naturally 81.5% of the time, producing concise Marathi answers (averaging 123.6 tokens).
3. **Slight drop in script purity (-2.20%):**
   The fine-tuned model exhibits slight English code-mixing inherited from machine-translated artifacts in Alpaca, causing Devanagari letter purity to drop slightly from 95.88% to 93.69%.

---

## 7. Hardware Profile on RTX 3050 Ti Laptop GPU (Task 0.6)

100 training steps of the baseline QLoRA configuration were profiled on the RTX 3050 Ti Laptop GPU (batch 1, accumulation 8, `max_seq_len=512`, optimizer `paged_adamw_8bit`).

Detailed logs are saved in `results/hw_profile.json`:

- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
- **Peak VRAM Allocated**: **1.38 GB**
- **Peak VRAM Reserved**: **1.45 GB** (only 36% of 4 GB capacity utilized)
- **Throughput**: **0.249 steps/sec** (≈ **1.99 samples/sec**)
- **Thermals & Power**:
  - Idle Temp: 51 °C | Load Temp: 59 °C (stable laptop thermals)
  - Power draw: 33.2 W (under 60 W TGP cap)
- **Extrapolated Time for 12k Examples**:
  - 1,500 optimizer steps / 0.249 steps/s = ~6,031 seconds ≈ **100.5 minutes** (~1.67 hours).
