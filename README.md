# मराठी LLM Fine-Tuning — Marathi Language Model

> Fine-tuning **Qwen2.5-1.5B-Instruct** on a Marathi Alpaca instruction dataset using **QLoRA + Unsloth** on consumer-grade hardware.

---

## Overview

Marathi is spoken by over 83 million people but remains significantly under-represented in modern LLMs. This project demonstrates that parameter-efficient fine-tuning (QLoRA) can meaningfully improve Marathi language generation in a small LLM — using only a 4 GB VRAM laptop GPU.

**Key result:** Corpus chrF++ improved from **15.39 → 25.83** (+10.44 points) over the base model across 10 Marathi prompts. The base model produces incoherent, language-mixed output; the fine-tuned model consistently generates structured Marathi prose.

---

## Hardware & Environment

| Component | Details |
|---|---|
| GPU | NVIDIA RTX 3050 Ti Laptop (4 GB VRAM) |
| OS | Bazzite Linux |
| Training env | Distrobox (Ubuntu) with CUDA passthrough |
| Inference | Ollama (host) |

---

## Model Details

| | |
|---|---|
| **Base model** | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` |
| **Dataset** | [`rachittshah/alpaca-marahti`](https://huggingface.co/datasets/rachittshah/alpaca-marahti) |
| **Dataset size** | 52,002 samples (34,499 after filtering) |
| **Method** | QLoRA — LoRA r=16, alpha=32 |
| **Epochs** | 1 (4,097 optimizer steps) |
| **Final val loss** | 0.4479 |
| **Adapter** | [DragonLegend/marathi-qwen2.5-lora](https://huggingface.co/DragonLegend/marathi-qwen2.5-lora) |

---

## Results

### Training Loss Curve

![Loss Curve](evals/loss_curve.png)

Training loss dropped from **1.36 → 0.46** over 4,097 steps. Validation loss tracked closely with no overfitting.

### chrF++ Evaluation

| Prompt | Base | Fine-Tuned | Δ |
|---|---|---|---|
| निरोगी राहण्यासाठी तीन टिप्स द्या. | 17.92 | 28.25 | +10.33 |
| तीन मुख्य रंग कोणते आहेत? | 9.57 | 42.65 | +33.07 |
| पाण्याचे महत्त्व सांगा. | 11.89 | 22.34 | +10.45 |
| भारताची राजधानी कोणती आहे? | 21.84 | 46.90 | +25.06 |
| सूर्यमाला म्हणजे काय? | 12.18 | 16.46 | +4.28 |
| मित्राला पत्र कसे लिहावे? | 14.62 | 17.31 | +2.69 |
| झाडे लावणे का महत्त्वाचे आहे? | 11.83 | 20.43 | +8.60 |
| संगणक म्हणजे काय आणि त्याचा उपयोग काय? | 17.41 | 20.22 | +2.81 |
| वेळेचे व्यवस्थापन कसे करावे? | 21.77 | 25.49 | +3.72 |
| चांगल्या विद्यार्थ्याचे गुण कोणते असतात? | 14.84 | 18.24 | +3.40 |
| **Corpus Average** | **15.39** | **25.83** | **+10.44** |

---

## Project Structure

```
marathi-llm/
├── config.py            # Single source of truth for all hyperparameters
├── train.py             # QLoRA fine-tuning with Unsloth + SFTTrainer
├── check_dataset.py     # Dataset inspection and token length analysis
├── plot_loss.py         # Training vs validation loss curve
├── infer.py             # Base vs fine-tuned side-by-side comparison
├── evaluate.py          # chrF++ scoring (requires sacrebleu)
├── chat.py              # Interactive terminal chat with fine-tuned model
└── export_gguf.py       # Merge adapter + export to GGUF for Ollama
```

---

## Setup

### Requirements

```bash
# Inside distrobox (training)
pip install unsloth trl transformers datasets sacrebleu

# On host (Ollama chat)
# Install Ollama: https://ollama.com
```

### Installation

```bash
git clone https://github.com/DragonLegend73/marathi-llm.git
cd marathi-llm
pip install unsloth trl transformers datasets sacrebleu
```

---

## Usage

### 1. Inspect the dataset
```bash
python check_dataset.py
```
Prints token length distribution and sample rows from the Marathi dataset.

### 2. Fine-tune
```bash
python train.py
```
Trains for 1 epoch using QLoRA. Saves LoRA adapter to `./outputs/lora_adapter/` and loss log to `./evals/loss_log.csv`.

### 3. Plot loss curve
```bash
python plot_loss.py
```
Generates `evals/loss_curve.png` showing training and validation loss.

### 4. Compare base vs fine-tuned
```bash
python infer.py
```
Runs 10 Marathi prompts through both models and saves `evals/comparison.csv`.

### 5. Compute chrF++ scores
```bash
python evaluate.py
```
Requires `evals/comparison.csv` from `infer.py`. Saves `evals/scores.csv`.

### 6. Chat in terminal
```bash
python chat.py
```
Interactive Marathi chat using the fine-tuned LoRA adapter.

### 7. Export to Ollama
```bash
python export_gguf.py
cd outputs_gguf
ollama create marathi-llm -f Modelfile
ollama run marathi-llm
```

---

## Key Design Decisions

**Why Qwen2.5-1.5B over Llama-3.2-3B?** Qwen2.5 has stronger multilingual pretraining coverage and fits in 4 GB VRAM at 4-bit quantisation. Llama's English-dominant corpus provides a weaker Marathi baseline.

**Why `LORA_DROPOUT=0`?** Unsloth disables its fast kernel patching when dropout > 0, which causes 0 LoRA layers to be patched and kills fine-tuning effectiveness.

**Why `MAX_SEQ_LEN=512`?** Devanagari script tokenises at ~5× the token count of equivalent English text. p90 of the dataset is 781 tokens. Testing at 768 and 640 caused OOM on the RTX 3050 Ti. 512 filtered 33.7% of samples but kept training stable.

**Why `translated_text` column?** The dataset contains both original English (`instruction`/`output`) and a pre-formatted Marathi translation (`translated_text`). Training on the English columns would fine-tune on English, defeating the purpose.

---

## Limitations

- **Factual accuracy** — the model learned Marathi fluency, not factual grounding. Some factual answers are incorrect.
- **Sequence truncation** — 33.7% of samples were filtered at 512 tokens, biasing training toward shorter responses.
- **Single epoch** — hardware constraints limited training to 1 epoch.
- **chrF++ reference sensitivity** — semantically correct but differently phrased outputs score lower than deserved.

---

## Hugging Face

The trained LoRA adapter is publicly available:

👉 [DragonLegend/marathi-qwen2.5-lora](https://huggingface.co/DragonLegend/marathi-qwen2.5-lora)

```python
from peft import PeftModel
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length=512,
    dtype=torch.float16,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "DragonLegend/marathi-qwen2.5-lora")
```

---

## References

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Popovic (2017). *chrF++: Words Helping Character N-grams.* WMT 2017.
- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster LLM fine-tuning.
- [rachittshah/alpaca-marahti](https://huggingface.co/datasets/rachittshah/alpaca-marahti) — Marathi Alpaca dataset.
