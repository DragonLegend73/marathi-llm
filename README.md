# Developing a Marathi LLM using QLoRA + Unsloth

Fine-tuning **Llama-3.2-3B-Instruct** on a 52 K Marathi instruction dataset
using **QLoRA (4-bit quantisation + Low-Rank Adaptation)** and the **Unsloth**
optimised training library — entirely on a 4 GB consumer GPU.

---

## Project Structure

```
marathi-llm/
├── config.py          ← Single source of truth for ALL hyperparameters
├── train.py           ← Main training script (QLoRA + Alpaca prompts)
├── infer.py           ← Interactive REPL for testing the model
├── evaluate.py        ← BLEU + Perplexity evaluation on held-out set
├── plot_loss.py       ← Generates loss curve PNG from training log
├── check_dataset.py   ← Dataset stats + token-length distribution
├── requirements.txt
└── evals/
    ├── loss_log.csv       ← Written by train.py
    ├── loss_curve.png     ← Written by plot_loss.py
    └── eval_results.json  ← Written by evaluate.py
```

---

## Quick Start

```bash
# 1. Install dependencies (on a CUDA machine / Colab)
pip install -r requirements.txt

# 2. Check the dataset and choose a good MAX_SEQ_LEN
python check_dataset.py

# 3. Train (edit config.py first if you want to change hyperparams)
python train.py

# 4. Chat with the fine-tuned model
python infer.py

# 5. Compare base vs fine-tuned side-by-side
python infer.py --base        # base model
python infer.py               # fine-tuned model

# 6. Quantitative evaluation (BLEU + Perplexity)
python evaluate.py --n 50

# 7. Plot the loss curve
python plot_loss.py
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | Llama-3.2-3B-Instruct | Best open instruction model that fits in 4 GB |
| Quantization | 4-bit (bitsandbytes NF4) | Cuts VRAM from ~6 GB → ~3.5 GB |
| LoRA rank | r=8 | Good capacity/VRAM trade-off |
| Prompt format | Alpaca instruction template | Teaches the model the Q→A pattern explicitly |
| Effective batch | 4 (1×4 grad accum) | Smoother gradient estimates without OOM |
| Sequence length | 512 | Covers 90%+ of dataset samples |

---

## Alpaca Prompt Template

```
### निर्देश (Instruction):
<Marathi instruction>

### इनपुट (Input):
<optional context>

### उत्तर (Response):
<model output>
```

Using this template (instead of raw `translated_text`) teaches the model the
**instruction-following** format, which is the core goal of the project.

---

## Results

| Metric | Value |
|--------|-------|
| Training steps | 500 |
| Initial loss | ~1.59 |
| Final loss | ~0.89 |
| Trainable params | ~0.19% of 3B |
| GPU VRAM used | < 4 GB |

---

## Limitations

- Context window capped at 512 tokens (longer Marathi texts are truncated)
- Only partial training completed (full-epoch training needs ~10× more steps)
- BLEU is a weak metric for generative tasks; human evaluation is recommended

---

## Future Work

- Full-epoch training (3–5 epochs on all 52 K samples)
- Push adapter to Hugging Face Hub for sharing
- GGUF export for CPU inference via llama.cpp
- IndicGLUE benchmark for standardised Marathi NLP evaluation
