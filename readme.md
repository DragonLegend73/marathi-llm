# Marathi LLM Fine-Tuning

Fine-tuning Qwen2.5-1.5B-Instruct on a Marathi Alpaca dataset using QLoRA + Unsloth.

## Hardware
- NVIDIA RTX 3050 Ti Laptop GPU (4GB VRAM)
- Bazzite Linux

## Model
- Base: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
- Dataset: `rachittshah/alpaca-marahti` (52,002 samples)
- Method: QLoRA (r=16, alpha=32)

## Setup
```bash
pip install unsloth trl transformers datasets sacrebleu
```

## Usage
```bash
python train.py       # Fine-tune
python infer.py       # Compare base vs fine-tuned
python evaluate.py    # chrF++ scores
python chat.py        # Interactive chat
python export_gguf.py # Export for Ollama
```
