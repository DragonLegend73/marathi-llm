"""
export_gguf.py — Merge LoRA adapter into base model and export to GGUF
----------------------------------------------------------------------
Run this after training is complete.
Output: ./outputs/marathi-llm.gguf  (Q4_K_M quantization)

Then load into Ollama:
    ollama create marathi-llm -f Modelfile
    ollama run marathi-llm

Usage:  python export_gguf.py
"""

import os, torch
from unsloth import FastLanguageModel
from config import MODEL_NAME, MAX_SEQ_LEN, ADAPTER_DIR, OUTPUT_DIR

MERGED_DIR = os.path.join(OUTPUT_DIR, "merged_model")
GGUF_PATH  = os.path.join(OUTPUT_DIR, "marathi-llm.gguf")
MODELFILE  = os.path.join(OUTPUT_DIR, "Modelfile")

# ── Step 1: Load base + adapter ───────────────────────────────
print("Loading base model + LoRA adapter …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = ADAPTER_DIR,   # Unsloth can load adapter dir directly
    max_seq_length= MAX_SEQ_LEN,
    dtype         = torch.float16,
    load_in_4bit  = True,
)
FastLanguageModel.for_inference(model)

# ── Step 2: Export to GGUF ────────────────────────────────────
# Q4_K_M is the best quality/size tradeoff for 1.5B models
print(f"Exporting to GGUF (Q4_K_M) → {GGUF_PATH}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method = "q4_k_m",
)
print("GGUF export complete ✓")

# ── Step 3: Write Modelfile for Ollama ────────────────────────
# Find the actual .gguf file Unsloth created
gguf_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".gguf")]
if not gguf_files:
    print("Warning: no .gguf file found in output dir. Check export above.")
    gguf_filename = "marathi-llm.gguf"
else:
    gguf_filename = gguf_files[0]
    print(f"Found GGUF file: {gguf_filename}")

modelfile_content = f"""FROM ./{gguf_filename}

# System prompt — tells the model to respond in Marathi
SYSTEM "तुम्ही एक मराठी भाषेतील सहाय्यक आहात. नेहमी मराठीत उत्तर द्या."

# Generation parameters
PARAMETER temperature {0.7}
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
"""

with open(MODELFILE, "w", encoding="utf-8") as f:
    f.write(modelfile_content)
print(f"Modelfile written → {MODELFILE}")

# ── Step 4: Print Ollama instructions ────────────────────────
print("\n" + "=" * 60)
print("Next steps to run in Ollama:")
print("=" * 60)
print(f"\n  cd {OUTPUT_DIR}")
print( "  ollama create marathi-llm -f Modelfile")
print( "  ollama run marathi-llm")
print("\nOr use the OpenAI-compatible API:")
print( "  ollama serve   # starts on http://localhost:11434")
print("=" * 60)
