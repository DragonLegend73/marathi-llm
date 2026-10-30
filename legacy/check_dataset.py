# legacy, superseded by src/marathi_llm/data/stats.py and src/marathi_llm/eval/fertility.py
"""
check_dataset.py — Inspect the Marathi Alpaca dataset
------------------------------------------------------
Prints column names, sample count, a few examples,
and token-length distribution so you can tune MAX_SEQ_LEN.
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from config import MODEL_NAME, DATASET_NAME, MAX_SEQ_LEN
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

print("Loading tokenizer …")
_, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= MAX_SEQ_LEN,
    dtype         = torch.float16,
    load_in_4bit  = True,
)

print("Loading dataset …")
ds = load_dataset(DATASET_NAME, split="train")

print(f"\nDataset size   : {len(ds):,}")
print(f"Columns        : {ds.column_names}")

# Token length distribution
lengths = []
# Replace the token counting loop with this:
for row in ds.select(range(min(500, len(ds)))):
    text = (row.get("translated_text") or "").strip()
    lengths.append(len(tokenizer.encode(text)))

lengths.sort()
p50 = lengths[len(lengths)//2]
p90 = lengths[int(len(lengths)*0.9)]
p95 = lengths[int(len(lengths)*0.95)]

print(f"\nToken length stats (first 500 samples):")
print(f"  Median (p50): {p50}")
print(f"  p90         : {p90}")
print(f"  p95         : {p95}")
print(f"  Max          : {lengths[-1]}")
print(f"\n  → Recommended MAX_SEQ_LEN ≥ {p90} to cover 90% of samples")

# Show 3 samples
print("\n" + "="*60)
print("  3 Sample Rows")
print("="*60)
for i in range(3):
    row = ds[i]
    instruction = (row.get('instruction') or '')[:120]
    user_input  = (row.get('input') or '')[:80]
    output      = (row.get('output') or '')[:120]

    print(f"\n[{i+1}] Instruction: {instruction}")
    print(f"    Input      : {user_input}")
    print(f"    Output     : {output}")
