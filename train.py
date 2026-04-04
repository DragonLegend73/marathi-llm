"""
train.py — Marathi LLM Fine-Tuning with QLoRA + Unsloth
---------------------------------------------------------
"""

import os, csv, torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from config import *

# ── Environment knobs (tuned for RTX 3050 Ti 4 GB) ───────────
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"]   = "1"
os.environ["TORCHDYNAMO_DISABLE"]          = "1"
os.environ["TORCH_COMPILE"]                = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]      = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"

EOS_TOKEN = None

def get_format_fn(eos_token):
    def format_sample(example):
        translated = (example.get("translated_text") or "").strip()
        text = f"{translated}{eos_token}"
        return {"text": text}
    return format_sample

# ── Load model ───────────────────────────────────────────────
print("Loading model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= MAX_SEQ_LEN,
    dtype         = torch.float16,
    load_in_4bit  = True,
)
EOS_TOKEN = tokenizer.eos_token

# ── Attach LoRA ───────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                    = LORA_R,
    lora_alpha           = LORA_ALPHA,
    lora_dropout         = LORA_DROPOUT,
    use_gradient_checkpointing = True,
    random_state         = 42,
)

# ── Dataset ───────────────────────────────────────────────────
print("Loading dataset …")
dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset columns: {dataset.column_names}")
print(f"Dataset size   : {len(dataset):,}")

dataset = dataset.map(get_format_fn(EOS_TOKEN), remove_columns=dataset.column_names)
original_size = len(dataset)

print("Filtering out sequences exceeding MAX_SEQ_LEN...")
def filter_length(example):
    return len(tokenizer(example["text"])["input_ids"]) <= MAX_SEQ_LEN

dataset = dataset.filter(filter_length, num_proc=os.cpu_count() or 4)

filtered_size = len(dataset)
print(f"Dataset size after filtering: {filtered_size:,} ({(1 - filtered_size/original_size)*100:.1f}% dropped)")

split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_data = split_dataset["train"]
eval_data  = split_dataset["test"]

# ── Trainer ───────────────────────────────────────────────────
args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = NUM_EPOCHS,    # Swapped max_steps for full dataset traversal
    eval_strategy               = "steps",
    eval_steps                  = EVAL_STEPS,    # Tied to new configuration scaling
    learning_rate               = LEARNING_RATE,
    fp16                        = True,
    logging_steps               = LOGGING_STEPS,
    save_steps                  = SAVE_STEPS,
    optim                       = "paged_adamw_8bit",
    report_to                   = "none",
    warmup_ratio                = 0.05,          # Allows LR to scale correctly across 6000+ steps
)

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_data,
    eval_dataset  = eval_data,
    dataset_text_field = "text",
    max_seq_length= MAX_SEQ_LEN,
    packing       = False,
    args          = args,
)

# ── Train ─────────────────────────────────────────────────────
print("Starting training …")
torch.cuda.empty_cache()
result = trainer.train()

# ── Save loss log ─────────────────────────────────────────────
os.makedirs("evals", exist_ok=True)
loss_log_path = "evals/loss_log.csv"
history = trainer.state.log_history
with open(loss_log_path, "w", newline="") as f:
    # Add eval_loss to fieldnames
    writer = csv.DictWriter(f, fieldnames=["step", "loss", "eval_loss"])
    writer.writeheader()
    for entry in history:
        # Match steps to unify training and eval rows
        row = {"step": entry["step"]}
        if "loss" in entry: row["loss"] = round(entry["loss"], 4)
        if "eval_loss" in entry: row["eval_loss"] = round(entry["eval_loss"], 4)
        writer.writerow(row)
print(f"Loss log saved → {loss_log_path}")

# ── Save LoRA adapter ─────────────────────────────────────────
print(f"Saving LoRA adapter → {ADAPTER_DIR}")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Done! ✓")
