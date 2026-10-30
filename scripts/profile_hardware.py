"""
scripts/profile_hardware.py — Task 0.6: Hardware profiling on RTX 3050 Ti Laptop GPU
-----------------------------------------------------------------------------------
Runs 100 training steps of the current v0 config (QLoRA 4-bit, batch 1, accum 8, seq 512).
Logs:
- steps/s (training throughput)
- Peak VRAM allocated and reserved (GB)
- Thermals and power draw via nvidia-smi
Saves output to results/hw_profile.json
"""

import os
import time
import json
import subprocess
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer

# Environment settings
os.environ["UNSLOTH_DISABLE_FUSED_LOSS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DATASET_NAME = "rachittshah/alpaca-marahti"
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
STEPS_TO_RUN = 100

def get_gpu_telemetry():
    try:
        res = subprocess.run([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "temperature_c": float(parts[0]),
            "power_draw_w": float(parts[1]),
            "gpu_utilization_pct": float(parts[2]),
            "memory_used_mb": float(parts[3]),
            "memory_total_mb": float(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("="*70)
    print("TASK 0.6: HARDWARE PROFILING (100 TRAINING STEPS)")
    print("="*70)

    initial_telemetry = get_gpu_telemetry()
    print(f"Initial GPU State: {initial_telemetry}")

    print("\n1. Loading model & tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    print("2. Attaching LoRA (r=16, alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        use_gradient_checkpointing=True,
        random_state=42,
    )

    print("3. Loading training split...")
    with open("data/splits/s0_train_ids.json", "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    train_ids = set(train_meta["row_ids"][:2000])

    ds = load_dataset(DATASET_NAME, split="train")
    eos_token = tokenizer.eos_token or "<|im_end|>"

    samples = []
    dropped_count = 0
    for idx in train_ids:
        row = ds[idx]
        trans = (row.get("translated_text") or "").strip()
        text = f"{trans}{eos_token}"
        input_ids = tokenizer(text)["input_ids"]
        if len(input_ids) <= MAX_SEQ_LEN:
            samples.append({"text": text})
        else:
            dropped_count += 1
        if len(samples) >= 1200:
            break

    print(f"Training samples loaded: {len(samples)} (dropped {dropped_count} exceeding 512 tokens)")

    from datasets import Dataset
    train_dataset = Dataset.from_list(samples)

    output_dir = "/tmp/hw_profile_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    from trl import SFTConfig

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=STEPS_TO_RUN,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        optim="paged_adamw_8bit",
        report_to="none",
        warmup_steps=0.05,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=args,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nStarting 100-step training profile...")
    t0 = time.perf_counter()
    train_result = trainer.train()
    t1 = time.perf_counter()

    elapsed_time = t1 - t0
    steps_per_sec = STEPS_TO_RUN / elapsed_time
    samples_per_sec = (STEPS_TO_RUN * BATCH_SIZE * GRAD_ACCUM) / elapsed_time

    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    final_telemetry = get_gpu_telemetry()

    print("\n" + "="*70)
    print("PROFILING RESULTS:")
    print(f"  Wall time for 100 steps : {elapsed_time:.2f} s")
    print(f"  Throughput              : {steps_per_sec:.3f} steps/s  ({samples_per_sec:.2f} samples/s)")
    print(f"  Peak VRAM Allocated     : {peak_alloc_gb:.2f} GB")
    print(f"  Peak VRAM Reserved      : {peak_res_gb:.2f} GB")
    print(f"  Final Temperature       : {final_telemetry.get('temperature_c', 'N/A')} °C")
    print(f"  Final Power Draw        : {final_telemetry.get('power_draw_w', 'N/A')} W")
    print(f"  Est. Time for 12k run   : {((12000 / 8) / steps_per_sec) / 60:.1f} minutes")
    print("="*70)

    # Save to results/hw_profile.json
    os.makedirs("results", exist_ok=True)
    res_path = "results/hw_profile.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": 4.0,
            "steps_profiled": STEPS_TO_RUN,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "max_seq_len": MAX_SEQ_LEN,
            "elapsed_seconds": round(elapsed_time, 2),
            "steps_per_second": round(steps_per_sec, 4),
            "samples_per_second": round(samples_per_sec, 3),
            "peak_vram_allocated_gb": round(peak_alloc_gb, 3),
            "peak_vram_reserved_gb": round(peak_res_gb, 3),
            "initial_telemetry": initial_telemetry,
            "final_telemetry": final_telemetry,
            "estimated_minutes_for_12k_run": round(((12000 / 8) / steps_per_sec) / 60, 2),
        }, f, indent=2)

    print(f"Saved profile to {res_path}")

if __name__ == "__main__":
    main()
