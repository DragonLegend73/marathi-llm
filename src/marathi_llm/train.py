"""Config-driven QLoRA fine-tuning script with response-only loss masking and 4 GB VRAM support."""

import os

# 4 GB VRAM stability environment variables (must be set before torch/unsloth init)
os.environ.setdefault("UNSLOTH_DISABLE_FUSED_LOSS", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from marathi_llm.generate import format_plain_prompt


ALL_LINEAR_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

ATTENTION_ONLY_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def get_git_commit() -> str:
    """Returns current HEAD git commit hash or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Loads a model/training YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_target_modules(spec: Union[str, List[str]]) -> List[str]:
    """Maps symbolic module group names to explicit linear projection names."""
    if isinstance(spec, list):
        return list(spec)
    key = str(spec).strip().lower()
    if key in ("all-linear", "all_linear", "all"):
        return list(ALL_LINEAR_MODULES)
    if key in ("attention-only", "attention_only", "attn"):
        return list(ATTENTION_ONLY_MODULES)
    return [m.strip() for m in key.split(",") if m.strip()]


def format_user_content(instruction: str, input_text: str = "") -> str:
    """Combines instruction and optional input into a clean user message."""
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inp:
        return f"{inst}\n\n{inp}"
    return inst


def load_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    """Loads JSONL rows from disk."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt_completion_records(
    rows: List[Dict[str, Any]],
    has_chat_template: bool = True,
) -> List[Dict[str, Any]]:
    """
    Converts standardized (instruction, input, response) rows into TRL 1.x
    prompt-completion records so `SFTConfig(completion_only_loss=True)` masks
    prompt tokens natively using the model's own chat template.
    """
    records: List[Dict[str, Any]] = []
    for row in rows:
        user_content = format_user_content(
            row.get("instruction", ""),
            row.get("input", ""),
        )
        response = (row.get("response", "") or "").strip()
        if not user_content or not response:
            continue

        if has_chat_template:
            records.append(
                {
                    "prompt": [{"role": "user", "content": user_content}],
                    "completion": [{"role": "assistant", "content": response}],
                }
            )
        else:
            records.append(
                {
                    "prompt": format_plain_prompt(row.get("instruction", ""), row.get("input", "")),
                    "completion": response,
                }
            )
    return records


def format_record_for_length_check(
    record: Dict[str, Any],
    tokenizer: Any,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Renders a prompt-completion record into exact full text for length measurement."""
    if isinstance(record["prompt"], list):
        from trl.data_utils import apply_chat_template

        kwargs: Dict[str, Any] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            rendered = apply_chat_template(record, tokenizer, **kwargs)
        except TypeError:
            rendered = apply_chat_template(record, tokenizer)
        return rendered["prompt"] + rendered["completion"]

    eos = getattr(tokenizer, "eos_token", "") or ""
    return f"{record['prompt']}{record['completion']}{eos}"


def filter_by_max_length(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    enable_thinking: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Filters prompt-completion records exceeding `max_seq_len` under the target
    tokenizer and returns truncation/length statistics.
    """
    kept: List[Dict[str, Any]] = []
    kept_lengths: List[int] = []
    total = len(records)

    for rec in records:
        full_text = format_record_for_length_check(
            rec, tokenizer, enable_thinking=enable_thinking
        )
        tok_len = len(tokenizer.encode(full_text, add_special_tokens=False))
        if tok_len <= max_seq_len:
            kept.append(rec)
            kept_lengths.append(tok_len)

    dropped = total - len(kept)
    pct_kept = (len(kept) / total * 100.0) if total > 0 else 0.0
    mean_tokens = (sum(kept_lengths) / len(kept_lengths)) if kept_lengths else 0.0

    stats = {
        "total": total,
        "kept": len(kept),
        "dropped": dropped,
        "pct_examples_kept": round(pct_kept, 2),
        "mean_tokens_kept": round(mean_tokens, 2),
        "max_seq_len": max_seq_len,
    }
    return kept, stats


def find_latest_checkpoint(checkpoints_dir: Path) -> Optional[str]:
    """Returns path to the highest-step checkpoint directory if present."""
    if not checkpoints_dir.exists():
        return None
    ckpts = []
    for p in checkpoints_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                step = int(p.name.split("-", 1)[1])
                ckpts.append((step, p))
            except ValueError:
                continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])


def write_loss_csv(log_history: List[Dict[str, Any]], output_path: Path) -> None:
    """Writes training and eval loss curves from Trainer log_history to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "epoch", "loss", "eval_loss", "learning_rate"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in log_history:
            if "loss" not in entry and "eval_loss" not in entry:
                continue
            row: Dict[str, Any] = {
                "step": entry.get("step", ""),
                "epoch": round(entry["epoch"], 4) if "epoch" in entry else "",
                "loss": round(entry["loss"], 5) if "loss" in entry else "",
                "eval_loss": round(entry["eval_loss"], 5) if "eval_loss" in entry else "",
                "learning_rate": entry.get("learning_rate", ""),
            }
            writer.writerow(row)


def infer_eval_data_path(train_data_path: str) -> Optional[str]:
    """Infers sibling dev JSONL from train data path if present."""
    p = Path(train_data_path)
    name = p.name
    if "_12k.jsonl" in name:
        candidate = p.with_name(name.replace("_12k.jsonl", "_dev500.jsonl"))
        if candidate.exists():
            return str(candidate)
    if "_train.jsonl" in name:
        candidate = p.with_name(name.replace("_train.jsonl", "_dev.jsonl"))
        if candidate.exists():
            return str(candidate)
    prefix = name.split("_")[0]
    for sibling in sorted(p.parent.glob(f"{prefix}*dev*.jsonl")):
        return str(sibling)
    return None


def get_gpu_temp_c() -> Optional[float]:
    """Reads current GPU temperature in Celsius via nvidia-smi if available."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def resolve_profile_data_path(profile_data: Optional[str], max_seq_len: int) -> Optional[str]:
    """Resolves `--profile-data worst_case` to the E1 worst-case JSONL pool for `max_seq_len`."""
    if not profile_data:
        return None
    if profile_data == "worst_case":
        candidate = Path("results/length_census/worst_case_pools") / f"worst_case_{max_seq_len}.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Worst-case profile pool not found at {candidate}. "
                "Run scripts/length_census.py first to generate worst-case length pools."
            )
        return str(candidate)
    return profile_data


def train_from_config(
    model_config_path: str,
    data_path: Optional[str],
    run_id: str,
    eval_data_path: Optional[str] = None,
    output_root: str = "results/runs",
    max_seq_len_override: Optional[int] = None,
    profile_data: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    lora_r_override: Optional[int] = None,
    use_rslora: bool = False,
    target_modules_override: Optional[str] = None,
    response_only_loss_override: Optional[bool] = None,
    per_device_batch_override: Optional[int] = None,
    grad_accum_override: Optional[int] = None,
    resume: bool = False,
    profile_only: bool = False,
) -> Dict[str, Any]:
    """Runs config-driven QLoRA training and saves adapter, logs, and summary."""
    # Import Unsloth BEFORE torch/transformers/trl for kernel patching
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    cfg = load_model_config(model_config_path)
    model_hf_id = cfg["hf_id"]
    max_seq_len = int(max_seq_len_override if max_seq_len_override is not None else cfg.get("max_seq_len", 512))
    resolved_data_path = resolve_profile_data_path(profile_data, max_seq_len) or data_path
    if not resolved_data_path:
        raise ValueError("Either --data or --profile-data must be provided.")
    data_path = resolved_data_path
    prequantized = bool(cfg.get("prequantized", False))
    load_in_4bit = bool(cfg.get("load_in_4bit", True))
    enable_thinking = cfg.get("enable_thinking", None)

    lora_cfg = dict(cfg.get("lora", {}))
    lora_r = int(lora_r_override if lora_r_override is not None else lora_cfg.get("r", 16))
    lora_alpha = int(lora_cfg.get("alpha", lora_r * 2))
    lora_dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = resolve_target_modules(
        target_modules_override
        if target_modules_override is not None
        else lora_cfg.get("target_modules", "all-linear")
    )

    train_cfg = dict(cfg.get("train", {}))
    lr = float(lr_override if lr_override is not None else train_cfg.get("lr", 2.0e-4))
    epochs = float(train_cfg.get("epochs", 1))
    per_device_batch = int(
        per_device_batch_override
        if per_device_batch_override is not None
        else train_cfg.get("per_device_batch", 2)
    )
    grad_accum = int(
        grad_accum_override
        if grad_accum_override is not None
        else train_cfg.get("grad_accum", 4)
    )
    grad_ckpt = train_cfg.get("gradient_checkpointing", "unsloth")
    response_only_loss = bool(
        response_only_loss_override
        if response_only_loss_override is not None
        else train_cfg.get("response_only_loss", True)
    )
    seed = int(seed_override if seed_override is not None else train_cfg.get("seed", 0))
    save_steps = int(train_cfg.get("save_steps", 250))
    eval_steps = int(train_cfg.get("eval_steps", 250))
    logging_steps = int(train_cfg.get("logging_steps", 25))
    max_steps = 100 if profile_only else (max_steps_override if max_steps_override is not None else -1)

    run_dir = Path(output_root) / run_id
    checkpoints_dir = run_dir / "checkpoints"
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_eval_path = eval_data_path if eval_data_path is not None else infer_eval_data_path(data_path)

    print("\n==========================================")
    print(f"Training Run: {run_id}")
    print(f"Model: {model_hf_id} (prequantized={prequantized}, max_seq_len={max_seq_len})")
    print(f"Data:  {data_path} (eval={resolved_eval_path or 'none'})")
    print(f"LoRA:  r={lora_r}, alpha={lora_alpha}, rslora={use_rslora}, modules={len(target_modules)}")
    print(
        f"Train: batch={per_device_batch}x{grad_accum} (eff={per_device_batch * grad_accum}), "
        f"lr={lr}, response_only_loss={response_only_loss}, seed={seed}, max_steps={max_steps}"
    )
    print("==========================================\n")

    # 1. Load model & tokenizer
    use_bf16 = is_bfloat16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_hf_id,
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=grad_ckpt,
        random_state=seed,
        use_rslora=use_rslora,
    )

    # 2. Load & format datasets
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    raw_train = load_jsonl_rows(data_path)
    train_records = build_prompt_completion_records(raw_train, has_chat_template=has_chat_template)
    train_records, train_len_stats = filter_by_max_length(
        train_records, tokenizer, max_seq_len=max_seq_len, enable_thinking=enable_thinking
    )
    print(
        f"Train length filter: kept {train_len_stats['kept']:,}/{train_len_stats['total']:,} "
        f"({train_len_stats['pct_examples_kept']:.2f}%, mean={train_len_stats['mean_tokens_kept']:.1f} tok)"
    )

    train_dataset = Dataset.from_list(train_records)

    eval_dataset = None
    eval_len_stats = None
    if resolved_eval_path and Path(resolved_eval_path).exists():
        raw_eval = load_jsonl_rows(resolved_eval_path)
        eval_records = build_prompt_completion_records(raw_eval, has_chat_template=has_chat_template)
        eval_records, eval_len_stats = filter_by_max_length(
            eval_records, tokenizer, max_seq_len=max_seq_len, enable_thinking=enable_thinking
        )
        print(
            f"Eval length filter:  kept {eval_len_stats['kept']:,}/{eval_len_stats['total']:,} "
            f"({eval_len_stats['pct_examples_kept']:.2f}%, mean={eval_len_stats['mean_tokens_kept']:.1f} tok)"
        )
        if eval_records:
            eval_dataset = Dataset.from_list(eval_records)

    # 3. Save resolved config snapshot
    resolved_config = {
        "run_id": run_id,
        "git_commit": get_git_commit(),
        "model_config": model_config_path,
        "hf_id": model_hf_id,
        "quant": "prequantized-bnb-4bit" if prequantized else "load_in_4bit-on-load",
        "data_path": data_path,
        "eval_data_path": resolved_eval_path,
        "max_seq_len": max_seq_len,
        "has_chat_template": has_chat_template,
        "enable_thinking": enable_thinking,
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "use_rslora": use_rslora,
            "target_modules": target_modules,
        },
        "train": {
            "lr": lr,
            "epochs": epochs,
            "max_steps": max_steps,
            "per_device_batch": per_device_batch,
            "grad_accum": grad_accum,
            "effective_batch": per_device_batch * grad_accum,
            "gradient_checkpointing": grad_ckpt,
            "response_only_loss": response_only_loss,
            "seed": seed,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "bf16": use_bf16,
            "fp16": not use_bf16,
        },
        "train_length_stats": train_len_stats,
        "eval_length_stats": eval_len_stats,
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)

    # 4. Configure TRL 1.x SFTTrainer
    chat_kwargs = {"enable_thinking": enable_thinking} if enable_thinking is not None else None
    sft_args_kwargs: Dict[str, Any] = dict(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=logging_steps,
        save_strategy="no" if profile_only else "steps",
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="steps" if (eval_dataset is not None and not profile_only) else "no",
        eval_steps=eval_steps,
        max_length=max_seq_len,
        packing=False,
        completion_only_loss=response_only_loss,
        seed=seed,
        data_seed=seed,
        report_to="none",
        dataset_num_proc=1,
    )
    if chat_kwargs is not None:
        sft_args_kwargs["chat_template_kwargs"] = chat_kwargs

    sft_args = SFTConfig(**sft_args_kwargs)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
    )

    resume_ckpt = find_latest_checkpoint(checkpoints_dir) if resume else None
    if resume and resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")

    # 5. Train & measure hardware telemetry
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed_s = time.time() - t0

    peak_alloc_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    steps_completed = int(train_result.global_step)
    steps_per_s = steps_completed / elapsed_s if elapsed_s > 0 else 0.0
    eff_batch = per_device_batch * grad_accum
    samples_per_s = steps_per_s * eff_batch
    tokens_per_s = samples_per_s * train_len_stats["mean_tokens_kept"]
    gpu_temp_c = get_gpu_temp_c()
    full_epoch_steps = (train_len_stats["kept"] + eff_batch - 1) // eff_batch
    full_epoch_eta_min = (full_epoch_steps / steps_per_s / 60.0) if steps_per_s > 0 else 0.0

    # Extract final train & eval losses
    final_train_loss = float(train_result.training_loss)
    final_eval_loss = None
    if eval_dataset is not None and not profile_only:
        eval_metrics = trainer.evaluate()
        if "eval_loss" in eval_metrics:
            final_eval_loss = float(eval_metrics["eval_loss"])

    write_loss_csv(trainer.state.log_history, run_dir / "loss.csv")

    if not profile_only:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"Saved adapter -> {adapter_dir}")

    summary = {
        "run_id": run_id,
        "git_commit": get_git_commit(),
        "model": model_hf_id,
        "quant": resolved_config["quant"],
        "data": Path(data_path).stem,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "profile_only": profile_only,
        "train": {
            "steps": steps_completed,
            "elapsed_s": round(elapsed_s, 2),
            "final_train_loss": round(final_train_loss, 5),
            "final_dev_loss": round(final_eval_loss, 5) if final_eval_loss is not None else None,
            "pct_examples_kept": train_len_stats["pct_examples_kept"],
            "mean_tokens_kept": train_len_stats["mean_tokens_kept"],
            "per_device_batch": per_device_batch,
            "grad_accum": grad_accum,
            "peak_vram_gb": round(peak_alloc_gb, 3),
            "peak_vram_reserved_gb": round(peak_reserved_gb, 3),
            "gpu_temp_c": gpu_temp_c,
            "steps_per_s": round(steps_per_s, 4),
            "samples_per_s": round(samples_per_s, 4),
            "tokens_per_s": round(tokens_per_s, 1),
            "full_1epoch_steps": full_epoch_steps,
            "full_1epoch_eta_min": round(full_epoch_eta_min, 1),
            "response_only_loss": response_only_loss,
        },
    }

    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================")
    print(f"Training Complete: {run_id}")
    print(
        f"  Steps:        {steps_completed} in {elapsed_s:.1f}s "
        f"({steps_per_s:.3f} steps/s, {samples_per_s:.2f} samples/s, {tokens_per_s:.1f} tok/s)"
    )
    print(
        f"  Peak VRAM:    {peak_alloc_gb:.2f} GB alloc / {peak_reserved_gb:.2f} GB reserved "
        f"(temp={gpu_temp_c}°C)"
    )
    print(f"  Train Loss:   {final_train_loss:.4f} | Dev Loss: {final_eval_loss}")
    print(f"  1-Epoch ETA:  ~{full_epoch_eta_min:.1f} min ({full_epoch_steps} steps)")
    print(f"  Summary:      {run_dir / 'train_summary.json'}")
    print("==========================================\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Config-driven Marathi LLM QLoRA Trainer")
    parser.add_argument(
        "--model", "--config", dest="model", required=True,
        help="Path to model YAML config (e.g. configs/models/qwen25_1p5b.yaml)",
    )
    parser.add_argument("--data", default=None, help="Path to training JSONL (e.g. data/processed/s0_alpaca_12k.jsonl)")
    parser.add_argument("--profile-data", default=None, help="'worst_case' or path to JSONL for --profile-only runs")
    parser.add_argument("--eval-data", default=None, help="Optional dev JSONL (auto-inferred if sibling *_dev500.jsonl exists)")
    parser.add_argument("--run-id", required=True, help="Unique run identifier under results/runs/<run_id>")
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override model config max_seq_len")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument("--target-modules", default=None, help="'all-linear', 'attention-only', or comma-separated list")
    parser.add_argument("--response-only-loss", dest="response_only_loss", action="store_true", default=None)
    parser.add_argument("--no-response-only-loss", dest="response_only_loss", action="store_false")
    parser.add_argument("--per-device-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in run directory")
    parser.add_argument("--profile-only", action="store_true", help="Run 100 steps only to measure VRAM, speed, and ETA")
    args = parser.parse_args()

    train_from_config(
        model_config_path=args.model,
        data_path=args.data,
        run_id=args.run_id,
        eval_data_path=args.eval_data,
        output_root=args.output_root,
        max_seq_len_override=args.max_seq_len,
        profile_data=args.profile_data,
        max_steps_override=args.max_steps,
        seed_override=args.seed,
        lr_override=args.lr,
        lora_r_override=args.lora_r,
        use_rslora=args.use_rslora,
        target_modules_override=args.target_modules,
        response_only_loss_override=args.response_only_loss,
        per_device_batch_override=args.per_device_batch,
        grad_accum_override=args.grad_accum,
        resume=args.resume,
        profile_only=args.profile_only,
    )


if __name__ == "__main__":
    main()
