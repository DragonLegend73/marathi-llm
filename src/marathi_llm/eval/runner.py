"""Unified evaluation runner executing benchmark suite and producing metrics.json."""

import os
import gc
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import transformers
import peft
import sacrebleu

from marathi_llm.generate import load_model_and_tokenizer
from marathi_llm.eval.milu import evaluate_milu
from marathi_llm.eval.flores import evaluate_flores_direction
from marathi_llm.eval.bpc import compute_bpc, load_marathi_reference_texts
from marathi_llm.eval.ifeval import run_indicifeval
from marathi_llm.eval.auto_metrics import evaluate_generations


def get_git_commit() -> str:
    """Returns current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_indicifeval_commit() -> str:
    """Returns git commit hash of IndicIFEval submodule/clone."""
    try:
        res = subprocess.run(["git", "-C", "IndicIFEval", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "1bb5f1bc4936cb20b8544e185bfd7cbed8f31464"


def save_metrics_checkpoint(metrics: Dict[str, Any], path: Path):
    """Saves metrics incrementally to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def run_full_evaluation(
    model_id: str,
    adapter_path: Optional[str] = None,
    run_id: str = "eval_run",
    output_dir: str = "results/runs",
    batch_size: int = 2,
    prompt_format: str = "chat",
    prequantized: Optional[bool] = None,
    force: bool = False,
    run_milu: bool = True,
    run_flores: bool = True,
    run_bpc: bool = True,
    run_ifeval: bool = True,
    ifeval_limit: Optional[int] = None,
    bpc_limit: Optional[int] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Runs the unified evaluation suite on RTX 3050 Ti under 4 GB VRAM."""
    run_dir = Path(output_dir) / run_id
    if run_dir.exists() and not force:
        # Check if already completed
        if (run_dir / "metrics.json").exists():
            raise FileExistsError(f"Run directory '{run_dir}' already exists. Use --force to overwrite.")

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"

    metrics = {
        "run_id": run_id,
        "git_commit": get_git_commit(),
        "indicifeval_commit": get_indicifeval_commit(),
        "model": model_id,
        "adapter": adapter_path or "none",
        "prompt_format": prompt_format,
        "decoding": {
            "strategy": "greedy",
            "temperature": None,
            "repetition_penalty": 1.0,
            "max_new_tokens": 256,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "sacrebleu": sacrebleu.__version__,
            "device": device,
            "batch_size": batch_size,
        },
    }

    # Initialize checkpoint
    save_metrics_checkpoint(metrics, metrics_path)

    print(f"\n==========================================")
    print(f"Starting Evaluation for: {run_id}")
    print(f"Model: {model_id} | Adapter: {adapter_path} | Format: {prompt_format}")
    print(f"==========================================\n")

    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        adapter_path=adapter_path,
        device=device,
        prequantized=prequantized,
    )

    metrics["quant"] = "prequantized-bnb-4bit" if getattr(model, "is_loaded_in_4bit", False) else "4bit"
    metrics["adapter_loaded"] = isinstance(model, peft.PeftModel) if adapter_path else False

    all_generations = []

    # 1. MILU Marathi (validation split: 810 questions)
    if run_milu:
        print("\n--- [1/4] Running MILU (Marathi MCQ, validation split) ---")
        milu_res = evaluate_milu(
            model=model,
            tokenizer=tokenizer,
            split="validation",
            batch_size=batch_size,
            prompt_format=prompt_format,
            device=device,
        )
        s = milu_res["summary"]
        metrics["milu_mr"] = {
            "split": s["split"],
            "acc": s["accuracy"],
            "ci95": s["ci_95"],
            "chance_level": s["chance_level"],
            "mean_option_letters_prob_mass": s.get("mean_option_letters_prob_mass", 0.0),
            "by_domain": {d: v["accuracy"] for d, v in s["domain_accuracy"].items()},
            "pred_letter_dist": s["predicted_letter_distribution"],
            "n": s["n_samples"],
            "shots": 0,
        }
        with open(run_dir / "milu_validation_records.json", "w", encoding="utf-8") as f:
            json.dump(milu_res["records"], f, indent=2, ensure_ascii=False)
        save_metrics_checkpoint(metrics, metrics_path)

    # 2. Flores-IN Translation (en->mr and mr->en dev)
    if run_flores:
        print("\n--- [2/4] Running Flores-IN (en->mr and mr->en) ---")
        flores_en_mr = evaluate_flores_direction(
            model=model,
            tokenizer=tokenizer,
            direction="en_mr",
            split="dev",
            batch_size=batch_size,
            prompt_format=prompt_format,
            device=device,
        )
        flores_mr_en = evaluate_flores_direction(
            model=model,
            tokenizer=tokenizer,
            direction="mr_en",
            split="dev",
            batch_size=batch_size,
            prompt_format=prompt_format,
            device=device,
        )

        all_generations.extend(flores_en_mr["records"])

        metrics["flores"] = {
            "en_mr_chrfpp": flores_en_mr["summary"]["corpus_chrf_pp"],
            "mr_en_chrfpp": flores_mr_en["summary"]["corpus_chrf_pp"],
            "ci95": {
                "en_mr": flores_en_mr["summary"]["ci_95"],
                "mr_en": flores_mr_en["summary"]["ci_95"],
            },
            "mean_sentence_chrfpp": {
                "en_mr": flores_en_mr["summary"]["mean_sentence_chrf_pp"],
                "mr_en": flores_mr_en["summary"]["mean_sentence_chrf_pp"],
            },
        }

        with open(run_dir / "flores_en_mr_records.json", "w", encoding="utf-8") as f:
            json.dump(flores_en_mr["records"], f, indent=2, ensure_ascii=False)
        with open(run_dir / "flores_mr_en_records.json", "w", encoding="utf-8") as f:
            json.dump(flores_mr_en["records"], f, indent=2, ensure_ascii=False)
        save_metrics_checkpoint(metrics, metrics_path)

    # 3. Held-out Bits Per Character (BPC)
    if run_bpc:
        print("\n--- [3/4] Running Held-out BPC ---")
        ref_texts = load_marathi_reference_texts(limit=bpc_limit)
        bpc_res = compute_bpc(model, tokenizer, ref_texts, device=device)
        metrics["heldout_bpc"] = {
            "flores_mr_dev": bpc_res["bits_per_character"],
            "flores_mr_dev_bpb": bpc_res["bits_per_byte"],
            "perplexity": bpc_res["token_perplexity"],
            "n_sentences": bpc_res["n_sentences"],
        }
        save_metrics_checkpoint(metrics, metrics_path)

    # 4. Automatic Generation Metrics
    if all_generations:
        auto_stats = evaluate_generations(all_generations)
        metrics["auto"] = {
            "script_purity": auto_stats.get("mean_script_purity", 0.0),
            "repeat_4gram": auto_stats.get("repetition_rate", 0.0),
            "distinct_4grams_ratio": auto_stats.get("mean_distinct_4grams_ratio", 0.0),
            "eos_rate": auto_stats.get("eos_rate", 0.0),
            "mean_len_tokens": auto_stats.get("mean_new_tokens", 0.0),
        }
        save_metrics_checkpoint(metrics, metrics_path)

    # Save intermediate generations
    with open(run_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for gen in all_generations:
            f.write(json.dumps(gen, ensure_ascii=False) + "\n")

    # Clean up PyTorch memory completely before running external subprocess lm-eval
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 5. IndicIFEval (Harness subprocess)
    if run_ifeval:
        print("\n--- [4/4] Running IndicIFEval ---")
        try:
            ifeval_out = run_dir / "ifeval"
            is_preq = metrics.get("quant") == "prequantized-bnb-4bit"
            ifeval_res = run_indicifeval(
                model_id=model_id,
                adapter_path=adapter_path,
                output_dir=str(ifeval_out),
                prequantized=is_preq,
                limit=ifeval_limit,
                device=device,
            )
            metrics["instruction_following"] = ifeval_res
        except Exception as e:
            print(f"Warning: IndicIFEval execution failed: {e}")
            metrics["instruction_following"] = {"error": str(e)}
        save_metrics_checkpoint(metrics, metrics_path)

    print(f"\n==========================================")
    print(f"Finished evaluation! Saved to: {metrics_path}")
    print(f"==========================================\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="results/runs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-format", default="chat", choices=["chat", "alpaca_v0", "raw"])
    parser.add_argument("--prequantized", action="store_true", default=None)
    parser.add_argument("--no-prequantized", action="store_false", dest="prequantized")
    parser.add_argument("--force", action="store_true", help="Overwrite existing run_id directory")
    parser.add_argument("--skip-milu", action="store_true")
    parser.add_argument("--skip-flores", action="store_true")
    parser.add_argument("--skip-bpc", action="store_true")
    parser.add_argument("--skip-ifeval", action="store_true")
    parser.add_argument("--ifeval-limit", type=int, default=None)
    parser.add_argument("--bpc-limit", type=int, default=None)
    args = parser.parse_args()

    run_full_evaluation(
        model_id=args.model,
        adapter_path=args.adapter,
        run_id=args.run_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        prompt_format=args.prompt_format,
        prequantized=args.prequantized,
        force=args.force,
        run_milu=not args.skip_milu,
        run_flores=not args.skip_flores,
        run_bpc=not args.skip_bpc,
        run_ifeval=not args.skip_ifeval,
        ifeval_limit=args.ifeval_limit,
        bpc_limit=args.bpc_limit,
    )


if __name__ == "__main__":
    main()
