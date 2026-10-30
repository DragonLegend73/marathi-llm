"""Wrapper for running IndicIFEval (Trans and Ground) instruction-following evaluation via lm-eval."""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def run_indicifeval(
    model_id: str,
    adapter_path: Optional[str] = None,
    output_dir: str = "results/ifeval",
    tasks: str = "indicifeval_ground_mr,indicifeval_trans_mr",
    include_path: str = "IndicIFEval/lm-evaluation-harness/custom_configs",
    prequantized: bool = True,
    limit: Optional[int] = None,
    device: str = "cuda:0",
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Executes IndicIFEval evaluation using lm-evaluation-harness.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build model arguments
    model_args_list = [f"pretrained={model_id}", "trust_remote_code=True"]
    if not prequantized:
        model_args_list.append("load_in_4bit=True")
    if adapter_path:
        model_args_list.append(f"peft={adapter_path}")

    model_args = ",".join(model_args_list)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--include_path", include_path,
        "--batch_size", str(batch_size),
        "--device", device,
        "--apply_chat_template",
        "--log_samples",
        "--output_path", str(out_path),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"Running IndicIFEval command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"lm_eval failed with returncode {result.returncode}:\n{result.stderr}")

    # Look for the generated output JSON in out_path
    json_files = list(out_path.glob("**/results*.json"))
    if not json_files:
        json_files = list(out_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No results JSON found in {out_path} after lm_eval execution.")

    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    with open(latest_file, "r", encoding="utf-8") as f:
        raw_res = json.load(f)

    results_by_task = raw_res.get("results", {})
    parsed_metrics = {
        "max_gen_toks": 1280,
    }

    for task_name, task_metrics in results_by_task.items():
        # Short key for Appendix B schema compatibility
        short_key = "trans_mr" if "trans" in task_name else ("ground_mr" if "ground" in task_name else task_name)

        def get_metric(metric_name):
            val = task_metrics.get(f"{metric_name},none")
            if val is None:
                val = task_metrics.get(metric_name)
            if val is None:
                raise KeyError(f"Metric '{metric_name}' not found in task '{task_name}' results: {list(task_metrics.keys())}")
            return float(val * 100.0)

        task_data = {
            "prompt_level_strict_acc": get_metric("prompt_level_strict_acc"),
            "prompt_level_loose_acc": get_metric("prompt_level_loose_acc"),
            "inst_level_strict_acc": get_metric("inst_level_strict_acc"),
            "inst_level_loose_acc": get_metric("inst_level_loose_acc"),
        }
        parsed_metrics[short_key] = task_data
        parsed_metrics[task_name] = task_data

    summary_file = out_path / "ifeval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(parsed_metrics, f, indent=2)

    return parsed_metrics


def main():
    parser = argparse.ArgumentParser(description="IndicIFEval instruction following evaluation")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tasks", default="indicifeval_ground_mr,indicifeval_trans_mr")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prequantized", action="store_true", default=True)
    parser.add_argument("--output-dir", default="results/ifeval")
    args = parser.parse_args()

    metrics = run_indicifeval(
        model_id=args.model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        tasks=args.tasks,
        prequantized=args.prequantized,
        limit=args.limit,
    )

    print("\n--- IndicIFEval Results ---")
    for key in ["ground_mr", "trans_mr"]:
        if key in metrics:
            m = metrics[key]
            print(f"\nTask: {key}")
            print(f"  Prompt Strict Acc: {m['prompt_level_strict_acc']:.2f}%")
            print(f"  Prompt Loose Acc:  {m['prompt_level_loose_acc']:.2f}%")
            print(f"  Inst Strict Acc:   {m['inst_level_strict_acc']:.2f}%")
            print(f"  Inst Loose Acc:    {m['inst_level_loose_acc']:.2f}%")


if __name__ == "__main__":
    main()
