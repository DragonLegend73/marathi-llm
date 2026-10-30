"""Wrapper for running IndicIFEval (Trans and Ground) instruction-following evaluation via lm-eval."""

import os
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
    model_args = f"pretrained={model_id},trust_remote_code=True"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    cmd = [
        ".venv/bin/lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--include_path", include_path,
        "--batch_size", str(batch_size),
        "--device", device,
        "--apply_chat_template",
        "--output_path", str(out_path),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"lm_eval failed with returncode {result.returncode}")

    # Look for the generated output JSON in out_path
    json_files = list(out_path.glob("**/results*.json"))
    if not json_files:
        # Check files directly in out_path
        json_files = list(out_path.glob("*.json"))

    parsed_metrics = {}
    if json_files:
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, "r", encoding="utf-8") as f:
            raw_res = json.load(f)

        results_by_task = raw_res.get("results", {})
        for task_name, task_metrics in results_by_task.items():
            parsed_metrics[task_name] = {
                "prompt_level_strict_acc": float(task_metrics.get("prompt_level_strict_acc,none", 0.0) * 100.0),
                "prompt_level_loose_acc": float(task_metrics.get("prompt_level_loose_acc,none", 0.0) * 100.0),
                "inst_level_strict_acc": float(task_metrics.get("inst_level_strict_acc,none", 0.0) * 100.0),
                "inst_level_loose_acc": float(task_metrics.get("inst_level_loose_acc,none", 0.0) * 100.0),
            }

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
    parser.add_argument("--output-dir", default="results/ifeval")
    args = parser.parse_args()

    metrics = run_indicifeval(
        model_id=args.model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        tasks=args.tasks,
        limit=args.limit,
    )

    print("\n--- IndicIFEval Results ---")
    for tname, m in metrics.items():
        print(f"\nTask: {tname}")
        print(f"  Prompt Strict Acc: {m['prompt_level_strict_acc']:.2f}%")
        print(f"  Prompt Loose Acc:  {m['prompt_level_loose_acc']:.2f}%")
        print(f"  Inst Strict Acc:   {m['inst_level_strict_acc']:.2f}%")
        print(f"  Inst Loose Acc:    {m['inst_level_loose_acc']:.2f}%")


if __name__ == "__main__":
    main()
