"""Runs Stage A Model Architecture Screening on S0 Alpaca-mr at max_seq_len=768.

Each candidate model is trained under identical hyperparameters:
  - Dataset: data/processed/s0_alpaca_12k.jsonl (12,000 examples, cap=768)
  - Training: 1 epoch (1,500 steps, effective batch 8), lr=2e-4, LoRA r=16/alpha=32
  - Response-only loss, seed=0
Evaluated on:
  - MILU Marathi dev (n=810)
  - Flores-IN dev (en->mr and mr->en) with --flores-limit 200 (screening policy)
  - Held-out Marathi BPC (n=997)
  - Automatic fluency and script purity metrics

Finalists use the full 997 Flores set; screening uses --flores-limit 200.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


STAGE_A_RUNS = [
    {
        "id": "A_qwen25_1p5b_768",
        "name": "A1: Qwen2.5-1.5B-Instruct (Control)",
        "config": "configs/models/qwen25_1p5b.yaml",
        "prompt_format": "chat",
    },
    {
        "id": "A2_qwen3_1p7b_768",
        "name": "A2: Qwen3-1.7B",
        "config": "configs/models/qwen3_1p7b.yaml",
        "prompt_format": "chat",
    },
    {
        "id": "A3_gemma3_1b_768",
        "name": "A3: Gemma-3-1B-it",
        "config": "configs/models/gemma3_1b.yaml",
        "prompt_format": "chat",
    },
    {
        "id": "A4_llama32_1b_768",
        "name": "A4: Llama-3.2-1B-Instruct",
        "config": "configs/models/llama32_1b.yaml",
        "prompt_format": "chat",
    },
    {
        "id": "A5_sarvam1_768",
        "name": "A5: Sarvam-1 (2B)",
        "config": "configs/models/sarvam1.yaml",
        "prompt_format": "plain",
    },
]

DATA_PATH = "data/processed/s0_alpaca_12k.jsonl"


def run_cmd(desc: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"[Stage A] {desc}", flush=True)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    print(f"{'='*70}\n", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) rc={proc.returncode} after {elapsed:.1f}s")
    print(f"\n<<< [DONE] {desc} in {elapsed/60.0:.1f} min\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run Stage A candidate model screening")
    parser.add_argument(
        "--arm",
        choices=["all", "A1", "A2", "A3", "A4", "A5"],
        default="all",
        help="Which arm to run (default: all)",
    )
    parser.add_argument(
        "--flores-limit",
        type=int,
        default=200,
        help="Flores sentence limit for screening (default: 200)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run evaluation only",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining even if adapter already exists",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Force re-evaluation even if metrics.json already exists",
    )
    args = parser.parse_args()

    py = sys.executable

    runs = STAGE_A_RUNS
    if args.arm != "all":
        runs = [r for r in STAGE_A_RUNS if r["id"].startswith(args.arm) or (args.arm == "A1" and r["id"].startswith("A_qwen25"))]

    print(f"Executing Stage A screening for: {[r['id'] for r in runs]}")
    t_total = time.time()

    for r in runs:
        run_id = r["id"]
        config_path = r["config"]
        prompt_fmt = r.get("prompt_format", "chat")
        adapter_path = f"results/runs/{run_id}/adapter"
        metrics_path = f"results/runs/{run_id}/metrics.json"

        # 1. Train
        if not args.eval_only:
            if (Path(adapter_path) / "adapter_model.safetensors").exists() and not args.force_train:
                print(f"\n[Stage A] Adapter already exists at {adapter_path}; skipping training (pass --force-train to retrain).", flush=True)
            else:
                train_cmd = [
                    py, "-m", "marathi_llm.train",
                    "--config", config_path,
                    "--data", DATA_PATH,
                    "--run-id", run_id,
                ]
                run_cmd(f"Train {r['name']}", train_cmd)

        # 2. Evaluate with --flores-limit
        if Path(metrics_path).exists() and not args.force_eval:
            print(f"\n[Stage A] Metrics already exist at {metrics_path}; skipping evaluation (pass --force-eval to re-evaluate).", flush=True)
        else:
            eval_cmd = [
                py, "-m", "marathi_llm.eval.runner",
                "--run-id", run_id,
                "--prompt-format", prompt_fmt,
                "--skip-ifeval",
                "--batch-size", "4",
                "--flores-limit", str(args.flores_limit),
                "--adapter", adapter_path,
            ]
            run_cmd(f"Evaluate {r['name']} (flores_limit={args.flores_limit})", eval_cmd)

    total_min = (time.time() - t_total) / 60.0
    print(f"\n{'='*70}")
    print(f"[Stage A Screen Complete] All arms finished in {total_min:.1f} min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
