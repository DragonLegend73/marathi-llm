"""Runs Stage B Data Screening (B2=S1 IndicAlign, B3=S2 SandLogic, B4=S3 Aya) on Qwen2.5-1.5B at max_seq_len=768.

Each run trains 1 epoch on the cleaned, deduplicated, decontaminated source at cap=768,
then evaluates on:
  - MILU Marathi dev (n=810)
  - Flores-IN dev (en->mr and mr->en) with --flores-limit 200 (screening policy)
  - Held-out Marathi BPC (n=997)
  - Automatic fluency and script purity metrics

Finalists and the control use the full 997 Flores set; screening uses --flores-limit 200.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


STAGE_B_RUNS = [
    {
        "id": "B2_s1_indicalign_12k",
        "name": "B2: S1 IndicAlign-Marathi (12,000 examples)",
        "data": "data/processed/s1_indicalign_12k.jsonl",
    },
    {
        "id": "B3_s2_sandlogic_12k",
        "name": "B3: S2 SandLogic-Marathi (12,000 examples)",
        "data": "data/processed/s2_sandlogic_12k.jsonl",
    },
    {
        "id": "B4_s3_aya_1p2k",
        "name": "B4: S3 Aya-Marathi (1,203 examples, all available)",
        "data": "data/processed/s3_aya_12k.jsonl",
    },
]


def run_cmd(desc: str, cmd: list[str]) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"[Stage B] {desc}", flush=True)
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
    parser = argparse.ArgumentParser(description="Run Stage B data screening")
    parser.add_argument(
        "--arm",
        choices=["all", "B2", "B3", "B4"],
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
    args = parser.parse_args()

    py = sys.executable

    runs = STAGE_B_RUNS
    if args.arm != "all":
        runs = [r for r in STAGE_B_RUNS if r["id"].startswith(args.arm)]

    print(f"Executing Stage B screening for: {[r['id'] for r in runs]}")
    t_total = time.time()

    for r in runs:
        run_id = r["id"]
        data_path = r["data"]
        adapter_path = f"results/runs/{run_id}/adapter"
        metrics_path = f"results/runs/{run_id}/metrics.json"

        # 1. Train
        if not args.eval_only:
            train_cmd = [
                py, "-m", "marathi_llm.train",
                "--config", "configs/models/qwen25_1p5b.yaml",
                "--data", data_path,
                "--run-id", run_id,
            ]
            run_cmd(f"Train {r['name']}", train_cmd)

        # 2. Evaluate with --flores-limit
        eval_cmd = [
            py, "-m", "marathi_llm.eval.runner",
            "--run-id", run_id,
            "--prompt-format", "chat",
            "--skip-ifeval",
            "--batch-size", "4",
            "--flores-limit", str(args.flores-limit),
            "--adapter", adapter_path,
        ]
        run_cmd(f"Evaluate {r['name']} (flores_limit={args.flores_limit})", eval_cmd)

    total_min = (time.time() - t_total) / 60.0
    print(f"\n{'='*70}")
    print(f"[Stage B Screen Complete] All arms finished in {total_min:.1f} min")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
