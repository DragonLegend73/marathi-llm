"""End-to-end offline runner for E3 (paired S1-6k cap=512 vs cap=768 experiment)."""

import os
import subprocess
import sys
import time


def run_step(desc: str, cmd: list[str]) -> None:
    print(f"\n==========================================", flush=True)
    print(f"[E3 Step] {desc}", flush=True)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    print(f"==========================================\n", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({desc}) with return code {proc.returncode} after {elapsed:.1f}s")
    print(f"\n<<< [E3 Step DONE] {desc} in {elapsed/60.0:.1f} min\n", flush=True)


def main():
    py = sys.executable

    # 1. Train cap=512 arm (6k rows = 750 steps at b2xa4, seed 0)
    run_step(
        "1/5 Train E3_s1_6k_cap512 (750 steps)",
        [
            py, "-m", "marathi_llm.train",
            "--config", "configs/models/qwen25_1p5b.yaml",
            "--data", "data/processed/e3_s1_cap512_6k.jsonl",
            "--eval-data", "data/processed/e3_s1_cap512_dev500.jsonl",
            "--max-seq-len", "512",
            "--seed", "0",
            "--run-id", "E3_s1_6k_cap512",
        ],
    )

    # 2. Evaluate cap=512 arm (MILU 810 + Flores-200 + BPC 997)
    run_step(
        "2/5 Evaluate E3_s1_6k_cap512",
        [
            py, "-m", "marathi_llm.eval.runner",
            "--run-id", "E3_s1_6k_cap512",
            "--prompt-format", "chat",
            "--skip-ifeval",
            "--flores-limit", "200",
            "--batch-size", "4",
            "--adapter", "results/runs/E3_s1_6k_cap512/adapter",
        ],
    )

    # 3. Train cap=768 arm (6k rows = 750 steps at b2xa4, seed 0)
    run_step(
        "3/5 Train E3_s1_6k_cap768 (750 steps)",
        [
            py, "-m", "marathi_llm.train",
            "--config", "configs/models/qwen25_1p5b.yaml",
            "--data", "data/processed/e3_s1_cap768_6k.jsonl",
            "--eval-data", "data/processed/e3_s1_cap768_dev500.jsonl",
            "--max-seq-len", "768",
            "--seed", "0",
            "--run-id", "E3_s1_6k_cap768",
        ],
    )

    # 4. Evaluate cap=768 arm (MILU 810 + Flores-200 + BPC 997)
    run_step(
        "4/5 Evaluate E3_s1_6k_cap768",
        [
            py, "-m", "marathi_llm.eval.runner",
            "--run-id", "E3_s1_6k_cap768",
            "--prompt-format", "chat",
            "--skip-ifeval",
            "--flores-limit", "200",
            "--batch-size", "4",
            "--adapter", "results/runs/E3_s1_6k_cap768/adapter",
        ],
    )

    # 5. Paired bootstrap comparison & pre-registered E3 verdict
    run_step(
        "5/5 Paired Bootstrap Comparison (E3_s1_6k_cap768 vs E3_s1_6k_cap512)",
        [
            py, "scripts/compare_e3.py",
            "--run-a", "E3_s1_6k_cap512",
            "--run-b", "E3_s1_6k_cap768",
        ],
    )


if __name__ == "__main__":
    main()
