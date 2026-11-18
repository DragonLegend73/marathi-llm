"""Trains and evaluates the max_seq_len=768 Stage A1 / Stage B1 control (A_qwen25_1p5b_768)."""

import os
import subprocess
import sys
import time


def run_step(desc: str, cmd: list[str]) -> None:
    print(f"\n==========================================", flush=True)
    print(f"[Control 768] {desc}", flush=True)
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
        raise RuntimeError(f"Step failed ({desc}) rc={proc.returncode} after {elapsed:.1f}s")
    print(f"\n<<< [Control 768 DONE] {desc} in {elapsed/60.0:.1f} min\n", flush=True)


def main():
    py = sys.executable

    run_step(
        "1/2 Train A_qwen25_1p5b_768 (12k S0, 1500 steps, max_seq_len=768)",
        [
            py, "-m", "marathi_llm.train",
            "--config", "configs/models/qwen25_1p5b.yaml",
            "--data", "data/processed/s0_alpaca_12k.jsonl",
            "--run-id", "A_qwen25_1p5b_768",
        ],
    )

    run_step(
        "2/2 Evaluate A_qwen25_1p5b_768 (MILU 810 + Full Flores 997 + BPC 997)",
        [
            py, "-m", "marathi_llm.eval.runner",
            "--run-id", "A_qwen25_1p5b_768",
            "--prompt-format", "chat",
            "--skip-ifeval",
            "--batch-size", "4",
            "--adapter", "results/runs/A_qwen25_1p5b_768/adapter",
        ],
    )


if __name__ == "__main__":
    main()
