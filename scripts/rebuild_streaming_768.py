"""Sequentially rebuilds S3, S1, and S2 at max_seq_len=768 in isolated subprocesses."""

import os
import subprocess
import sys


CONFIGS = [
    "configs/data/s3_aya.yaml",
    "configs/data/s1_indicalign.yaml",
    "configs/data/s2_sandlogic.yaml",
]


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    for cfg in CONFIGS:
        print(f"\n>>> Rebuilding {cfg} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "marathi_llm.data.build", "--config", cfg],
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed rebuilding {cfg} (rc={proc.returncode})")


if __name__ == "__main__":
    main()
