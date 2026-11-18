"""E2 Worst-Case Sequence-Length Memory & Throughput Probe (plan_2.md Section 2.5).

Runs 100 worst-case steps per cap in {512, 768, 1024, 1536, 2048} in isolated
subprocesses (clean CUDA allocator per cap), automatically retrying with
batch=1 x grad_accum=8 if batch=2 x grad_accum=4 OOMs, and evaluates the
pre-registered E4 decision rule against E1 census stats.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


CAPS = [512, 768, 1024, 1536, 2048]
MODEL_CFG = "configs/models/qwen25_1p5b.yaml"
OUT_JSON = Path("results/length_census/e2_probes.json")
OUT_MD = Path("results/length_census/e2_summary.md")


def run_single_probe(cap: int, batch: int, grad_accum: int, run_id: str) -> Dict[str, Any]:
    """Executes one 100-step worst-case profile in a fresh subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "marathi_llm.train",
        "--config",
        MODEL_CFG,
        "--profile-only",
        "--max-seq-len",
        str(cap),
        "--profile-data",
        "worst_case",
        "--per-device-batch",
        str(batch),
        "--grad-accum",
        str(grad_accum),
        "--run-id",
        run_id,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"

    print(f"\n>>> [E2 Probe] cap={cap}, batch={batch}x{grad_accum} (run_id={run_id}) ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    summary_path = Path("results/runs") / run_id / "train_summary.json"
    if proc.returncode == 0 and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tr = data["train"]
        print(
            f"<<< [E2 Probe PASS] cap={cap} batch={batch}x{grad_accum}: "
            f"alloc={tr['peak_vram_gb']:.2f} GB, reserved={tr['peak_vram_reserved_gb']:.2f} GB, "
            f"steps/s={tr['steps_per_s']:.3f}, tok/s={tr['tokens_per_s']:.1f}, temp={tr['gpu_temp_c']}C",
            flush=True,
        )
        return {
            "status": "ok",
            "cap": cap,
            "run_id": run_id,
            "per_device_batch": batch,
            "grad_accum": grad_accum,
            "train": tr,
        }

    err_tail = (proc.stderr or proc.stdout or "")[-1500:]
    is_oom = "out of memory" in err_tail.lower() or "cudaerrormemoryallocation" in err_tail.lower()
    print(
        f"<<< [E2 Probe {'OOM' if is_oom else 'FAIL'}] cap={cap} batch={batch}x{grad_accum} "
        f"(rc={proc.returncode})",
        flush=True,
    )
    return {
        "status": "oom" if is_oom else "error",
        "cap": cap,
        "run_id": run_id,
        "per_device_batch": batch,
        "grad_accum": grad_accum,
        "returncode": proc.returncode,
        "error_tail": err_tail,
    }


def build_report(probes: List[Dict[str, Any]]) -> str:
    """Builds Markdown table for E2 results + projected 12k ETAs + E4 gate evaluation."""
    census_dir = Path("results/length_census")
    census: Dict[str, Any] = {}
    for key, fname in [
        ("s0", "s0_alpaca.json"),
        ("s1", "s1_indicalign.json"),
        ("s2", "s2_sandlogic.json"),
        ("s3", "s3_aya.json"),
    ]:
        p = census_dir / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                census[key] = json.load(f)["summary"]

    lines = ["# E2 Worst-Case VRAM & Throughput Probe (`Qwen2.5-1.5B-Instruct`)\n"]
    lines.append(
        "| Cap | Config | Status | Peak Alloc (GB) | Peak Reserved (GB) | Steps/s | Tokens/s | Temp (°C) | ETA S0 (min) | ETA S1 (min) | ETA S2 (min) |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in probes:
        cap = r["cap"]
        cfg_str = f"b{r['per_device_batch']}×a{r['grad_accum']}"
        if r["status"] != "ok":
            lines.append(
                f"| {cap} | {cfg_str} | **{r['status'].upper()}** | - | - | - | - | - | - | - | - |"
            )
            continue
        tr = r["train"]
        tps = tr["tokens_per_s"]
        etas = {}
        for sk in ("s0", "s1", "s2"):
            if sk in census and str(cap) in census[sk]["by_cap"] and tps > 0:
                mean_tok = census[sk]["by_cap"][str(cap)]["mean_tokens_kept"]
                etas[sk] = f"{(12000 * mean_tok / tps / 60.0):.1f}"
            else:
                etas[sk] = "-"
        temp_s = f"{tr['gpu_temp_c']:.1f}" if tr.get("gpu_temp_c") is not None else "-"
        lines.append(
            f"| {cap} | {cfg_str} | OK | {tr['peak_vram_gb']:.3f} | {tr['peak_vram_reserved_gb']:.3f} "
            f"| {tr['steps_per_s']:.3f} | {tps:.1f} | {temp_s} | {etas['s0']} | {etas['s1']} | {etas['s2']} |"
        )

    # E4 Gate Evaluation for candidates {512, 768, 1024}
    lines.append("\n## E4 Gate Evaluation (Candidates: 512, 768, 1024)\n")
    base_min_kept = min(
        census["s1"]["by_cap"]["512"]["kept_pct"],
        census["s2"]["by_cap"]["512"]["kept_pct"],
    )
    lines.append("| Candidate | Hardware (≤3.4 GB res) | Time (S0/S1/S2 ETA ≤150m) | Benefit (min S1/S2 Δ ≥ +15 pp) | Verdict |")
    lines.append("|---:|---|---|---|---|")

    for cand in [512, 768, 1024]:
        # Pick best passing run for this cap (prefer b2xa4, else b1xa8)
        ok_runs = [p for p in probes if p["cap"] == cand and p["status"] == "ok"]
        if not ok_runs:
            lines.append(f"| {cand} | FAIL (OOM) | - | - | FAIL |")
            continue
        tr = ok_runs[0]["train"]
        hw_pass = tr["peak_vram_reserved_gb"] <= 3.4
        tps = tr["tokens_per_s"]
        eta_s0 = 12000 * census["s0"]["by_cap"][str(cand)]["mean_tokens_kept"] / tps / 60.0
        eta_s1 = 12000 * census["s1"]["by_cap"][str(cand)]["mean_tokens_kept"] / tps / 60.0
        eta_s2 = 12000 * census["s2"]["by_cap"][str(cand)]["mean_tokens_kept"] / tps / 60.0
        time_pass = eta_s0 <= 150.0 and eta_s1 <= 150.0 and eta_s2 <= 150.0
        min_kept = min(
            census["s1"]["by_cap"][str(cand)]["kept_pct"],
            census["s2"]["by_cap"][str(cand)]["kept_pct"],
        )
        delta_pp = min_kept - base_min_kept
        benefit_pass = True if cand == 512 else (delta_pp >= 15.0)
        overall = "PASS" if (hw_pass and time_pass and benefit_pass) else "FAIL"
        lines.append(
            f"| {cand} | {'PASS' if hw_pass else 'FAIL'} ({tr['peak_vram_reserved_gb']:.2f} GB) "
            f"| {'PASS' if time_pass else 'FAIL'} ({eta_s0:.0f}m / {eta_s1:.0f}m / {eta_s2:.0f}m) "
            f"| {'BASELINE' if cand == 512 else ('PASS' if benefit_pass else 'FAIL')} ({min_kept:.1f}%, {delta_pp:+.1f} pp) "
            f"| **{overall}** |"
        )

    return "\n".join(lines) + "\n"


def main():
    probes: List[Dict[str, Any]] = []
    for cap in CAPS:
        res = run_single_probe(cap, batch=2, grad_accum=4, run_id=f"profile_len_{cap}")
        probes.append(res)
        if res["status"] == "oom":
            res_b1 = run_single_probe(cap, batch=1, grad_accum=8, run_id=f"profile_len_{cap}_b1")
            probes.append(res_b1)
        # Incremental save after every probe
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(probes, f, indent=2)

    md = build_report(probes)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
