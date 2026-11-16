"""Aggregation and analysis module producing results/summary.md and visualization figures."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np

from marathi_llm.eval.stats import paired_bootstrap_ci, paired_bootstrap_corpus_chrf, mcnemar_test


def load_all_runs(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Loads all metrics.json files from runs_dir subdirectories."""
    runs = {}
    if not runs_dir.exists():
        return runs

    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir():
            metrics_file = run_path / "metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        runs[run_path.name] = json.load(f)
                except Exception as e:
                    print(f"Warning: could not read {metrics_file}: {e}")
    return runs


def format_ci(val: Optional[float], ci: Optional[List[float]], fmt: str = ".2f") -> str:
    """Formats a value and its 95% confidence interval."""
    if val is None:
        return "—"
    val_str = f"{val:{fmt}}"
    if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        return f"{val_str} [{ci[0]:{fmt}}, {ci[1]:{fmt}}]"
    return val_str


def build_overview_table(runs: Dict[str, Dict[str, Any]]) -> str:
    """Constructs markdown summary table comparing headline metrics across runs."""
    headers = [
        "Run ID",
        "Model",
        "Adapter",
        "Prompt Format",
        "MILU Val Acc (%)",
        "Flores en→mr (chrF++)",
        "Flores mr→en (chrF++)",
        "Held-out BPC",
        "Script Purity (%)",
        "EOS Rate (%)",
        "IFEval Ground (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for run_id, m in runs.items():
        model_name = m.get("model", "—").split("/")[-1]
        adapter_name = m.get("adapter", "none")
        if adapter_name and adapter_name != "none":
            adapter_name = adapter_name.split("/")[-1]
        prompt_fmt = m.get("prompt_format", "chat")

        milu_val = m.get("milu_mr", {}).get("acc")
        milu_ci = m.get("milu_mr", {}).get("ci95")
        milu_str = format_ci(milu_val, milu_ci, fmt=".2f")

        flores = m.get("flores", {})
        en_mr_val = flores.get("en_mr_chrfpp")
        en_mr_ci = flores.get("ci95", {}).get("en_mr")
        en_mr_str = format_ci(en_mr_val, en_mr_ci, fmt=".2f")

        mr_en_val = flores.get("mr_en_chrfpp")
        mr_en_ci = flores.get("ci95", {}).get("mr_en")
        mr_en_str = format_ci(mr_en_val, mr_en_ci, fmt=".2f")

        bpc_val = m.get("heldout_bpc", {}).get("flores_mr_dev")
        bpc_ci = m.get("heldout_bpc", {}).get("ci95")
        bpc_str = format_ci(bpc_val, bpc_ci, fmt=".4f")

        auto = m.get("auto", {})
        purity_str = f"{auto.get('script_purity', 0.0):.2f}%" if "script_purity" in auto else "—"
        eos_str = f"{auto.get('eos_rate', 0.0):.2f}%" if "eos_rate" in auto else "—"

        ifeval = m.get("instruction_following", {})
        ifeval_ground = ifeval.get("ground_mr", {}).get("prompt_level_strict_acc")
        ifeval_str = f"{ifeval_ground:.2f}%" if ifeval_ground is not None else "—"

        row = [
            f"`{run_id}`",
            model_name,
            adapter_name,
            f"`{prompt_fmt}`",
            milu_str,
            en_mr_str,
            mr_en_str,
            bpc_str,
            purity_str,
            eos_str,
            ifeval_str,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def run_paired_comparisons(
    runs_dir: Path,
    baseline_id: str,
    target_id: str,
    n_bootstraps: int = 1000,
) -> Dict[str, Any]:
    """Runs paired statistical significance tests between baseline and target runs."""
    base_dir = runs_dir / baseline_id
    target_dir = runs_dir / target_id

    results = {
        "baseline_id": baseline_id,
        "target_id": target_id,
        "metrics": {},
    }

    # 1. MILU McNemar & Paired Difference
    milu_base_f = base_dir / "milu_validation_records.json"
    milu_target_f = target_dir / "milu_validation_records.json"
    if milu_base_f.exists() and milu_target_f.exists():
        with open(milu_base_f, "r", encoding="utf-8") as f:
            base_records = json.load(f)
        with open(milu_target_f, "r", encoding="utf-8") as f:
            target_records = json.load(f)

        if len(base_records) == len(target_records):
            correct_a = [bool(r.get("correct")) for r in base_records]
            correct_b = [bool(r.get("correct")) for r in target_records]
            mcnemar_res = mcnemar_test(correct_a, correct_b)

            acc_a = float(np.mean(correct_a)) * 100.0
            acc_b = float(np.mean(correct_b)) * 100.0
            paired_diff = paired_bootstrap_ci(
                [float(x) * 100.0 for x in correct_a],
                [float(x) * 100.0 for x in correct_b],
                n_bootstraps=n_bootstraps,
            )

            results["metrics"]["milu"] = {
                "acc_a": acc_a,
                "acc_b": acc_b,
                "delta": acc_b - acc_a,
                "ci95_diff": [paired_diff["ci_low"], paired_diff["ci_high"]],
                "mcnemar_p_value": mcnemar_res["p_value"],
                "is_significant": paired_diff["is_significant"],
            }

    # 2. Flores en->mr Paired chrF++ Difference
    f_en_mr_base = base_dir / "flores_en_mr_records.json"
    f_en_mr_target = target_dir / "flores_en_mr_records.json"
    if f_en_mr_base.exists() and f_en_mr_target.exists():
        with open(f_en_mr_base, "r", encoding="utf-8") as f:
            rec_a = json.load(f)
        with open(f_en_mr_target, "r", encoding="utf-8") as f:
            rec_b = json.load(f)

        if len(rec_a) == len(rec_b):
            scores_a = [r.get("chrf_pp", 0.0) for r in rec_a]
            scores_b = [r.get("chrf_pp", 0.0) for r in rec_b]
            paired_sent = paired_bootstrap_ci(scores_a, scores_b, n_bootstraps=n_bootstraps)

            hyps_a = [r.get("output", r.get("generation", "")) for r in rec_a]
            hyps_b = [r.get("output", r.get("generation", "")) for r in rec_b]
            refs = [r.get("target", "") for r in rec_a]
            # Fast corpus bootstrap with 50 resamples
            paired_chrf = paired_bootstrap_corpus_chrf(hyps_a, hyps_b, refs, n_bootstraps=min(50, n_bootstraps))

            results["metrics"]["flores_en_mr"] = {
                "score_a": paired_chrf["score_a"],
                "score_b": paired_chrf["score_b"],
                "delta": paired_chrf["diff_corpus_chrf"],
                "ci95_diff": [paired_chrf["ci_low"], paired_chrf["ci_high"]],
                "p_value": paired_sent["p_value"],
                "is_significant": paired_sent["is_significant"],
            }

    # 3. Flores mr->en Paired chrF++ Difference
    f_mr_en_base = base_dir / "flores_mr_en_records.json"
    f_mr_en_target = target_dir / "flores_mr_en_records.json"
    if f_mr_en_base.exists() and f_mr_en_target.exists():
        with open(f_mr_en_base, "r", encoding="utf-8") as f:
            rec_a = json.load(f)
        with open(f_mr_en_target, "r", encoding="utf-8") as f:
            rec_b = json.load(f)

        if len(rec_a) == len(rec_b):
            scores_a = [r.get("chrf_pp", 0.0) for r in rec_a]
            scores_b = [r.get("chrf_pp", 0.0) for r in rec_b]
            paired_sent = paired_bootstrap_ci(scores_a, scores_b, n_bootstraps=n_bootstraps)

            hyps_a = [r.get("output", r.get("generation", "")) for r in rec_a]
            hyps_b = [r.get("output", r.get("generation", "")) for r in rec_b]
            refs = [r.get("target", "") for r in rec_a]
            # Fast corpus bootstrap with 50 resamples
            paired_chrf = paired_bootstrap_corpus_chrf(hyps_a, hyps_b, refs, n_bootstraps=min(50, n_bootstraps))

            results["metrics"]["flores_mr_en"] = {
                "score_a": paired_chrf["score_a"],
                "score_b": paired_chrf["score_b"],
                "delta": paired_chrf["diff_corpus_chrf"],
                "ci95_diff": [paired_chrf["ci_low"], paired_chrf["ci_high"]],
                "p_value": paired_sent["p_value"],
                "is_significant": paired_sent["is_significant"],
            }

    # 4. Held-out BPC Paired Difference
    bpc_base_f = base_dir / "bpc_records.json"
    bpc_target_f = target_dir / "bpc_records.json"
    if bpc_base_f.exists() and bpc_target_f.exists():
        with open(bpc_base_f, "r", encoding="utf-8") as f:
            bpc_rec_a = json.load(f)
        with open(bpc_target_f, "r", encoding="utf-8") as f:
            bpc_rec_b = json.load(f)

        if len(bpc_rec_a) == len(bpc_rec_b):
            scores_a = [r.get("bpc", 0.0) for r in bpc_rec_a]
            scores_b = [r.get("bpc", 0.0) for r in bpc_rec_b]
            paired_bpc = paired_bootstrap_ci(scores_a, scores_b, n_bootstraps=n_bootstraps)
            results["metrics"]["bpc"] = {
                "bpc_a": float(np.mean(scores_a)),
                "bpc_b": float(np.mean(scores_b)),
                "delta": paired_bpc["diff_mean"],
                "ci95_diff": [paired_bpc["ci_low"], paired_bpc["ci_high"]],
                "p_value": paired_bpc["p_value"],
                "is_significant": paired_bpc["is_significant"],
            }

    return results


def format_paired_table(paired: Dict[str, Any]) -> str:
    """Formats paired comparison results into a markdown table."""
    metrics = paired.get("metrics", {})
    if not metrics:
        return ""

    headers = [
        "Benchmark / Metric",
        f"Baseline ({paired['baseline_id']})",
        f"Target ({paired['target_id']})",
        "Delta (Δ)",
        "95% CI of Δ",
        "Significant? (p < 0.05)",
    ]
    lines = [
        f"### Paired Comparison: `{paired['target_id']}` vs `{paired['baseline_id']}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    if "milu" in metrics:
        m = metrics["milu"]
        sig = "✅ Yes" if m["is_significant"] else "❌ No"
        lines.append(
            f"| **MILU Validation Acc** | {m['acc_a']:.2f}% | {m['acc_b']:.2f}% | **{m['delta']:+.2f} pp** | [{m['ci95_diff'][0]:+.2f}, {m['ci95_diff'][1]:+.2f}] | {sig} (McNemar p={m['mcnemar_p_value']:.4f}) |"
        )

    if "flores_en_mr" in metrics:
        m = metrics["flores_en_mr"]
        sig = "✅ Yes" if m["is_significant"] else "❌ No"
        lines.append(
            f"| **Flores en→mr (chrF++)** | {m['score_a']:.2f} | {m['score_b']:.2f} | **{m['delta']:+.2f}** | [{m['ci95_diff'][0]:+.2f}, {m['ci95_diff'][1]:+.2f}] | {sig} (p={m['p_value']:.4f}) |"
        )

    if "flores_mr_en" in metrics:
        m = metrics["flores_mr_en"]
        sig = "✅ Yes" if m["is_significant"] else "❌ No"
        lines.append(
            f"| **Flores mr→en (chrF++)** | {m['score_a']:.2f} | {m['score_b']:.2f} | **{m['delta']:+.2f}** | [{m['ci95_diff'][0]:+.2f}, {m['ci95_diff'][1]:+.2f}] | {sig} (p={m['p_value']:.4f}) |"
        )

    if "bpc" in metrics:
        m = metrics["bpc"]
        sig = "✅ Yes" if m["is_significant"] else "❌ No"
        lines.append(
            f"| **Held-out Flores BPC** | {m['bpc_a']:.4f} | {m['bpc_b']:.4f} | **{m['delta']:+.4f}** | [{m['ci95_diff'][0]:+.4f}, {m['ci95_diff'][1]:+.4f}] | {sig} (p={m['p_value']:.4f}) |"
        )

    return "\n".join(lines)


def calculate_stage_a_ranking(runs: Dict[str, Dict[str, Any]]) -> str:
    """
    Applies the pre-registered Stage A selection rule (EVAL.md Section 4):
    1. Primary: Mean rank across MILU val acc, Flores en->mr, and Flores mr->en.
    2. Equivalence: Differences where 95% paired CI includes 0.0 count as tied.
    3. Tie-breaker: IndicIFEval Ground strict -> BPC (lower better) -> training speed.
    """
    eligible = {}
    for run_id, m in runs.items():
        milu = m.get("milu_mr", {}).get("acc")
        en_mr = m.get("flores", {}).get("en_mr_chrfpp")
        mr_en = m.get("flores", {}).get("mr_en_chrfpp")
        bpc = m.get("heldout_bpc", {}).get("flores_mr_dev")

        if milu is not None and en_mr is not None and mr_en is not None:
            eligible[run_id] = {
                "milu": milu,
                "en_mr": en_mr,
                "mr_en": mr_en,
                "bpc": bpc if bpc is not None else float("inf"),
                "ifeval": m.get("instruction_following", {}).get("ground_mr", {}).get("prompt_level_strict_acc", 0.0),
            }

    if not eligible:
        return ""

    run_ids = list(eligible.keys())
    # Rank MILU (descending)
    milu_order = sorted(run_ids, key=lambda r: eligible[r]["milu"], reverse=True)
    milu_ranks = {r: rank + 1 for rank, r in enumerate(milu_order)}

    # Rank en_mr (descending)
    en_mr_order = sorted(run_ids, key=lambda r: eligible[r]["en_mr"], reverse=True)
    en_mr_ranks = {r: rank + 1 for rank, r in enumerate(en_mr_order)}

    # Rank mr_en (descending)
    mr_en_order = sorted(run_ids, key=lambda r: eligible[r]["mr_en"], reverse=True)
    mr_en_ranks = {r: rank + 1 for rank, r in enumerate(mr_en_order)}

    rows = []
    for r in run_ids:
        r_milu = milu_ranks[r]
        r_en_mr = en_mr_ranks[r]
        r_mr_en = mr_en_ranks[r]
        mean_rank = (r_milu + r_en_mr + r_mr_en) / 3.0
        rows.append({
            "run_id": r,
            "mean_rank": mean_rank,
            "r_milu": r_milu,
            "r_en_mr": r_en_mr,
            "r_mr_en": r_mr_en,
            "bpc": eligible[r]["bpc"],
            "ifeval": eligible[r]["ifeval"],
        })

    # Sort by mean_rank (ascending), then tie-breaker (ifeval descending, bpc ascending)
    rows.sort(key=lambda x: (x["mean_rank"], -x["ifeval"], x["bpc"]))

    lines = [
        "### Pre-Registered Stage A Model Ranking (EVAL.md Protocol)",
        "",
        "| Rank | Run ID | Mean Rank | MILU Rank | en→mr Rank | mr→en Rank | IFEval Ground (%) | Held-out BPC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for overall_rank, r in enumerate(rows, 1):
        lines.append(
            f"| **#{overall_rank}** | `{r['run_id']}` | **{r['mean_rank']:.2f}** | #{r['r_milu']} | #{r['r_en_mr']} | #{r['r_mr_en']} | {r['ifeval']:.2f}% | {r['bpc']:.4f} |"
        )

    return "\n".join(lines)


def plot_summary_figures(runs: Dict[str, Dict[str, Any]], output_path: Path):
    """Plots comparative bar charts across runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_ids = list(runs.keys())
        if not run_ids:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Marathi LLM Benchmark Results Comparison", fontsize=14, fontweight="bold")

        labels = [r.replace("_qwen25_1p5b", "").replace("_", " ") for r in run_ids]
        x = np.arange(len(run_ids))
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"][:len(run_ids)]

        # 1. MILU Validation Accuracy
        milu_accs = [runs[r].get("milu_mr", {}).get("acc", 0.0) for r in run_ids]
        milu_errs = []
        for r in run_ids:
            ci = runs[r].get("milu_mr", {}).get("ci95")
            acc = runs[r].get("milu_mr", {}).get("acc", 0.0)
            if ci and len(ci) == 2:
                milu_errs.append([acc - ci[0], ci[1] - acc])
            else:
                milu_errs.append([0.0, 0.0])
        milu_errs = np.array(milu_errs).T

        ax = axes[0, 0]
        ax.bar(x, milu_accs, yerr=milu_errs, capsize=5, color=colors, alpha=0.85)
        ax.axhline(25.0, color="gray", linestyle="--", label="Chance (25%)")
        ax.set_title("MILU Validation Accuracy (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc="upper left")

        # 2. Flores-IN Translation chrF++
        en_mr_scores = [runs[r].get("flores", {}).get("en_mr_chrfpp", 0.0) for r in run_ids]
        mr_en_scores = [runs[r].get("flores", {}).get("mr_en_chrfpp", 0.0) for r in run_ids]
        width = 0.35

        ax = axes[0, 1]
        ax.bar(x - width/2, en_mr_scores, width, label="en→mr", color="#2b5c8f", alpha=0.85)
        ax.bar(x + width/2, mr_en_scores, width, label="mr→en", color="#d95f02", alpha=0.85)
        ax.set_title("Flores-IN Translation (Corpus chrF++)")
        ax.set_ylabel("chrF++")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # 3. Held-out BPC
        bpc_scores = [runs[r].get("heldout_bpc", {}).get("flores_mr_dev", 0.0) for r in run_ids]
        ax = axes[1, 0]
        ax.bar(x, bpc_scores, color=colors, alpha=0.85)
        ax.set_title("Held-out Flores Marathi BPC (lower is better)")
        ax.set_ylabel("Bits per Character")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        # 4. Auto Generation Metrics: Script Purity & EOS Rate
        purity_scores = [runs[r].get("auto", {}).get("script_purity", 0.0) for r in run_ids]
        eos_scores = [runs[r].get("auto", {}).get("eos_rate", 0.0) for r in run_ids]

        ax = axes[1, 1]
        ax.bar(x - width/2, purity_scores, width, label="Script Purity (%)", color="#1b9e77", alpha=0.85)
        ax.bar(x + width/2, eos_scores, width, label="EOS Rate (%)", color="#7570b3", alpha=0.85)
        ax.set_title("Automatic Quality Metrics (%)")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure to: {output_path}")
    except Exception as e:
        print(f"Warning: could not generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Marathi LLM Benchmark Results Aggregator")
    parser.add_argument("--runs-dir", default="results/runs", help="Path to runs directory")
    parser.add_argument("--output-summary", default="results/summary.md", help="Output summary markdown path")
    parser.add_argument("--figures-dir", default="results/figures", help="Directory for generated figures")
    parser.add_argument("--n-bootstraps", type=int, default=1000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs = load_all_runs(runs_dir)
    if not runs:
        print(f"No runs found in {runs_dir}")
        return

    print(f"Found {len(runs)} evaluated run(s): {list(runs.keys())}")

    summary_content = [
        "# Marathi LLM Study — Benchmark Evaluation Summary",
        "",
        f"Generated automatically from `{runs_dir}` across {len(runs)} run(s).",
        "",
        "## 1. Overview Comparison Table",
        "",
        build_overview_table(runs),
        "",
    ]

    # Paired comparisons if baseline and adapter exist
    if "base_qwen25_1p5b" in runs and "v0_adapter_qwen25_1p5b" in runs:
        print("Running paired comparisons between base and v0_adapter...")
        paired = run_paired_comparisons(
            runs_dir=runs_dir,
            baseline_id="base_qwen25_1p5b",
            target_id="v0_adapter_qwen25_1p5b",
            n_bootstraps=args.n_bootstraps,
        )
        summary_content.extend([
            "## 2. Statistical Significance & Paired Tests",
            "",
            format_paired_table(paired),
            "",
        ])

    # Stage A selection ranking
    ranking_md = calculate_stage_a_ranking(runs)
    if ranking_md:
        summary_content.extend([
            "## 3. Pre-Registered Model Ranking",
            "",
            ranking_md,
            "",
        ])

    summary_md = "\n".join(summary_content)
    output_path = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Wrote summary to: {output_path}")

    # Generate figure
    fig_path = Path(args.figures_dir) / "summary_metrics.png"
    plot_summary_figures(runs, fig_path)

    print("\n" + "=" * 60)
    print(summary_md)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
