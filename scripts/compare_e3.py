"""Computes paired bootstrap 95% CIs for E3 (S1-6k cap=768 vs cap=512)."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import sacrebleu
from scipy.stats import binomtest


def paired_milu_bootstrap(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap & exact McNemar test on MILU accuracy (B - A)."""
    assert len(recs_a) == len(recs_b)
    a = np.array([1.0 if r["correct"] else 0.0 for r in recs_a])
    b = np.array([1.0 if r["correct"] else 0.0 for r in recs_b])
    n = len(a)

    acc_a = float(a.mean() * 100.0)
    acc_b = float(b.mean() * 100.0)
    delta = acc_b - acc_a

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = (b[idx].mean(axis=1) - a[idx].mean(axis=1)) * 100.0
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    # Exact McNemar on discordant pairs
    n_01 = int(((a == 0) & (b == 1)).sum())
    n_10 = int(((a == 1) & (b == 0)).sum())
    p_mcnemar = float(binomtest(min(n_01, n_10), n_01 + n_10, 0.5).pvalue) if (n_01 + n_10) > 0 else 1.0

    return {
        "n": n,
        "a": round(acc_a, 2),
        "b": round(acc_b, 2),
        "delta": round(delta, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "mcnemar_b_wins": n_01,
        "mcnemar_a_wins": n_10,
        "mcnemar_p": round(p_mcnemar, 4),
    }


def paired_flores_bootstrap(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired corpus chrF++ bootstrap (B - A) on identical resampled sentence indices."""
    assert len(recs_a) == len(recs_b)
    n = len(recs_a)
    hyps_a = [r.get("output", r.get("generation", "")) for r in recs_a]
    hyps_b = [r.get("output", r.get("generation", "")) for r in recs_b]
    refs = [r["target"] for r in recs_a]

    score_a = float(sacrebleu.corpus_chrf(hyps_a, [refs], word_order=2).score)
    score_b = float(sacrebleu.corpus_chrf(hyps_b, [refs], word_order=2).score)
    delta = score_b - score_a

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sa = sacrebleu.corpus_chrf([hyps_a[j] for j in idx], [[refs[j] for j in idx]], word_order=2).score
        sb = sacrebleu.corpus_chrf([hyps_b[j] for j in idx], [[refs[j] for j in idx]], word_order=2).score
        diffs[i] = sb - sa

    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "n": n,
        "a": round(score_a, 2),
        "b": round(score_b, 2),
        "delta": round(delta, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


def paired_bpc_bootstrap(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired corpus BPC bootstrap (B - A; negative delta means B is better)."""
    assert len(recs_a) == len(recs_b)
    n = len(recs_a)
    nll_a = np.array([r["nll"] for r in recs_a], dtype=np.float64)
    nll_b = np.array([r["nll"] for r in recs_b], dtype=np.float64)
    chars = np.array([r["chars"] for r in recs_a], dtype=np.float64)
    ln2 = math.log(2.0)

    bpc_a = float(nll_a.sum() / (ln2 * chars.sum()))
    bpc_b = float(nll_b.sum() / (ln2 * chars.sum()))
    delta = bpc_b - bpc_a

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    c_sum = chars[idx].sum(axis=1)
    diffs = (nll_b[idx].sum(axis=1) - nll_a[idx].sum(axis=1)) / (ln2 * c_sum)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    return {
        "n": n,
        "a": round(bpc_a, 4),
        "b": round(bpc_b, 4),
        "delta": round(delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


def main():
    parser = argparse.ArgumentParser(description="Paired comparison for E3 (cap=768 vs cap=512)")
    parser.add_argument("--run-a", default="E3_s1_6k_cap512")
    parser.add_argument("--run-b", default="E3_s1_6k_cap768")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--out-json", default="results/length_census/e3_comparison.json")
    parser.add_argument("--out-md", default="results/length_census/e3_comparison.md")
    args = parser.parse_args()

    dir_a = Path(args.runs_dir) / args.run_a
    dir_b = Path(args.runs_dir) / args.run_b

    with open(dir_a / "metrics.json") as f:
        met_a = json.load(f)
    with open(dir_b / "metrics.json") as f:
        met_b = json.load(f)

    milu = paired_milu_bootstrap(
        json.load(open(dir_a / "milu_validation_records.json")),
        json.load(open(dir_b / "milu_validation_records.json")),
    )
    en_mr = paired_flores_bootstrap(
        json.load(open(dir_a / "flores_en_mr_records.json")),
        json.load(open(dir_b / "flores_en_mr_records.json")),
    )
    mr_en = paired_flores_bootstrap(
        json.load(open(dir_a / "flores_mr_en_records.json")),
        json.load(open(dir_b / "flores_mr_en_records.json")),
    )
    bpc = paired_bpc_bootstrap(
        json.load(open(dir_a / "bpc_records.json")),
        json.load(open(dir_b / "bpc_records.json")),
    )

    # Evaluate pre-registered E3 decision rule (docs/EVAL.md Section 5)
    # Adopt 768 if BPC improves with paired CI excluding 0,
    # OR if MILU and Flores both improve and neither gets worse beyond their CIs.
    bpc_improves_excl_0 = bpc["ci95"][1] < 0.0
    no_regression_beyond_ci = (
        milu["ci95"][1] >= 0.0
        and en_mr["ci95"][1] >= 0.0
        and mr_en["ci95"][1] >= 0.0
        and bpc["ci95"][0] <= 0.0
    )
    both_improve = (milu["delta"] > 0.0) and (en_mr["delta"] > 0.0 and mr_en["delta"] > 0.0)
    adopt_768 = bpc_improves_excl_0 or (both_improve and no_regression_beyond_ci)
    verdict = (
        "ADOPT 768"
        if adopt_768
        else "KEEP 512 (no detectable effect at this sensitivity)"
    )

    data_stats_path = Path("results/length_census/e3_s1_data_stats.json")
    data_stats = json.load(open(data_stats_path)) if data_stats_path.exists() else {}

    result = {
        "run_a": args.run_a,
        "run_b": args.run_b,
        "verdict": verdict,
        "adopt_768": adopt_768,
        "bpc_improves_excl_0": bpc_improves_excl_0,
        "milu": milu,
        "flores_en_mr": en_mr,
        "flores_mr_en": mr_en,
        "heldout_bpc": bpc,
        "auto_a": met_a.get("auto", {}),
        "auto_b": met_b.get("auto", {}),
        "data_stats": data_stats.get("arms", {}),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = [
        f"# E3 Paired Comparison: `{args.run_b}` vs `{args.run_a}` (S1 IndicAlign, 6k, 750 steps, seed 0)\n",
        f"**Pre-Registered E3 Verdict**: **{verdict}**\n",
        "| Metric | cap=512 | cap=768 | Paired Δ (768 − 512) | Paired 95% CI | Crosses 0? |",
        "|---|---:|---:|---:|---|---|",
        f"| **MILU val acc (%)** *(n={milu['n']})* | {milu['a']:.2f} | {milu['b']:.2f} | {milu['delta']:+.2f} pp | [{milu['ci95'][0]:+.2f}, {milu['ci95'][1]:+.2f}] | {'Yes' if milu['crosses_zero'] else '**No**'} *(McNemar p={milu['mcnemar_p']})* |",
        f"| **Flores en→mr chrF++** *(n={en_mr['n']})* | {en_mr['a']:.2f} | {en_mr['b']:.2f} | {en_mr['delta']:+.2f} | [{en_mr['ci95'][0]:+.2f}, {en_mr['ci95'][1]:+.2f}] | {'Yes' if en_mr['crosses_zero'] else '**No**'} |",
        f"| **Flores mr→en chrF++** *(n={mr_en['n']})* | {mr_en['a']:.2f} | {mr_en['b']:.2f} | {mr_en['delta']:+.2f} | [{mr_en['ci95'][0]:+.2f}, {mr_en['ci95'][1]:+.2f}] | {'Yes' if mr_en['crosses_zero'] else '**No**'} |",
        f"| **Held-out Marathi BPC** *(n={bpc['n']}, lower better)* | {bpc['a']:.4f} | {bpc['b']:.4f} | {bpc['delta']:+.4f} | [{bpc['ci95'][0]:+.4f}, {bpc['ci95'][1]:+.4f}] | {'Yes' if bpc['crosses_zero'] else '**No**'} |",
        "",
        "### Arm Composition & Token Budget (chat-template counter)",
    ]
    for cap_key in ("512", "768"):
        arm_info = data_stats.get("arms", {}).get(cap_key)
        if arm_info:
            tot_tok_m = arm_info["n_train"] * arm_info["mean_chat_tokens"] / 1e6
            lines.append(
                f"- **cap={cap_key}**: `n={arm_info['n_train']:,}`, mean=`{arm_info['mean_chat_tokens']:.1f}` tok (~`{tot_tok_m:.2f}M` total tok), "
                f"rows >512: `{arm_info['rows_gt_512']}` (`{arm_info['pct_rows_gt_512']:.1f}%`), sub-configs: `{arm_info['by_subconfig']}`"
            )
    lines.extend([
        "",
        "*Note: The 768 arm sees more tokens and a shifted S1 sub-corpus mix under equal examples (6,000 rows), so any observed difference reflects longer rows allowed plus more tokens and sub-corpus shift, not sequence length alone.*",
        "",
        "### Translation Auto-Metrics (do not read EOS-rate shifts as quality changes)",
        f"- **cap=512**: script_purity={met_a['auto']['script_purity']:.2f}%, lang_id_mr={met_a['auto']['lang_id_mr']:.2f}%, repeat_4gram={met_a['auto']['repeat_4gram']:.2f}%, eos_rate={met_a['auto']['eos_rate']:.2f}%, mean_len={met_a['auto']['mean_len_tokens']:.1f} tok",
        f"- **cap=768**: script_purity={met_b['auto']['script_purity']:.2f}%, lang_id_mr={met_b['auto']['lang_id_mr']:.2f}%, repeat_4gram={met_b['auto']['repeat_4gram']:.2f}%, eos_rate={met_b['auto']['eos_rate']:.2f}%, mean_len={met_b['auto']['mean_len_tokens']:.1f} tok",
    ])
    md = "\n".join(lines) + "\n"
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
