"""Computes paired bootstrap 95% CIs for E3 (S1-6k cap=768 vs cap=512)."""

import argparse
import json
from pathlib import Path

from marathi_llm.eval.paired import (
    paired_milu_bootstrap,
    paired_flores_bootstrap,
    paired_bpc_bootstrap,
)



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
