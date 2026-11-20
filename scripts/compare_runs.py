"""Script for paired comparison of two evaluation runs on identical benchmarks."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from marathi_llm.eval.paired import (
    paired_milu_bootstrap,
    paired_flores_bootstrap,
    paired_bpc_bootstrap,
)


def resolve_run_dir(run_spec: str, base_dir: Path) -> Path:
    p = Path(run_spec)
    if p.exists() and p.is_dir():
        return p
    candidate = base_dir / run_spec
    if candidate.exists() and candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find run directory for '{run_spec}' (checked '{p}' and '{candidate}')")


def load_records(run_dir: Path, filename: str) -> Optional[List[Dict[str, Any]]]:
    path = run_dir / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Paired statistical comparison between two runs")
    parser.add_argument("--a", required=True, help="Run ID or path for Baseline / Arm A")
    parser.add_argument("--b", required=True, help="Run ID or path for Candidate / Arm B")
    parser.add_argument("--runs-dir", default="results/runs", help="Base directory containing runs")
    parser.add_argument("--flores-limit", type=int, default=200, help="Max Flores examples to compare (default: 200)")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path")
    parser.add_argument("--out-md", default=None, help="Optional output Markdown path")
    args = parser.parse_args()

    base_dir = Path(args.runs_dir)
    dir_a = resolve_run_dir(args.a, base_dir)
    dir_b = resolve_run_dir(args.b, base_dir)

    run_a_id = dir_a.name
    run_b_id = dir_b.name

    met_a = load_metrics(dir_a)
    met_b = load_metrics(dir_b)

    results: Dict[str, Any] = {
        "run_a": run_a_id,
        "run_b": run_b_id,
        "flores_limit": args.flores_limit,
    }

    # 1. MILU
    milu_a = load_records(dir_a, "milu_validation_records.json")
    milu_b = load_records(dir_b, "milu_validation_records.json")
    if milu_a and milu_b:
        results["milu"] = paired_milu_bootstrap(milu_a, milu_b)
    else:
        results["milu"] = None

    # 2. Flores en->mr
    en_mr_a = load_records(dir_a, "flores_en_mr_records.json")
    en_mr_b = load_records(dir_b, "flores_en_mr_records.json")
    if en_mr_a and en_mr_b:
        results["flores_en_mr"] = paired_flores_bootstrap(en_mr_a, en_mr_b, limit=args.flores_limit)
    else:
        results["flores_en_mr"] = None

    # 3. Flores mr->en
    mr_en_a = load_records(dir_a, "flores_mr_en_records.json")
    mr_en_b = load_records(dir_b, "flores_mr_en_records.json")
    if mr_en_a and mr_en_b:
        results["flores_mr_en"] = paired_flores_bootstrap(mr_en_a, mr_en_b, limit=args.flores_limit)
    else:
        results["flores_mr_en"] = None

    # 4. Heldout BPC
    bpc_a = load_records(dir_a, "bpc_records.json")
    bpc_b = load_records(dir_b, "bpc_records.json")
    if bpc_a and bpc_b:
        results["heldout_bpc"] = paired_bpc_bootstrap(bpc_a, bpc_b)
    else:
        results["heldout_bpc"] = None

    # Auto metrics
    results["auto_a"] = met_a.get("auto", {})
    results["auto_b"] = met_b.get("auto", {})

    # Generate Markdown summary
    lines = [
        f"# Paired Run Comparison: `{run_b_id}` (B) vs `{run_a_id}` (A)\n",
        "| Metric | Arm A | Arm B | Paired Δ (B − A) | Paired 95% CI | CI Excludes 0? |",
        "|---|---:|---:|---:|---|---|",
    ]

    if results["milu"]:
        m = results["milu"]
        excl = "**Yes**" if not m["crosses_zero"] else "No"
        lines.append(
            f"| **MILU val acc (%)** *(n={m['n']})* | {m['a']:.2f} | {m['b']:.2f} | {m['delta']:+.2f} pp | [{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}] | {excl} *(McNemar p={m['mcnemar_p']})* |"
        )
    if results["flores_en_mr"]:
        f_em = results["flores_en_mr"]
        excl = "**Yes**" if not f_em["crosses_zero"] else "No"
        lines.append(
            f"| **Flores en→mr chrF++** *(n={f_em['n']})* | {f_em['a']:.2f} | {f_em['b']:.2f} | {f_em['delta']:+.2f} | [{f_em['ci95'][0]:+.2f}, {f_em['ci95'][1]:+.2f}] | {excl} |"
        )
    if results["flores_mr_en"]:
        f_me = results["flores_mr_en"]
        excl = "**Yes**" if not f_me["crosses_zero"] else "No"
        lines.append(
            f"| **Flores mr→en chrF++** *(n={f_me['n']})* | {f_me['a']:.2f} | {f_me['b']:.2f} | {f_me['delta']:+.2f} | [{f_me['ci95'][0]:+.2f}, {f_me['ci95'][1]:+.2f}] | {excl} |"
        )
    if results["heldout_bpc"]:
        b = results["heldout_bpc"]
        excl = "**Yes**" if not b["crosses_zero"] else "No"
        lines.append(
            f"| **Held-out Marathi BPC** *(n={b['n']}, lower better)* | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {excl} |"
        )

    lines.extend([
        "",
        "### Generation Auto-Metrics Comparison",
        f"- **Arm A (`{run_a_id}`)**: script_purity={results['auto_a'].get('script_purity', 0.0):.2f}%, lang_id_mr={results['auto_a'].get('lang_id_mr', 0.0):.2f}%, repeat_4gram={results['auto_a'].get('repeat_4gram', 0.0):.2f}%, eos_rate={results['auto_a'].get('eos_rate', 0.0):.2f}%, mean_len={results['auto_a'].get('mean_len_tokens', 0.0):.1f} tok",
        f"- **Arm B (`{run_b_id}`)**: script_purity={results['auto_b'].get('script_purity', 0.0):.2f}%, lang_id_mr={results['auto_b'].get('lang_id_mr', 0.0):.2f}%, repeat_4gram={results['auto_b'].get('repeat_4gram', 0.0):.2f}%, eos_rate={results['auto_b'].get('eos_rate', 0.0):.2f}%, mean_len={results['auto_b'].get('mean_len_tokens', 0.0):.1f} tok",
    ])

    md_out = "\n".join(lines) + "\n"
    print(md_out)

    if args.out_json:
        out_j = Path(args.out_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved paired results JSON to {args.out_json}")

    if args.out_md:
        out_m = Path(args.out_md)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        with open(out_m, "w", encoding="utf-8") as f:
            f.write(md_out)
        print(f"Saved paired results Markdown to {args.out_md}")


if __name__ == "__main__":
    main()
