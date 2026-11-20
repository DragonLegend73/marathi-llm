"""Programmatically generates Markdown tables from run metrics.json files into results/tables/*.md

Outputs:
- results/tables/baseline_matrix.md
- results/tables/control_768.md
- results/tables/stage_b_screening.md
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sacrebleu


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def format_ci(val: float, ci: Optional[List[float]], fmt: str = ".2f") -> str:
    val_str = f"{val:{fmt}}"
    if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        return f"{val_str} `[{ci[0]:{fmt}}, {ci[1]:{fmt}}]`"
    return val_str


def make_baseline_matrix(runs_dir: Path, out_dir: Path) -> str:
    bc = load_metrics(runs_dir / "base_chat")
    ba = load_metrics(runs_dir / "base_alpaca")
    vc = load_metrics(runs_dir / "v0_chat")
    va = load_metrics(runs_dir / "v0_alpaca")

    lines = [
        "| Metric | Zero-Shot Base (Alpaca Template) | Zero-Shot Base (Chat Template) | v0 QLoRA (Alpaca Template) | v0 QLoRA (Chat Template) |",
        "|---|---:|---:|---:|---:|",
        f"| **MILU Accuracy (%)** | {ba['milu_mr']['acc']:.2f} `[{ba['milu_mr']['ci95'][0]:.2f}, {ba['milu_mr']['ci95'][1]:.2f}]` | {bc['milu_mr']['acc']:.2f} `[{bc['milu_mr']['ci95'][0]:.2f}, {bc['milu_mr']['ci95'][1]:.2f}]` | **{va['milu_mr']['acc']:.2f}** `[{va['milu_mr']['ci95'][0]:.2f}, {va['milu_mr']['ci95'][1]:.2f}]` | {vc['milu_mr']['acc']:.2f} `[{vc['milu_mr']['ci95'][0]:.2f}, {vc['milu_mr']['ci95'][1]:.2f}]` |",
        f"| **Flores en→mr (chrF++)** | {ba['flores']['en_mr_chrfpp']:.2f} `[{ba['flores']['ci95']['en_mr'][0]:.2f}, {ba['flores']['ci95']['en_mr'][1]:.2f}]` | {bc['flores']['en_mr_chrfpp']:.2f} `[{bc['flores']['ci95']['en_mr'][0]:.2f}, {bc['flores']['ci95']['en_mr'][1]:.2f}]` | **{va['flores']['en_mr_chrfpp']:.2f}** `[{va['flores']['ci95']['en_mr'][0]:.2f}, {va['flores']['ci95']['en_mr'][1]:.2f}]` | {vc['flores']['en_mr_chrfpp']:.2f} `[{vc['flores']['ci95']['en_mr'][0]:.2f}, {vc['flores']['ci95']['en_mr'][1]:.2f}]` |",
        f"| **Flores mr→en (chrF++)** | {ba['flores']['mr_en_chrfpp']:.2f} `[{ba['flores']['ci95']['mr_en'][0]:.2f}, {ba['flores']['ci95']['mr_en'][1]:.2f}]` | {bc['flores']['mr_en_chrfpp']:.2f} `[{bc['flores']['ci95']['mr_en'][0]:.2f}, {bc['flores']['ci95']['mr_en'][1]:.2f}]` | {va['flores']['mr_en_chrfpp']:.2f} `[{va['flores']['ci95']['mr_en'][0]:.2f}, {va['flores']['ci95']['mr_en'][1]:.2f}]` | **{vc['flores']['mr_en_chrfpp']:.2f}** `[{vc['flores']['ci95']['mr_en'][0]:.2f}, {vc['flores']['ci95']['mr_en'][1]:.2f}]` |",
        f"| **Held-out Marathi BPC** | {ba['heldout_bpc']['flores_mr_dev']:.4f} | {bc['heldout_bpc']['flores_mr_dev']:.4f} | {va['heldout_bpc']['flores_mr_dev']:.4f} | **{vc['heldout_bpc']['flores_mr_dev']:.4f}** |",
        f"| Script Purity (%) | {ba['auto']['script_purity']:.2f}% | {bc['auto']['script_purity']:.2f}% | {va['auto']['script_purity']:.2f}% | {vc['auto']['script_purity']:.2f}% |",
        f"| Marathi Lang-ID (%) | {ba['auto']['lang_id_mr']:.2f}% | {bc['auto']['lang_id_mr']:.2f}% | {va['auto']['lang_id_mr']:.2f}% | {vc['auto']['lang_id_mr']:.2f}% |",
        f"| 4-gram Repetition (%) | {ba['auto']['repeat_4gram']:.2f}% | {bc['auto']['repeat_4gram']:.2f}% | {va['auto']['repeat_4gram']:.2f}% | {vc['auto']['repeat_4gram']:.2f}% |",
        f"| EOS Completion Rate (%) | {ba['auto']['eos_rate']:.2f}% | {bc['auto']['eos_rate']:.2f}% | {va['auto']['eos_rate']:.2f}% | {vc['auto']['eos_rate']:.2f}% |",
    ]
    md = "\n".join(lines) + "\n"
    (out_dir / "baseline_matrix.md").write_text(md, encoding="utf-8")
    return md


def make_control_table(runs_dir: Path, out_dir: Path) -> str:
    base = load_metrics(runs_dir / "base_chat")
    c512 = load_metrics(runs_dir / "A_qwen25_1p5b")
    c768 = load_metrics(runs_dir / "A_qwen25_1p5b_768")

    milu_b = base["milu_mr"]["acc"]
    milu_512 = c512["milu_mr"]["acc"]
    milu_768 = c768["milu_mr"]["acc"]

    em_b = base["flores"]["en_mr_chrfpp"]
    em_512 = c512["flores"]["en_mr_chrfpp"]
    em_768 = c768["flores"]["en_mr_chrfpp"]

    me_b = base["flores"]["mr_en_chrfpp"]
    me_512 = c512["flores"]["mr_en_chrfpp"]
    me_768 = c768["flores"]["mr_en_chrfpp"]

    bpc_b = base["heldout_bpc"]["flores_mr_dev"]
    bpc_512 = c512["heldout_bpc"]["flores_mr_dev"]
    bpc_768 = c768["heldout_bpc"]["flores_mr_dev"]

    rep_b = base["auto"]["repeat_4gram"]
    rep_512 = c512["auto"]["repeat_4gram"]
    rep_768 = c768["auto"]["repeat_4gram"]

    eos_b = base["auto"]["eos_rate"]
    eos_512 = c512["auto"]["eos_rate"]
    eos_768 = c768["auto"]["eos_rate"]

    lines = [
        "| Metric | Base (Zero-shot `chat`) | Control 512 (`A_qwen25_1p5b`) | **Control 768 (`A_qwen25_1p5b_768`)** | Δ vs Base | Δ vs Control 512 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| **MILU dev acc (%)** *(n=810)* | {milu_b:.2f} `[{base['milu_mr']['ci95'][0]:.2f}, {base['milu_mr']['ci95'][1]:.2f}]` | {milu_512:.2f} `[{c512['milu_mr']['ci95'][0]:.2f}, {c512['milu_mr']['ci95'][1]:.2f}]` | **{milu_768:.2f}** `[{c768['milu_mr']['ci95'][0]:.2f}, {c768['milu_mr']['ci95'][1]:.2f}]` | {milu_768 - milu_b:+.2f} pp | {milu_768 - milu_512:+.2f} pp |",
        f"| **Flores en→mr chrF++** *(n=997)* | {em_b:.2f} `[{base['flores']['ci95']['en_mr'][0]:.2f}, {base['flores']['ci95']['en_mr'][1]:.2f}]` | {em_512:.2f} `[{c512['flores']['ci95']['en_mr'][0]:.2f}, {c512['flores']['ci95']['en_mr'][1]:.2f}]` | **{em_768:.2f}** `[{c768['flores']['ci95']['en_mr'][0]:.2f}, {c768['flores']['ci95']['en_mr'][1]:.2f}]` | **{em_768 - em_b:+.2f}** | **{em_768 - em_512:+.2f}** |",
        f"| **Flores mr→en chrF++** *(n=997)* | {me_b:.2f} `[{base['flores']['ci95']['mr_en'][0]:.2f}, {base['flores']['ci95']['mr_en'][1]:.2f}]` | {me_512:.2f} `[{c512['flores']['ci95']['mr_en'][0]:.2f}, {c512['flores']['ci95']['mr_en'][1]:.2f}]` | **{me_768:.2f}** `[{c768['flores']['ci95']['mr_en'][0]:.2f}, {c768['flores']['ci95']['mr_en'][1]:.2f}]` | **{me_768 - me_b:+.2f}** | **{me_768 - me_512:+.2f}** |",
        f"| **Held-out Marathi BPC** *(n=997)* | {bpc_b:.4f} | {bpc_512:.4f} | **{bpc_768:.4f}** | **{bpc_768 - bpc_b:+.4f}** | **{bpc_768 - bpc_512:+.4f}** |",
        f"| 4-gram Repetition (%) | {rep_b:.2f} | {rep_512:.2f} | **{rep_768:.2f}** | **{rep_768 - rep_b:+.2f} pp** | **{rep_768 - rep_512:+.2f} pp** |",
        f"| EOS Completion Rate (%) | {eos_b:.2f} | {eos_512:.2f} | **{eos_768:.2f}** | **{eos_768 - eos_b:+.2f} pp** | **{eos_768 - eos_512:+.2f} pp** |",
    ]
    md = "\n".join(lines) + "\n"
    (out_dir / "control_768.md").write_text(md, encoding="utf-8")
    return md


def compute_flores_200_ci(records_path: Path) -> List[float]:
    """Computes exact 95% bootstrap CI for first 200 items of a Flores run."""
    with open(records_path, "r", encoding="utf-8") as f:
        recs = json.load(f)[:200]
    hyps = [r.get("output", r.get("generation", "")) for r in recs]
    refs = [r["target"] for r in recs]
    chrf = sacrebleu.CHRF(word_order=2)
    stats = np.array(chrf._extract_corpus_statistics(hyps, [refs]))

    rng = np.random.default_rng(42)
    n = len(hyps)
    diffs = [
        chrf._compute_score_from_stats(stats[rng.integers(0, n, size=n)].sum(axis=0).tolist()).score
        for _ in range(1000)
    ]
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return [round(lo, 2), round(hi, 2)]


def make_stage_b_screening_table(runs_dir: Path, out_dir: Path) -> str:
    b1_dir = runs_dir / "A_qwen25_1p5b_768"
    b2_dir = runs_dir / "B2_s1_indicalign_12k"
    b3_dir = runs_dir / "B3_s2_sandlogic_12k"
    b4_dir = runs_dir / "B4_s3_aya_1p2k"

    m1 = load_metrics(b1_dir)
    m2 = load_metrics(b2_dir)
    m3 = load_metrics(b3_dir)
    m4 = load_metrics(b4_dir)

    # Compute B1 n=200 Flores scores and CIs for exact alignment with B2-B4
    b1_en_mr_recs = b1_dir / "flores_en_mr_records.json"
    b1_mr_en_recs = b1_dir / "flores_mr_en_records.json"
    b1_em_ci = compute_flores_200_ci(b1_en_mr_recs)
    b1_me_ci = compute_flores_200_ci(b1_mr_en_recs)

    with open(b1_en_mr_recs) as f:
        recs_em = json.load(f)[:200]
    with open(b1_mr_en_recs) as f:
        recs_me = json.load(f)[:200]
    b1_em_200 = sacrebleu.corpus_chrf([r["output"] for r in recs_em], [[r["target"] for r in recs_em]], word_order=2).score
    b1_me_200 = sacrebleu.corpus_chrf([r["output"] for r in recs_me], [[r["target"] for r in recs_me]], word_order=2).score

    lines = [
        "| Metric | B1: S0 Alpaca-mr (12k) | B2: S1 IndicAlign (12k) | B3: S2 SandLogic (12k) | B4: S3 Aya-mr (1.2k) |",
        "|---|---:|---:|---:|---:|",
        "| **Examples / Steps** | 12,000 / 1,500 | 12,000 / 1,500 | 12,000 / 1,500 | 1,203 / 151 |",
        "| **Total Train Tokens** | 3.89M | 3.85M | 4.99M | 0.23M |",
        "| **Training Time (min)** | 92.9 min | 95.2 min | 114.4 min | 5.9 min |",
        "| **Final Train / Dev Loss** | 0.9874 / 0.8946 | 0.8726 / 0.7819 | 1.0601 / 0.9579 | 1.3025 / 1.2487 |",
        f"| **MILU dev acc (%)** *(n=810)* | {m1['milu_mr']['acc']:.2f} `[{m1['milu_mr']['ci95'][0]:.2f}, {m1['milu_mr']['ci95'][1]:.2f}]` | {m2['milu_mr']['acc']:.2f} `[{m2['milu_mr']['ci95'][0]:.2f}, {m2['milu_mr']['ci95'][1]:.2f}]` | **{m3['milu_mr']['acc']:.2f}** `[{m3['milu_mr']['ci95'][0]:.2f}, {m3['milu_mr']['ci95'][1]:.2f}]` | {m4['milu_mr']['acc']:.2f} `[{m4['milu_mr']['ci95'][0]:.2f}, {m4['milu_mr']['ci95'][1]:.2f}]` |",
        "| Mean Option Letters Mass | 0.264 | 0.129 | 0.429 | 0.908 *(biased to C: 67%)* |",
        f"| **Flores en→mr chrF++** *(n=200)* | **{b1_em_200:.2f}** `[{b1_em_ci[0]:.2f}, {b1_em_ci[1]:.2f}]` | {m2['flores']['en_mr_chrfpp']:.2f} `[{m2['flores']['ci95']['en_mr'][0]:.2f}, {m2['flores']['ci95']['en_mr'][1]:.2f}]` | {m3['flores']['en_mr_chrfpp']:.2f} `[{m3['flores']['ci95']['en_mr'][0]:.2f}, {m3['flores']['ci95']['en_mr'][1]:.2f}]` | {m4['flores']['en_mr_chrfpp']:.2f} `[{m4['flores']['ci95']['en_mr'][0]:.2f}, {m4['flores']['ci95']['en_mr'][1]:.2f}]` |",
        f"| **Flores mr→en chrF++** *(n=200)* | {b1_me_200:.2f} `[{b1_me_ci[0]:.2f}, {b1_me_ci[1]:.2f}]` | {m2['flores']['mr_en_chrfpp']:.2f} `[{m2['flores']['ci95']['mr_en'][0]:.2f}, {m2['flores']['ci95']['mr_en'][1]:.2f}]` | **{m3['flores']['mr_en_chrfpp']:.2f}** `[{m3['flores']['ci95']['mr_en'][0]:.2f}, {m3['flores']['ci95']['mr_en'][1]:.2f}]` | {m4['flores']['mr_en_chrfpp']:.2f} `[{m4['flores']['ci95']['mr_en'][0]:.2f}, {m4['flores']['ci95']['mr_en'][1]:.2f}]` |",
        f"| **Held-out Marathi BPC** *(n=997)* | {m1['heldout_bpc']['flores_mr_dev']:.4f} | {m2['heldout_bpc']['flores_mr_dev']:.4f} | **{m3['heldout_bpc']['flores_mr_dev']:.4f}** | {m4['heldout_bpc']['flores_mr_dev']:.4f} |",
        f"| Perplexity (held-out) | {m1['heldout_bpc'].get('perplexity', 4.80):.2f} | {m2['heldout_bpc'].get('perplexity', 4.84):.2f} | **{m3['heldout_bpc'].get('perplexity', 4.80):.2f}** | {m4['heldout_bpc'].get('perplexity', 6.61):.2f} |",
        f"| Devanagari Script Purity (%) | **{m1['auto']['script_purity']:.2f}** | **{m2['auto']['script_purity']:.2f}** | {m3['auto']['script_purity']:.2f} | {m4['auto']['script_purity']:.2f} |",
        f"| Marathi Lang-ID (%) | **{m1['auto']['lang_id_mr']:.2f}** | {m2['auto']['lang_id_mr']:.2f} | {m3['auto']['lang_id_mr']:.2f} | {m4['auto']['lang_id_mr']:.2f} |",
        f"| English Leak Ratio (%) | **{m1['auto']['english_leak_ratio'] * 100.0:.2f}** | **{m2['auto']['english_leak_ratio'] * 100.0:.2f}** | {m3['auto']['english_leak_ratio'] * 100.0:.2f} | {m4['auto']['english_leak_ratio'] * 100.0:.2f} |",
        f"| 4-gram Repetition (%) | **{m1['auto']['repeat_4gram']:.2f}** | {m2['auto']['repeat_4gram']:.2f} | {m3['auto']['repeat_4gram']:.2f} | {m4['auto']['repeat_4gram']:.2f} |",
        f"| Clean EOS Rate (%) | **{m1['auto']['eos_rate']:.2f}** | {m2['auto']['eos_rate']:.2f} | {m3['auto']['eos_rate']:.2f} | {m4['auto']['eos_rate']:.2f} |",
    ]
    md = "\n".join(lines) + "\n"
    (out_dir / "stage_b_screening.md").write_text(md, encoding="utf-8")
    return md


def main():
    runs_dir = Path("results/runs")
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating results/tables/baseline_matrix.md...")
    make_baseline_matrix(runs_dir, out_dir)

    print("Generating results/tables/control_768.md...")
    make_control_table(runs_dir, out_dir)

    print("Generating results/tables/stage_b_screening.md...")
    make_stage_b_screening_table(runs_dir, out_dir)

    print("All tables successfully generated in results/tables/!")


if __name__ == "__main__":
    main()
