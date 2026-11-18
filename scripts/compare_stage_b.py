"""Summarizes and compares Stage B data screening results against the control (A_qwen25_1p5b_768)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    runs_dir = Path("results/runs")
    control_id = "A_qwen25_1p5b_768"
    b_runs = [
        ("B1 (Control, S0 Alpaca 12k)", control_id),
        ("B2 (S1 IndicAlign 12k)", "B2_s1_indicalign_12k"),
        ("B3 (S2 SandLogic 12k)", "B3_s2_sandlogic_12k"),
        ("B4 (S3 Aya 1.2k)", "B4_s3_aya_1p2k"),
    ]

    print(f"\n{'='*100}")
    print(f"{'Run':<30} | {'MILU (%)':<10} | {'en->mr':<8} | {'mr->en':<8} | {'BPC':<8} | {'Purity':<8} | {'mr-ID':<8} | {'Rep4':<8}")
    print(f"{'-'*100}")

    summary_rows = []
    for label, rid in b_runs:
        m = load_metrics(runs_dir / rid / "metrics.json")
        if not m:
            print(f"{label:<30} | {'(pending)':<10} | {'--':<8} | {'--':<8} | {'--':<8} | {'--':<8} | {'--':<8} | {'--':<8}")
            continue

        milu = m.get("milu_mr", {}).get("acc", 0.0)
        en_mr = m.get("flores", {}).get("en_mr_chrfpp", 0.0)
        mr_en = m.get("flores", {}).get("mr_en_chrfpp", 0.0)
        bpc = m.get("heldout_bpc", {}).get("flores_mr_dev", 0.0)
        purity = m.get("auto", {}).get("script_purity", 0.0)
        lang_id = m.get("auto", {}).get("lang_id_mr", 0.0)
        rep4 = m.get("auto", {}).get("repeat_4gram", 0.0)

        print(f"{label:<30} | {milu:10.2f} | {en_mr:8.2f} | {mr_en:8.2f} | {bpc:8.4f} | {purity:8.1f}% | {lang_id:8.1f}% | {rep4:8.1f}%")
        summary_rows.append({
            "label": label,
            "run_id": rid,
            "milu": milu,
            "flores_en_mr": en_mr,
            "flores_mr_en": mr_en,
            "bpc": bpc,
            "script_purity": purity,
            "lang_id_mr": lang_id,
            "repeat_4gram": rep4,
        })

    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
