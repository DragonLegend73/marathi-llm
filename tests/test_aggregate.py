import json
import pytest
from pathlib import Path
from marathi_llm.aggregate import load_all_runs, build_overview_table, calculate_stage_a_ranking


def test_aggregate_load_and_format(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create dummy run A
    run_a = runs_dir / "run_a"
    run_a.mkdir()
    metrics_a = {
        "run_id": "run_a",
        "model": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "adapter": "none",
        "prompt_format": "chat",
        "milu_mr": {"acc": 27.16, "ci95": [24.2, 30.3]},
        "flores": {
            "en_mr_chrfpp": 9.34,
            "mr_en_chrfpp": 25.87,
            "ci95": {"en_mr": [9.05, 9.60], "mr_en": [25.17, 26.57]},
        },
        "heldout_bpc": {"flores_mr_dev": 2.96, "ci95": [2.90, 3.02]},
        "auto": {"script_purity": 90.7, "eos_rate": 37.4},
    }
    with open(run_a / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_a, f)

    # Create dummy run B
    run_b = runs_dir / "run_b"
    run_b.mkdir()
    metrics_b = {
        "run_id": "run_b",
        "model": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "adapter": "DragonLegend/marathi-qwen2.5-lora",
        "prompt_format": "alpaca_v0",
        "milu_mr": {"acc": 30.50, "ci95": [27.5, 33.5]},
        "flores": {
            "en_mr_chrfpp": 12.50,
            "mr_en_chrfpp": 26.10,
            "ci95": {"en_mr": [12.1, 12.9], "mr_en": [25.5, 26.7]},
        },
        "heldout_bpc": {"flores_mr_dev": 2.85, "ci95": [2.80, 2.90]},
        "auto": {"script_purity": 94.2, "eos_rate": 81.5},
    }
    with open(run_b / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_b, f)

    runs = load_all_runs(runs_dir)
    assert len(runs) == 2
    assert "run_a" in runs
    assert "run_b" in runs

    table = build_overview_table(runs)
    assert "run_a" in table
    assert "run_b" in table
    assert "27.16" in table
    assert "30.50" in table

    ranking = calculate_stage_a_ranking(runs)
    assert "#1" in ranking
    assert "run_b" in ranking
