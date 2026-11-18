"""Dataset funnel statistics tracking and data card generation."""

import json
from pathlib import Path
from typing import Dict, Any, List


class FunnelTracker:
    """Tracks counts of accepted and rejected rows through data cleaning stages."""

    def __init__(self, source_name: str, pipeline_version: str = "2.2"):
        self.source_name = source_name
        self.pipeline_version = pipeline_version
        self.metadata: Dict[str, Any] = {}
        self.counts = {
            "raw_ingested": 0,
            "failed_empty_or_short": 0,
            "failed_template_artifact": 0,
            "failed_script_purity": 0,
            "failed_latin_ratio": 0,
            "failed_repetition": 0,
            "failed_lang_id": 0,
            "failed_max_seq_len": 0,
            "passed_clean": 0,
            "failed_dedup": 0,
            "passed_dedup": 0,
            "failed_decontamination": 0,
            "final_accepted": 0,
        }
        self.rejection_details: Dict[str, int] = {}

    def log_rejection(self, stage: str, reason: str):
        key = f"failed_{stage}"
        self.counts[key] = self.counts.get(key, 0) + 1
        self.rejection_details[reason] = self.rejection_details.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "pipeline_version": self.pipeline_version,
            "source": self.source_name,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        d["counts"] = self.counts
        d["rejection_details"] = self.rejection_details
        return d

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def format_funnel_markdown_table(stats_list: List[Dict[str, Any]]) -> str:
    """Formats a markdown comparison table of the cleaning funnels across sources."""
    headers = [
        "Source",
        "Raw Ingested",
        "Clean Filter Pass",
        "Dedup Pass",
        "Decontaminated",
        "Final Yield (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for s in stats_list:
        c = s.get("counts", {})
        raw = c.get("raw_ingested", 0)
        clean = c.get("passed_clean", 0)
        dedup = c.get("passed_dedup", 0)
        final = c.get("final_accepted", 0)
        yield_pct = (final / raw * 100.0) if raw > 0 else 0.0

        row = [
            f"**{s.get('source')}**",
            f"{raw:,}",
            f"{clean:,}",
            f"{dedup:,}",
            f"{final:,}",
            f"{yield_pct:.1f}%",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
