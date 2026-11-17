"""Audit script checking training data overlap against all evaluation benchmarks across n-gram sizes."""

import os
import json
import collections
from pathlib import Path
from datasets import load_dataset
from marathi_llm.data.decontaminate import BenchmarkDecontaminator, load_all_eval_targets
from marathi_llm.progress import get_progress_bar

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def run_audit(sizes=(8, 12, 20)):
    print("Loading raw training dataset: rachittshah/alpaca-marahti...")
    ds = load_dataset("rachittshah/alpaca-marahti", split="train")
    print(f"Loaded {len(ds):,} raw rows.")

    report = {}
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for n in sizes:
        print(f"\n--- Checking overlap for n-gram size = {n} ---")
        dec = BenchmarkDecontaminator(ngram_size=n)
        load_all_eval_targets(dec)

        hits = collections.Counter()
        samples = []

        pbar = get_progress_bar(total=len(ds), desc=f"Auditing n={n}", unit="row")
        for i, row in enumerate(ds):
            # Exactly what v0 trained on:
            text = (row.get("translated_text") or "")
            bad, tag = dec.check_contamination(text)
            if bad:
                hits[tag] += 1
                if len(samples) < 20:
                    samples.append({"row": i, "tag": tag, "text": text[:200]})
            pbar.update(1)
        pbar.close()

        total_hits = sum(hits.values())
        print(f"Audit results for n={n}: Total hits = {total_hits:,} / {len(ds):,} ({total_hits / len(ds) * 100:.2f}%)")
        print("Breakdown by tag:")
        for t, count in hits.most_common(15):
            print(f"  {t}: {count}")

        report[n] = {
            "total_rows": len(ds),
            "total_hits": total_hits,
            "hit_percentage": total_hits / len(ds) * 100,
            "hits": dict(hits),
            "samples": samples,
        }

    out_file = results_dir / "audit_v0_contamination.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved audit report to {out_file}")
    return report


if __name__ == "__main__":
    run_audit()
