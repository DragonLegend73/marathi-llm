"""Fixed-budget dataset builder orchestrating cleaning, dedup, decontamination, and sampling."""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from marathi_llm.progress import get_progress_bar
from marathi_llm.data.load import (
    stream_s0_alpaca,
    stream_s1_indicalign,
    stream_s2_sandlogic,
    stream_s3_aya,
)
from marathi_llm.data.clean import normalize_marathi_text, validate_example
from marathi_llm.data.dedup import DatasetDeduplicator
from marathi_llm.data.decontaminate import BenchmarkDecontaminator, load_all_eval_targets
from marathi_llm.data.stats import FunnelTracker


def build_dataset_pipeline(
    config_path: str,
    output_dir: str = "data/processed",
    eval_decontaminator: Optional[BenchmarkDecontaminator] = None,
    tokenizer=None,
) -> Dict[str, Any]:
    """Runs the end-to-end cleaning and build pipeline for a given data source config."""
    cfg_file = Path(config_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    source_name = cfg["name"]
    source_id = cfg.get("source_id", source_name)
    print(f"\n==========================================")
    print(f"Building Dataset Pipeline: {source_name} ({source_id})")
    print(f"Config: {config_path}")
    print(f"==========================================\n")

    tracker = FunnelTracker(source_name, pipeline_version="2.1")
    deduplicator = DatasetDeduplicator(
        threshold=cfg.get("dedup", {}).get("minhash_threshold", 0.8),
        shingle_size=cfg.get("dedup", {}).get("ngram_size", 5),
    )

    clean_cfg = cfg.get("clean", {})
    min_purity = clean_cfg.get("min_devanagari_purity", 60.0)
    max_latin = clean_cfg.get("max_latin_ratio", 0.15)
    min_resp_chars = clean_cfg.get("min_response_chars", 10)
    require_mr_lang = clean_cfg.get("require_mr_lang_id", False)
    max_seq_len = clean_cfg.get("max_seq_len", 512)

    # 1. Select stream loader
    if source_id == "S0" or "alpaca" in source_name:
        stream = stream_s0_alpaca()
    elif source_id == "S1" or "indicalign" in source_name:
        stream = stream_s1_indicalign(configs=cfg.get("configs"))
    elif source_id == "S2" or "sandlogic" in source_name:
        stream = stream_s2_sandlogic(
            split=cfg.get("split", "marathi"),
            limit=cfg.get("sample_limit", 200000),
        )
    elif source_id == "S3" or "aya" in source_name:
        stream = stream_s3_aya()
    else:
        raise ValueError(f"Unknown source: {source_name}")

    # Set aside Aya eval held-outs if S3
    native_eval_prompts = []
    bpc_eval_responses = []

    accepted_pool = []
    pbar = get_progress_bar(total=200000, desc=f"Processing {source_id}", unit="row")

    for raw_row in stream:
        tracker.counts["raw_ingested"] += 1
        pbar.update(1)

        # Normalize text
        inst = normalize_marathi_text(raw_row.get("instruction", ""))
        inp = normalize_marathi_text(raw_row.get("input", ""))
        resp = normalize_marathi_text(raw_row.get("response", ""))

        # If S3, carve out held-outs first before any training consideration
        if source_id == "S3" and len(native_eval_prompts) < 100:
            native_eval_prompts.append({
                "id": f"aya_eval_{len(native_eval_prompts)}",
                "prompt": inst,
                "reference": resp,
            })
            continue

        if source_id == "S3" and len(bpc_eval_responses) < 500:
            bpc_eval_responses.append(resp)
            continue

        # Validation & Quality Filtering
        is_valid, reject_reason = validate_example(
            instruction=inst,
            response=resp,
            input_text=inp,
            min_devanagari_purity=min_purity,
            max_latin_ratio=max_latin,
            min_response_chars=min_resp_chars,
            require_mr_lang_id=require_mr_lang,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        if not is_valid:
            stage = "clean"
            if "script_purity" in reject_reason:
                stage = "script_purity"
            elif "latin" in reject_reason:
                stage = "latin_ratio"
            elif "repetition" in reject_reason:
                stage = "repetition"
            elif "language_id" in reject_reason:
                stage = "lang_id"
            elif "seq_len" in reject_reason:
                stage = "max_seq_len"
            elif "artifact" in reject_reason:
                stage = "template_artifact"
            elif "short" in reject_reason or "empty" in reject_reason:
                stage = "empty_or_short"
            tracker.log_rejection(stage, reject_reason)
            continue

        tracker.counts["passed_clean"] += 1

        # Deduplication (Exact + MinHash)
        item_id = raw_row.get("id", f"{source_id}_{tracker.counts['passed_clean']}")
        if deduplicator.is_duplicate(f"{inst} {resp}", key_id=item_id):
            tracker.log_rejection("dedup", "duplicate_instruction_or_response")
            continue

        tracker.counts["passed_dedup"] += 1

        # Decontamination against all evaluation sets
        if eval_decontaminator is not None:
            is_contam, contam_bench = eval_decontaminator.check_contamination(f"{inst} {resp}")
            if is_contam:
                tracker.log_rejection("decontamination", contam_bench)
                continue

        tracker.counts["final_accepted"] += 1
        accepted_pool.append({
            "id": item_id,
            "instruction": inst,
            "input": inp,
            "response": resp,
            "source": raw_row.get("source", source_name),
        })

    pbar.close()

    # Save held-outs if S3
    if native_eval_prompts:
        eval_dir = Path("data/eval")
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "native_prompts.jsonl", "w", encoding="utf-8") as f:
            for p in native_eval_prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved {len(native_eval_prompts)} held-out native evaluation prompts to data/eval/native_prompts.jsonl")

    if bpc_eval_responses:
        eval_dir = Path("data/eval")
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "bpc_responses.jsonl", "w", encoding="utf-8") as f:
            for r in bpc_eval_responses:
                f.write(json.dumps({"response": r}, ensure_ascii=False) + "\n")
        print(f"Saved {len(bpc_eval_responses)} held-out BPC evaluation responses to data/eval/bpc_responses.jsonl")

    # Sample Fixed Budget: 12k train, 500 dev
    budget_cfg = cfg.get("budget", {})
    seed = budget_cfg.get("seed", 42)
    random.seed(seed)
    random.shuffle(accepted_pool)

    dev_count = min(budget_cfg.get("dev_samples", 500), len(accepted_pool) // 10)
    train_target = budget_cfg.get("train_samples", 12000)

    dev_set = accepted_pool[:dev_count]
    remaining = accepted_pool[dev_count:]

    if train_target and len(remaining) >= train_target:
        train_set = remaining[:train_target]
    else:
        train_set = remaining

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    train_file = out_p / f"{source_name}_12k.jsonl"
    dev_file = out_p / f"{source_name}_dev500.jsonl"
    stats_file = out_p / f"{source_name}.stats.json"

    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(dev_file, "w", encoding="utf-8") as f:
        for ex in dev_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    tracker.save(stats_file)

    print(f"\nFinished {source_name}:")
    print(f"  Raw: {tracker.counts['raw_ingested']:,}")
    print(f"  Passed Clean: {tracker.counts['passed_clean']:,}")
    print(f"  Passed Dedup: {tracker.counts['passed_dedup']:,}")
    print(f"  Final Pool: {tracker.counts['final_accepted']:,}")
    print(f"  Built Train: {len(train_set):,} -> {train_file}")
    print(f"  Built Dev: {len(dev_set):,} -> {dev_file}")
    print(f"  Stats: {stats_file}\n")

    return tracker.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Marathi LLM Dataset Builder")
    parser.add_argument("--config", default="configs/data/s0_alpaca.yaml")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--tokenizer-name", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-decontamination", action="store_true")
    args = parser.parse_args()

    tokenizer = None
    if not args.skip_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer: {args.tokenizer_name}...")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        except Exception as e:
            print(f"Warning: could not load tokenizer {args.tokenizer_name}: {e}")

    decontaminator = None
    if not args.skip_decontamination:
        print("Initializing decontamination index...")
        decontaminator = BenchmarkDecontaminator()
        load_all_eval_targets(decontaminator)

    build_dataset_pipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        eval_decontaminator=decontaminator,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    main()
