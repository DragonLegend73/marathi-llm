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
from marathi_llm.data.clean import (
    count_training_tokens,
    normalize_marathi_text,
    validate_example,
)
from marathi_llm.data.dedup import DatasetDeduplicator
from marathi_llm.data.decontaminate import BenchmarkDecontaminator, load_all_eval_targets
from marathi_llm.data.stats import FunnelTracker


def build_dataset_pipeline(
    config_path: str,
    output_dir: str = "data/processed",
    eval_decontaminator: Optional[BenchmarkDecontaminator] = None,
    tokenizer=None,
    output_name: Optional[str] = None,
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

    tracker = FunnelTracker(source_name, pipeline_version="2.2")
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
        stream = stream_s1_indicalign(
            configs=cfg.get("configs"),
            per_config_limit=cfg.get("per_config_limit"),
        )
    elif source_id == "S2" or "sandlogic" in source_name:
        stream = stream_s2_sandlogic(
            split=cfg.get("split", "marathi"),
            limit=cfg.get("sample_limit", 200000),
        )
    elif source_id == "S3" or "aya" in source_name:
        stream = stream_s3_aya()
    else:
        raise ValueError(f"Unknown source: {source_name}")

    # Load frozen S3 held-out source_ids if present so exact held-out rows never enter S3 train/dev
    frozen_s3_ids = set()
    if source_id == "S3":
        for eval_p in (Path("data/eval/native_prompts.jsonl"), Path("data/eval/bpc_responses.jsonl")):
            if eval_p.exists():
                with open(eval_p, "r", encoding="utf-8") as ef:
                    for line in ef:
                        if line.strip():
                            obj = json.loads(line)
                            if obj.get("source_id"):
                                frozen_s3_ids.add(obj["source_id"])
        if frozen_s3_ids:
            print(f"Loaded {len(frozen_s3_ids)} frozen S3 held-out IDs from data/eval/ to exclude.")

    accepted_pool = []
    pbar = get_progress_bar(total=cfg.get("sample_limit") or 52000, desc=f"Processing {source_id}", unit="row")

    for raw_row in stream:
        tracker.counts["raw_ingested"] += 1
        pbar.update(1)

        if frozen_s3_ids and raw_row.get("id") in frozen_s3_ids:
            tracker.log_rejection("decontamination", "frozen_aya_heldout_id")
            continue

        # Normalize text
        inst = normalize_marathi_text(raw_row.get("instruction", ""))
        inp = normalize_marathi_text(raw_row.get("input", ""))
        resp = normalize_marathi_text(raw_row.get("response", ""))

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

    # Deterministic shuffle of clean, deduplicated, decontaminated pool
    budget_cfg = cfg.get("budget", {})
    seed = budget_cfg.get("seed", 42)
    random.seed(seed)
    random.shuffle(accepted_pool)

    # Carve out S3 held-outs ONLY if they do not already exist (keep held-out eval sets frozen)
    native_eval_prompts = []
    bpc_eval_responses = []
    heldout_cfg = cfg.get("eval_heldout", {})
    eval_dir = Path("data/eval")
    native_eval_file = eval_dir / "native_prompts.jsonl"
    bpc_eval_file = eval_dir / "bpc_responses.jsonl"

    if source_id == "S3" and heldout_cfg:
        if native_eval_file.exists() and bpc_eval_file.exists():
            print(
                "Keeping existing frozen held-out sets in data/eval/ "
                "(matching S3 rows were already removed during decontamination)."
            )
        else:
            n_native = min(heldout_cfg.get("native_prompts_count", 100), len(accepted_pool))
            for i in range(n_native):
                ex = accepted_pool[i]
                native_eval_prompts.append({
                    "id": f"aya_eval_{i}",
                    "prompt": ex["instruction"],
                    "reference": ex["response"],
                    "source_id": ex["id"],
                })
            accepted_pool = accepted_pool[n_native:]

            n_bpc = min(heldout_cfg.get("bpc_responses_count", 500), len(accepted_pool))
            for i in range(n_bpc):
                ex = accepted_pool[i]
                bpc_eval_responses.append({
                    "id": f"aya_bpc_{i}",
                    "response": ex["response"],
                    "source_id": ex["id"],
                })
            accepted_pool = accepted_pool[n_bpc:]

    if native_eval_prompts:
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(native_eval_file, "w", encoding="utf-8") as f:
            for p in native_eval_prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved {len(native_eval_prompts)} held-out native evaluation prompts to {native_eval_file}")

    if bpc_eval_responses:
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(bpc_eval_file, "w", encoding="utf-8") as f:
            for r in bpc_eval_responses:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(bpc_eval_responses)} held-out BPC evaluation responses to {bpc_eval_file}")

    # Sample Fixed Budget: 12k train, 500 dev (uniform sampling)
    dev_count = min(budget_cfg.get("dev_samples", 500), len(accepted_pool) // 10)
    train_target = budget_cfg.get("train_samples", 12000)

    dev_set = accepted_pool[:dev_count]
    remaining = accepted_pool[dev_count:]

    if train_target and len(remaining) >= train_target:
        train_set = remaining[:train_target]
    else:
        train_set = remaining

    # Record length-counter metadata, token budgets, and per-subconfig example + token shares
    sub_counts: Dict[str, int] = {}
    sub_tokens: Dict[str, int] = {}
    total_train_tokens = 0
    for ex in train_set:
        src_tag = ex.get("source", source_name)
        sub_counts[src_tag] = sub_counts.get(src_tag, 0) + 1
        if tokenizer is not None:
            n_tok = count_training_tokens(
                ex["instruction"], ex["response"], tokenizer, input_text=ex.get("input", "")
            )
            total_train_tokens += n_tok
            sub_tokens[src_tag] = sub_tokens.get(src_tag, 0) + n_tok

    n_train = len(train_set)
    by_subconfig: Dict[str, Any] = {}
    for tag in sorted(sub_counts.keys()):
        cnt = sub_counts[tag]
        entry: Dict[str, Any] = {
            "count": cnt,
            "example_share_pct": round(cnt / n_train * 100.0, 2) if n_train > 0 else 0.0,
        }
        if total_train_tokens > 0:
            toks = sub_tokens.get(tag, 0)
            entry["tokens"] = toks
            entry["token_share_pct"] = round(toks / total_train_tokens * 100.0, 2)
            entry["mean_tokens"] = round(toks / cnt, 2) if cnt > 0 else 0.0
        by_subconfig[tag] = entry

    tracker.metadata = {
        "length_counter": "chat_template" if (tokenizer is not None and getattr(tokenizer, "chat_template", None)) else "plain",
        "max_seq_len": max_seq_len,
        "sampling": "uniform",
        "seed": seed,
        "n_train": n_train,
        "n_dev": len(dev_set),
        "total_tokens_train": total_train_tokens if tokenizer is not None else None,
        "mean_tokens_train": round(total_train_tokens / n_train, 2) if (tokenizer is not None and n_train > 0) else None,
        "by_subconfig": by_subconfig,
    }

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    prefix = output_name if output_name else f"{source_name}_12k"
    dev_prefix = f"{output_name}_dev" if output_name else f"{source_name}_dev500"
    stats_prefix = output_name if output_name else source_name

    train_file = out_p / f"{prefix}.jsonl"
    dev_file = out_p / f"{dev_prefix}.jsonl"
    stats_file = out_p / f"{stats_prefix}.stats.json"

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
    parser.add_argument("--output-name", default=None, help="Custom prefix for output train/dev/stats files")
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
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
