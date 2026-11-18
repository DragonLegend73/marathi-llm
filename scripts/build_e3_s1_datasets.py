"""Builds paired 6k S1 IndicAlign datasets for E3 (cap=512 vs cap=768) in one pass.

Uses identical cleaning, deduplication, and 52,149-target benchmark decontamination,
measuring exact sequence length in Qwen2.5 chat-template format (prompt + response + EOS).
"""

import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml
from transformers import AutoTokenizer

from marathi_llm.data.clean import normalize_marathi_text, validate_example
from marathi_llm.data.decontaminate import BenchmarkDecontaminator, load_all_eval_targets
from marathi_llm.data.dedup import DatasetDeduplicator
from marathi_llm.data.load import stream_s1_indicalign
from marathi_llm.progress import get_progress_bar
from marathi_llm.train import build_prompt_completion_records, format_record_for_length_check


def measure_chat_len(inst: str, inp: str, resp: str, tokenizer: Any) -> int:
    recs = build_prompt_completion_records(
        [{"instruction": inst, "input": inp, "response": resp}],
        has_chat_template=True,
    )
    if not recs:
        return 0
    full = format_record_for_length_check(recs[0], tokenizer)
    return len(tokenizer.encode(full, add_special_tokens=False))


def main():
    with open("configs/data/s1_indicalign.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tok_id = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    print(f"Loading tokenizer: {tok_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    print("Initializing decontamination index...")
    decontaminator = BenchmarkDecontaminator()
    load_all_eval_targets(decontaminator)

    deduplicator = DatasetDeduplicator(
        threshold=cfg.get("dedup", {}).get("minhash_threshold", 0.8),
        shingle_size=cfg.get("dedup", {}).get("ngram_size", 5),
    )

    clean_cfg = cfg.get("clean", {})
    min_purity = clean_cfg.get("min_devanagari_purity", 60.0)
    max_latin = clean_cfg.get("max_latin_ratio", 0.15)
    min_resp_chars = clean_cfg.get("min_response_chars", 10)
    require_mr_lang = clean_cfg.get("require_mr_lang_id", True)

    stream = stream_s1_indicalign(
        configs=cfg.get("configs"),
        per_config_limit=cfg.get("per_config_limit", 15000),
    )

    pool_512: List[Dict[str, Any]] = []
    pool_768: List[Dict[str, Any]] = []
    counts = collections.Counter()

    pbar = get_progress_bar(total=cfg.get("sample_limit", 105000), desc="E3 Build S1", unit="row")

    for raw_row in stream:
        counts["raw_ingested"] += 1
        pbar.update(1)

        inst = normalize_marathi_text(raw_row.get("instruction", ""))
        inp = normalize_marathi_text(raw_row.get("input", ""))
        resp = normalize_marathi_text(raw_row.get("response", ""))

        is_valid, _ = validate_example(
            instruction=inst,
            response=resp,
            input_text=inp,
            min_devanagari_purity=min_purity,
            max_latin_ratio=max_latin,
            min_response_chars=min_resp_chars,
            require_mr_lang_id=require_mr_lang,
            tokenizer=None,
        )
        if not is_valid:
            counts["failed_quality"] += 1
            continue

        tok_len = measure_chat_len(inst, inp, resp, tokenizer)
        if tok_len <= 0 or tok_len > 768:
            counts["exceeds_768"] += 1
            continue

        counts["passed_clean_768"] += 1

        item_id = raw_row.get("id", f"s1_{counts['passed_clean_768']}")
        if deduplicator.is_duplicate(f"{inst} {resp}", key_id=item_id):
            counts["failed_dedup"] += 1
            continue

        is_contam, _ = decontaminator.check_contamination(f"{inst} {resp}")
        if is_contam:
            counts["failed_decontam"] += 1
            continue

        record = {
            "id": item_id,
            "instruction": inst,
            "input": inp,
            "response": resp,
            "source": raw_row.get("source", "s1_indicalign"),
            "chat_tokens": tok_len,
        }
        pool_768.append(record)
        if tok_len <= 512:
            pool_512.append(record)

    pbar.close()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = 42

    stats: Dict[str, Any] = {
        "counts": dict(counts),
        "pool_512_size": len(pool_512),
        "pool_768_size": len(pool_768),
        "arms": {},
    }

    for cap, pool in [(512, pool_512), (768, pool_768)]:
        shuffled = list(pool)
        random.Random(seed).shuffle(shuffled)
        dev_set = shuffled[:500]
        train_set = shuffled[500:6500]

        train_path = out_dir / f"e3_s1_cap{cap}_6k.jsonl"
        dev_path = out_dir / f"e3_s1_cap{cap}_dev500.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for ex in train_set:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with open(dev_path, "w", encoding="utf-8") as f:
            for ex in dev_set:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        sub_dist = dict(collections.Counter(ex["source"] for ex in train_set))
        mean_tok = sum(ex["chat_tokens"] for ex in train_set) / len(train_set)
        gt_512 = sum(1 for ex in train_set if ex["chat_tokens"] > 512)

        stats["arms"][str(cap)] = {
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "n_train": len(train_set),
            "n_dev": len(dev_set),
            "mean_chat_tokens": round(mean_tok, 2),
            "rows_gt_512": gt_512,
            "pct_rows_gt_512": round(gt_512 / len(train_set) * 100.0, 2),
            "by_subconfig": dict(sorted(sub_dist.items())),
        }
        print(
            f"[E3 S1 cap={cap}] train={len(train_set)}, dev={len(dev_set)}, "
            f"mean_tok={mean_tok:.1f}, rows>512={gt_512} ({gt_512/len(train_set)*100:.1f}%)"
        )
        print(f"  Sub-configs: {dict(sorted(sub_dist.items()))}")

    stats_path = Path("results/length_census/e3_s1_data_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved E3 dataset stats -> {stats_path}")


if __name__ == "__main__":
    main()
