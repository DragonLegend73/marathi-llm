"""E1 Sequence-Length Census (plan_2.md Section 2.5).

Streams S0-S3 through load -> normalize -> quality filters (skipping dedup/LSH
and decontamination for bounded RAM), measures exact chat-template training
lengths (prompt + response + EOS), saves per-source length arrays and summary
tables, builds worst-case 0.9*cap..cap pools for E2 GPU profiling, and computes
kept-% coverage across Stage A tokenizers on a 2,000-row sample per source.
"""

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from transformers import AutoTokenizer

from marathi_llm.data.clean import normalize_marathi_text, validate_example
from marathi_llm.data.load import (
    stream_s0_alpaca,
    stream_s1_indicalign,
    stream_s2_sandlogic,
    stream_s3_aya,
)
from marathi_llm.progress import get_progress_bar
from marathi_llm.train import build_prompt_completion_records, format_record_for_length_check


CAPS = [384, 512, 768, 1024, 1536, 2048]
PROFILE_CAPS = [512, 768, 1024, 1536, 2048]

SOURCE_CONFIGS = {
    "s0": "configs/data/s0_alpaca.yaml",
    "s1": "configs/data/s1_indicalign.yaml",
    "s2": "configs/data/s2_sandlogic.yaml",
    "s3": "configs/data/s3_aya.yaml",
}

DEFAULT_STAGE_A_TOKENIZERS = [
    ("qwen25_1p5b", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", None),
    ("qwen3_1p7b", "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", False),
    ("gemma3_1b", "unsloth/gemma-3-1b-it-unsloth-bnb-4bit", None),
    ("llama32_1b", "unsloth/Llama-3.2-1B-Instruct-bnb-4bit", None),
    ("sarvam1", "sarvamai/sarvam-1", None),
]


def summarize_lengths(lengths: List[int], caps: List[int] = CAPS) -> Dict[str, Any]:
    """Computes percentiles and per-cap kept-% / mean-token stats from a list of lengths."""
    if not lengths:
        return {
            "n": 0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "mean_all": 0.0,
            "max": 0,
            "by_cap": {},
        }
    arr = np.array(lengths, dtype=np.int32)
    by_cap: Dict[str, Any] = {}
    for c in caps:
        mask = arr <= c
        kept = arr[mask]
        by_cap[str(c)] = {
            "kept_count": int(kept.size),
            "kept_pct": round(float(kept.size / arr.size * 100.0), 2),
            "mean_tokens_kept": round(float(kept.mean()), 2) if kept.size > 0 else 0.0,
        }
    return {
        "n": int(arr.size),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p90": round(float(np.percentile(arr, 90)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
        "mean_all": round(float(arr.mean()), 2),
        "max": int(arr.max()),
        "by_cap": by_cap,
    }


def open_source_stream(source_key: str, cfg: Dict[str, Any]):
    """Opens the configured raw row stream for s0/s1/s2/s3."""
    if source_key == "s0":
        return stream_s0_alpaca()
    if source_key == "s1":
        return stream_s1_indicalign(
            configs=cfg.get("configs"),
            per_config_limit=cfg.get("per_config_limit"),
        )
    if source_key == "s2":
        return stream_s2_sandlogic(
            split=cfg.get("split", "marathi"),
            limit=cfg.get("sample_limit", 100000),
        )
    if source_key == "s3":
        return stream_s3_aya()
    raise ValueError(f"Unknown source key: {source_key}")


def measure_chat_token_length(
    inst: str,
    inp: str,
    resp: str,
    tokenizer: Any,
    has_chat_template: bool,
    enable_thinking: Optional[bool] = None,
) -> int:
    """Computes exact token count in training chat-template format (prompt + completion + EOS)."""
    recs = build_prompt_completion_records(
        [{"instruction": inst, "input": inp, "response": resp}],
        has_chat_template=has_chat_template,
    )
    if not recs:
        return 0
    full_text = format_record_for_length_check(
        recs[0], tokenizer, enable_thinking=enable_thinking
    )
    return len(tokenizer.encode(full_text, add_special_tokens=False))


def run_source_census(
    source_key: str,
    tokenizer: Any,
    worst_case_pools: Dict[int, List[Dict[str, Any]]],
    worst_case_target: int = 200,
    sample_size: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs quality-filtered length census for one source without dedup/LSH."""
    cfg_path = SOURCE_CONFIGS[source_key]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clean_cfg = cfg.get("clean", {})
    min_purity = clean_cfg.get("min_devanagari_purity", 60.0)
    max_latin = clean_cfg.get("max_latin_ratio", 0.15)
    min_resp_chars = clean_cfg.get("min_response_chars", 10)
    require_mr_lang = clean_cfg.get("require_mr_lang_id", False)

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    stream = open_source_stream(source_key, cfg)

    rng = random.Random(seed)
    lengths: List[int] = []
    subconfig_lengths: Dict[str, List[int]] = collections.defaultdict(list)
    reservoir: List[Dict[str, Any]] = []
    raw_count = 0
    passed_count = 0

    est_total = cfg.get("sample_limit") or (52000 if source_key == "s0" else 3600)
    pbar = get_progress_bar(total=est_total, desc=f"Census {source_key.upper()}", unit="row")

    for raw_row in stream:
        raw_count += 1
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
            tokenizer=None,  # Skip length filter so we measure full length distribution
        )
        if not is_valid:
            continue

        tok_len = measure_chat_token_length(inst, inp, resp, tokenizer, has_chat_template)
        if tok_len <= 0:
            continue

        passed_count += 1
        lengths.append(tok_len)
        sub_tag = raw_row.get("source", cfg["name"])
        subconfig_lengths[sub_tag].append(tok_len)

        row_payload = {
            "id": raw_row.get("id", f"{source_key}_{passed_count}"),
            "instruction": inst,
            "input": inp,
            "response": resp,
            "source": sub_tag,
            "qwen25_chat_tokens": tok_len,
        }

        # Reservoir sample for cross-tokenizer Stage A coverage
        if len(reservoir) < sample_size:
            reservoir.append(row_payload)
        else:
            j = rng.randint(0, passed_count - 1)
            if j < sample_size:
                reservoir[j] = row_payload

        # Collect worst-case rows (0.9 * cap <= tok_len <= cap) for E2 GPU profiling
        for cap in PROFILE_CAPS:
            lo = int(0.9 * cap)
            if lo <= tok_len <= cap and len(worst_case_pools[cap]) < worst_case_target:
                worst_case_pools[cap].append(row_payload)

    pbar.close()

    summary = summarize_lengths(lengths)
    subconfig_summary = {
        tag: summarize_lengths(tag_lens, caps=[512, 768, 1024, 1536, 2048])
        for tag, tag_lens in sorted(subconfig_lengths.items())
    }

    result = {
        "source": cfg["name"],
        "source_key": source_key,
        "raw_ingested": raw_count,
        "passed_other_filters": passed_count,
        "summary": summary,
        "by_subconfig": subconfig_summary,
        "lengths": lengths,
    }
    return result, reservoir


def evaluate_stage_a_tokenizers(
    samples_by_source: Dict[str, List[Dict[str, Any]]],
    tokenizer_specs: List[Tuple[str, str, Optional[bool]]],
) -> Dict[str, Any]:
    """Measures kept % at 512 / 768 / 1024 under each Stage A tokenizer."""
    coverage: Dict[str, Any] = {}
    for name, hf_id, enable_thinking in tokenizer_specs:
        print(f"\n[Stage A Tokenizer Coverage] Loading {name} ({hf_id})...")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: skipping {name} ({hf_id}): {e}")
            coverage[name] = {"hf_id": hf_id, "error": str(e)}
            continue

        has_chat = bool(getattr(tok, "chat_template", None))
        per_source: Dict[str, Any] = {}
        for src_key, rows in samples_by_source.items():
            lens = [
                measure_chat_token_length(
                    r["instruction"],
                    r["input"],
                    r["response"],
                    tok,
                    has_chat,
                    enable_thinking=enable_thinking,
                )
                for r in rows
            ]
            per_source[src_key] = summarize_lengths(lens, caps=[512, 768, 1024])
        coverage[name] = {
            "hf_id": hf_id,
            "has_chat_template": has_chat,
            "by_source": per_source,
        }
    return coverage


def format_summary_markdown(
    source_results: Dict[str, Dict[str, Any]],
    stage_a_coverage: Dict[str, Any],
) -> str:
    """Formats clean Markdown tables for E1 length census."""
    lines: List[str] = ["# E1 Sequence-Length Census (Qwen2.5 Chat Template)\n"]

    lines.append("## 1. All Sources × Caps (Qwen2.5-1.5B Chat Template + EOS)\n")
    lines.append(
        "| Source | Pass Clean | p50 | p90 | p99 | ≤384 | ≤512 | ≤768 | ≤1024 | ≤1536 | ≤2048 | Mean@512 | Mean@768 | Mean@1024 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, res in source_results.items():
        s = res["summary"]
        bc = s["by_cap"]
        lines.append(
            f"| **{res['source']}** | {s['n']:,} | {s['p50']:.0f} | {s['p90']:.0f} | {s['p99']:.0f} "
            f"| {bc['384']['kept_pct']:.1f}% | {bc['512']['kept_pct']:.1f}% | {bc['768']['kept_pct']:.1f}% "
            f"| {bc['1024']['kept_pct']:.1f}% | {bc['1536']['kept_pct']:.1f}% | {bc['2048']['kept_pct']:.1f}% "
            f"| {bc['512']['mean_tokens_kept']:.1f} | {bc['768']['mean_tokens_kept']:.1f} | {bc['1024']['mean_tokens_kept']:.1f} |"
        )

    if "s1" in source_results and source_results["s1"]["by_subconfig"]:
        lines.append("\n## 2. S1 IndicAlign Kept % by Sub-Config\n")
        lines.append("| Sub-Config | Pass Clean | p50 | p90 | ≤512 | ≤768 | ≤1024 | ≤1536 | ≤2048 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tag, sub in source_results["s1"]["by_subconfig"].items():
            bc = sub["by_cap"]
            lines.append(
                f"| `{tag}` | {sub['n']:,} | {sub['p50']:.0f} | {sub['p90']:.0f} "
                f"| {bc['512']['kept_pct']:.1f}% | {bc['768']['kept_pct']:.1f}% "
                f"| {bc['1024']['kept_pct']:.1f}% | {bc['1536']['kept_pct']:.1f}% | {bc['2048']['kept_pct']:.1f}% |"
            )

    if stage_a_coverage:
        lines.append("\n## 3. Stage A Tokenizer Coverage at ≤512 / ≤768 / ≤1024 (2,000-row Sample per Source)\n")
        src_keys = list(source_results.keys())
        header = "| Model / Tokenizer | " + " | ".join(f"{k.upper()} (≤512 / ≤768 / ≤1024)" for k in src_keys) + " |"
        sep = "|---|" + "|".join(["---:"] * len(src_keys)) + "|"
        lines.append(header)
        lines.append(sep)
        for m_name, cov in stage_a_coverage.items():
            if "error" in cov:
                lines.append(f"| `{m_name}` | " + " | ".join(["*(failed)*"] * len(src_keys)) + " |")
                continue
            cells = []
            for k in src_keys:
                if k in cov["by_source"]:
                    bc = cov["by_source"][k]["by_cap"]
                    cells.append(
                        f"{bc['512']['kept_pct']:.1f}% / {bc['768']['kept_pct']:.1f}% / {bc['1024']['kept_pct']:.1f}%"
                    )
                else:
                    cells.append("-")
            lines.append(f"| `{m_name}` | " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="E1 Sequence-Length Census (plan_2.md §2.5)")
    parser.add_argument("--sources", nargs="+", default=["s0", "s1", "s2", "s3"])
    parser.add_argument("--tokenizer", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--out", default="results/length_census")
    parser.add_argument("--worst-case-target", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--skip-stage-a-tokenizers", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pools_dir = out_dir / "worst_case_pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading primary tokenizer: {args.tokenizer}...")
    primary_tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    worst_case_pools: Dict[int, List[Dict[str, Any]]] = {c: [] for c in PROFILE_CAPS}
    source_results: Dict[str, Dict[str, Any]] = {}
    samples_by_source: Dict[str, List[Dict[str, Any]]] = {}

    for src_key in args.sources:
        res, sample = run_source_census(
            src_key,
            primary_tok,
            worst_case_pools=worst_case_pools,
            worst_case_target=args.worst_case_target,
            sample_size=args.sample_size,
        )
        source_results[src_key] = res
        samples_by_source[src_key] = sample

        src_out = out_dir / f"{res['source']}.json"
        with open(src_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {src_out} (n={res['summary']['n']:,}, p50={res['summary']['p50']}, p90={res['summary']['p90']})")

    # Save worst-case pools for E2 GPU profiling
    for cap, rows in worst_case_pools.items():
        pool_path = pools_dir / f"worst_case_{cap}.jsonl"
        with open(pool_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved worst-case pool for cap={cap}: {len(rows)} rows -> {pool_path}")

    # Stage A tokenizer coverage
    stage_a_coverage: Dict[str, Any] = {}
    if not args.skip_stage_a_tokenizers:
        stage_a_coverage = evaluate_stage_a_tokenizers(samples_by_source, DEFAULT_STAGE_A_TOKENIZERS)
        with open(out_dir / "stage_a_tokenizer_coverage.json", "w", encoding="utf-8") as f:
            json.dump(stage_a_coverage, f, indent=2)

    md = format_summary_markdown(source_results, stage_a_coverage)
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + md)
    print(f"Full E1 summary written to {summary_path}")


if __name__ == "__main__":
    main()
