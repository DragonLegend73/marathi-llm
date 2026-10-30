"""Tokenizer fertility analysis for Marathi text.

Measures:
- Tokens per word
- Tokens per character
- % of examples exceeding sequence length limits (e.g. 512)
"""

import json
import argparse
from typing import List, Dict, Any
from pathlib import Path
from transformers import AutoTokenizer


def load_marathi_eval_sentences(
    flores_path: str = None,
    s0_dev_ids_path: str = "data/splits/s0_dev_ids.json",
    s0_dataset_name: str = "rachittshah/alpaca-marahti",
) -> List[str]:
    """Loads ~2,000 Marathi sentences from Flores-IN and S0 dev split."""
    sentences = []

    # 1. Flores-IN Marathi source sentences
    resolved_flores_path = flores_path
    if not resolved_flores_path or not Path(resolved_flores_path).exists():
        try:
            from huggingface_hub import hf_hub_download
            resolved_flores_path = hf_hub_download(
                repo_id="google/IndicGenBench_flores_in",
                filename="flores_mr_en_dev.json",
                repo_type="dataset",
            )
        except Exception as e:
            print(f"Warning: could not resolve Flores-IN from HF Hub: {e}")

    if resolved_flores_path and Path(resolved_flores_path).exists():
        with open(resolved_flores_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            examples = data.get("examples", [])
            for ex in examples:
                src = ex.get("source", "").strip()
                if src:
                    sentences.append(src)

    # 2. S0 dev split responses
    s0_file = Path(s0_dev_ids_path)
    if s0_file.exists():
        try:
            from datasets import load_dataset
            ds = load_dataset(s0_dataset_name, split="train")
            with open(s0_file, "r", encoding="utf-8") as f:
                raw_ids = json.load(f)
                if isinstance(raw_ids, dict):
                    dev_ids = set(raw_ids.get("row_ids", []))
                else:
                    dev_ids = set(raw_ids)
            for idx in dev_ids:
                if idx < len(ds):
                    resp = (ds[idx].get("output") or "").strip()
                    if resp:
                        sentences.append(resp)
        except Exception as e:
            print(f"Warning: could not load S0 dev responses: {e}")

    return sentences


def compute_fertility(tokenizer, sentences: List[str], max_len: int = 512) -> Dict[str, float]:
    """Computes fertility statistics for a given tokenizer over a list of sentences."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    exceeds_max_len = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        token_ids = tokenizer.encode(sent, add_special_tokens=False)
        n_tok = len(token_ids)
        total_tokens += n_tok
        total_words += len(words)
        total_chars += len(sent)
        if n_tok > max_len:
            exceeds_max_len += 1

    n_sents = len(sentences)
    return {
        "n_sentences": n_sents,
        "tokens_per_word": float(total_tokens / total_words) if total_words > 0 else 0.0,
        "tokens_per_char": float(total_tokens / total_chars) if total_chars > 0 else 0.0,
        "pct_exceeding_max_len": float((exceeds_max_len / n_sents) * 100.0) if n_sents > 0 else 0.0,
    }


def analyze_tokenizers(
    tokenizer_ids: List[str],
    sentences: List[str],
    max_len: int = 512,
) -> Dict[str, Dict[str, float]]:
    """Analyzes multiple tokenizers and returns a summary dict."""
    results = {}
    for tid in tokenizer_ids:
        print(f"Loading tokenizer: {tid}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
            res = compute_fertility(tokenizer, sentences, max_len=max_len)
            results[tid] = res
        except Exception as e:
            print(f"Failed to analyze tokenizer {tid}: {e}")
    return results


def format_markdown_table(results: Dict[str, Dict[str, float]]) -> str:
    """Formats fertility results as a markdown table."""
    lines = [
        "| Model / Tokenizer | Tokens / Word | Tokens / Char | % Exceeding 512 |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {r['tokens_per_word']:.2f} | {r['tokens_per_char']:.2f} | {r['pct_exceeding_max_len']:.1f}% |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze tokenizer fertility on Marathi text")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=[
            "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "meta-llama/Llama-3.2-1B-Instruct",
            "google/gemma-2-2b-it",
            "sarvamai/sarvam-1",
        ],
        help="List of tokenizer IDs",
    )
    parser.add_argument("--flores", default="data/raw/IndicGenBench_flores_in/flores_mr_en_dev.json")
    parser.add_argument("--s0-dev", default="data/splits/s0_dev_ids.json")
    parser.add_argument("--output", default="results/fertility.json")
    args = parser.parse_args()

    sentences = load_marathi_eval_sentences(args.flores, args.s0_dev)
    print(f"Loaded {len(sentences)} Marathi evaluation sentences.")

    results = analyze_tokenizers(args.tokenizers, sentences)
    print("\n" + format_markdown_table(results) + "\n")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
