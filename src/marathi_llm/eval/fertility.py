"""Tokenizer fertility analysis for Marathi text.

Measures:
- Tokens per word on native/human Marathi (Flores-IN dev)
- Tokens per word on translated Marathi (S0 Alpaca dev responses)
- Tokens per character
- % of full S0 training examples exceeding sequence length limits (512 tokens)
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def extract_instruction_and_response(translated_text: str) -> Tuple[str, str]:
    """Extracts Marathi instruction and response from S0 Alpaca translated_text."""
    inst = ""
    resp = ""
    if "### प्रतिसाद:" in translated_text:
        parts = translated_text.split("### प्रतिसाद:")
        resp = parts[1].strip() if len(parts) > 1 else ""
        inst_part = parts[0]
        if "### सूचना:" in inst_part:
            inst = inst_part.split("### सूचना:")[1].strip()
            if "### इनपुट:" in inst:
                inst = inst.split("### इनपुट:")[0].strip()
    return inst, resp


def load_marathi_eval_data(
    flores_path: str = None,
    s0_dev_ids_path: str = "data/splits/s0_dev_ids.json",
    s0_dataset_name: str = "rachittshah/alpaca-marahti",
) -> Dict[str, List[str]]:
    """Loads Flores Marathi sentences, S0 Marathi responses, and full S0 examples."""
    flores_sentences = []

    # 1. Flores-IN Marathi source sentences (human-written reference)
    resolved_flores_path = flores_path
    if not resolved_flores_path or not Path(resolved_flores_path).exists():
        try:
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
                    flores_sentences.append(src)

    # 2. S0 dev split: extract Marathi response and full example from translated_text
    s0_responses = []
    s0_full_examples = []
    s0_file = Path(s0_dev_ids_path)

    if s0_file.exists():
        try:
            from datasets import load_dataset
            ds = load_dataset(s0_dataset_name, split="train")
            with open(s0_file, "r", encoding="utf-8") as f:
                raw_ids = json.load(f)
                dev_ids = set(raw_ids.get("row_ids", []) if isinstance(raw_ids, dict) else raw_ids)

            for idx in sorted(dev_ids):
                if idx < len(ds):
                    raw_row = ds[idx]
                    trans_text = raw_row.get("translated_text", "")
                    inst, resp = extract_instruction_and_response(trans_text)
                    if resp:
                        s0_responses.append(resp)
                    if inst and resp:
                        full_text = f"### सूचना:\n{inst}\n\n### उत्तर:\n{resp}"
                        s0_full_examples.append(full_text)
        except Exception as e:
            print(f"Warning: could not load S0 dev responses: {e}")

    return {
        "flores": flores_sentences,
        "s0_responses": s0_responses,
        "s0_full_examples": s0_full_examples,
    }


def compute_text_fertility(tokenizer, texts: List[str]) -> Dict[str, float]:
    """Computes tokens/word and tokens/char on a set of texts."""
    total_tokens = 0
    total_words = 0
    total_chars = 0

    for t in texts:
        words = t.split()
        if not words:
            continue
        token_ids = tokenizer.encode(t, add_special_tokens=False)
        total_tokens += len(token_ids)
        total_words += len(words)
        total_chars += len(t)

    return {
        "n_samples": len(texts),
        "tokens_per_word": float(total_tokens / total_words) if total_words > 0 else 0.0,
        "tokens_per_char": float(total_tokens / total_chars) if total_chars > 0 else 0.0,
    }


def compute_truncation_rate(tokenizer, full_examples: List[str], max_len: int = 512) -> float:
    """Computes % of full training examples that exceed max_len tokens."""
    if not full_examples:
        return 0.0

    exceeds = 0
    for ex in full_examples:
        tok_len = len(tokenizer.encode(ex, add_special_tokens=False))
        if tok_len > max_len:
            exceeds += 1

    return float((exceeds / len(full_examples)) * 100.0)


def analyze_tokenizers(
    tokenizer_ids: List[str],
    data: Dict[str, List[str]],
    max_len: int = 512,
) -> Dict[str, Dict[str, Any]]:
    """Analyzes multiple tokenizers across native and translated Marathi."""
    results = {}
    for tid in tokenizer_ids:
        print(f"Analyzing tokenizer: {tid}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
            flores_res = compute_text_fertility(tokenizer, data["flores"])
            s0_res = compute_text_fertility(tokenizer, data["s0_responses"])
            all_texts = data["flores"] + data["s0_responses"]
            combined_res = compute_text_fertility(tokenizer, all_texts)
            trunc_pct = compute_truncation_rate(tokenizer, data["s0_full_examples"], max_len=max_len)

            results[tid] = {
                "flores_tokens_per_word": flores_res["tokens_per_word"],
                "s0_tokens_per_word": s0_res["tokens_per_word"],
                "combined_tokens_per_word": combined_res["tokens_per_word"],
                "combined_tokens_per_char": combined_res["tokens_per_char"],
                "pct_exceeding_512": trunc_pct,
                "n_flores": len(data["flores"]),
                "n_s0_responses": len(data["s0_responses"]),
                "n_s0_full_examples": len(data["s0_full_examples"]),
            }
        except Exception as e:
            print(f"Failed to analyze tokenizer {tid}: {e}")
    return results


def format_markdown_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Formats fertility results as a markdown table."""
    lines = [
        "| Model / Tokenizer | Flores T/W | S0-mr T/W | Combined T/W | Combined T/Char | % S0 > 512 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| `{name}` | {r['flores_tokens_per_word']:.2f} | {r['s0_tokens_per_word']:.2f} | "
            f"{r['combined_tokens_per_word']:.2f} | {r['combined_tokens_per_char']:.2f} | {r['pct_exceeding_512']:.1f}% |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze tokenizer fertility on Marathi text")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=[
            "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "sarvamai/sarvam-1",
            "google/gemma-2-2b-it",
        ],
        help="List of tokenizer IDs",
    )
    parser.add_argument("--flores", default=None)
    parser.add_argument("--s0-dev", default="data/splits/s0_dev_ids.json")
    parser.add_argument("--output", default="results/fertility.json")
    args = parser.parse_args()

    data = load_marathi_eval_data(args.flores, args.s0_dev)
    print(f"Loaded {len(data['flores'])} Flores sentences, {len(data['s0_responses'])} S0 Marathi responses, "
          f"and {len(data['s0_full_examples'])} full S0 examples.")

    results = analyze_tokenizers(args.tokenizers, data)
    print("\n" + format_markdown_table(results) + "\n")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
