"""Held-out bits-per-character (BPC) evaluation on human-written Marathi."""

import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from huggingface_hub import hf_hub_download
from marathi_llm.generate import load_model_and_tokenizer


def load_marathi_reference_texts(limit: Optional[int] = 500) -> List[str]:
    """Loads held-out human-written Marathi sentences from Flores-IN dev."""
    try:
        path = hf_hub_download(
            repo_id="google/IndicGenBench_flores_in",
            filename="flores_mr_en_dev.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [ex.get("source", "").strip() for ex in data.get("examples", [])]
        texts = [t for t in texts if t]
        if limit:
            texts = texts[:limit]
        return texts
    except Exception as e:
        print(f"Error loading Flores Marathi text: {e}")
        return []


def compute_bpc(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda:0",
) -> Dict[str, float]:
    """
    Computes Bits-Per-Character (BPC) and Bits-Per-Byte (BPB) over human-written Marathi texts.
    Formula:
      BPC = Total NLL / (ln(2) * total_chars)
      BPB = Total NLL / (ln(2) * total_utf8_bytes)
    """
    total_nll = 0.0
    total_chars = 0
    total_bytes = 0
    total_tokens = 0

    model.eval()

    for idx, text in enumerate(texts):
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt").to(device)
        n_tokens = inputs["input_ids"].shape[1]
        if n_tokens < 2:
            continue

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()  # mean NLL per predicted token

        n_predicted_tokens = n_tokens - 1
        seq_nll = loss * n_predicted_tokens

        total_nll += seq_nll
        total_tokens += n_predicted_tokens
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

        if (idx + 1) % 100 == 0:
            print(f"Evaluated BPC on {idx + 1}/{len(texts)} sentences...")

    ln2 = math.log(2.0)
    bpc = total_nll / (ln2 * total_chars) if total_chars > 0 else 0.0
    bpb = total_nll / (ln2 * total_bytes) if total_bytes > 0 else 0.0
    perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else 0.0

    return {
        "n_sentences": len(texts),
        "total_chars": total_chars,
        "total_utf8_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bits_per_character": float(bpc),
        "bits_per_byte": float(bpb),
        "token_perplexity": float(perplexity),
    }


def main():
    parser = argparse.ArgumentParser(description="Held-out BPC evaluation on Marathi")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="results/bpc.json")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)
    texts = load_marathi_reference_texts(limit=args.limit)
    print(f"Loaded {len(texts)} Marathi reference sentences.")

    results = compute_bpc(model, tokenizer, texts)
    print("\n--- BPC Evaluation Results ---")
    print(f"Bits per Character (BPC): {results['bits_per_character']:.4f}")
    print(f"Bits per Byte (BPB):      {results['bits_per_byte']:.4f}")
    print(f"Token Perplexity:         {results['token_perplexity']:.2f}")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
