"""Held-out bits-per-character (BPC) evaluation on human-written Marathi."""

import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from marathi_llm.generate import load_model_and_tokenizer
from marathi_llm.eval.stats import bootstrap_ci


def load_marathi_reference_texts(limit: Optional[int] = None) -> List[str]:
    """Loads held-out human-written Marathi sentences from Flores-IN dev (all 997 sentences by default)."""
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
        raise RuntimeError(f"Failed to load Flores Marathi reference texts: {e}") from e


def compute_bpc(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Computes Bits-Per-Character (BPC) and Bits-Per-Byte (BPB) over human-written Marathi texts.
    Prepends BOS/EOS token so the first text token is scored, ensuring fair character accounting across tokenizers.
    Convention:
      - If tokenizer.bos_token_id is present: use bos_token_id.
      - Else fallback to tokenizer.eos_token_id.
    Formula:
      BPC = Total NLL / (ln(2) * total_chars)
      BPB = Total NLL / (ln(2) * total_utf8_bytes)
    Returns summary metrics and per-sentence records for paired significance testing.
    """
    total_nll = 0.0
    total_chars = 0
    total_bytes = 0
    total_tokens = 0

    model.eval()

    # Documented BOS convention: use bos_token_id if available, else eos_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    ln2 = math.log(2.0)

    per_sentence_records = []
    sentence_bpcs = []

    for idx, text in enumerate(texts):
        if not text:
            continue

        raw_ids = tokenizer.encode(text, add_special_tokens=False)
        # Prepend BOS token if not already present, so token 0 is predicted given BOS
        if bos_id is not None and (len(raw_ids) == 0 or raw_ids[0] != bos_id):
            input_ids = [bos_id] + raw_ids
        else:
            input_ids = raw_ids

        if len(input_ids) < 2:
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs.loss.item()  # mean NLL per predicted token

        # Number of predicted tokens equals number of original text tokens (len(input_ids) - 1)
        n_predicted = len(input_ids) - 1
        seq_nll = loss * n_predicted
        char_count = len(text)
        byte_count = len(text.encode("utf-8"))

        sent_bpc = float(seq_nll / (ln2 * char_count)) if char_count > 0 else 0.0
        sent_bpb = float(seq_nll / (ln2 * byte_count)) if byte_count > 0 else 0.0

        total_nll += seq_nll
        total_tokens += n_predicted
        total_chars += char_count
        total_bytes += byte_count

        sentence_bpcs.append(sent_bpc)
        per_sentence_records.append({
            "idx": idx,
            "text": text,
            "nll": float(seq_nll),
            "chars": char_count,
            "bytes": byte_count,
            "tokens": n_predicted,
            "bpc": sent_bpc,
            "bpb": sent_bpb,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(texts):
            print(f"Evaluated BPC on {idx + 1}/{len(texts)} sentences...")

    bpc = total_nll / (ln2 * total_chars) if total_chars > 0 else 0.0
    bpb = total_nll / (ln2 * total_bytes) if total_bytes > 0 else 0.0
    perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else 0.0

    bpc_ci = bootstrap_ci(sentence_bpcs, n_bootstraps=1000)

    summary = {
        "n_sentences": len(texts),
        "total_chars": total_chars,
        "total_utf8_bytes": total_bytes,
        "total_tokens": total_tokens,
        "total_nll": float(total_nll),
        "bits_per_character": float(bpc),
        "bits_per_byte": float(bpb),
        "token_perplexity": float(perplexity),
        "bpc_sentence_ci95": [float(bpc_ci["ci_low"]), float(bpc_ci["ci_high"])],
        "bos_token_id_used": bos_id,
    }

    return {"summary": summary, "records": per_sentence_records}


def main():
    parser = argparse.ArgumentParser(description="Held-out BPC evaluation on Marathi")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/bpc.json")
    args = parser.parse_args()

    model, tokenizer, prequantized = load_model_and_tokenizer(args.model, args.adapter)
    texts = load_marathi_reference_texts(limit=args.limit)
    print(f"Loaded {len(texts)} Marathi reference sentences.")

    results = compute_bpc(model, tokenizer, texts)
    s = results["summary"]
    print("\n--- BPC Evaluation Results ---")
    print(f"Bits per Character (BPC): {s['bits_per_character']:.4f}")
    print(f"Bits per Byte (BPB):      {s['bits_per_byte']:.4f}")
    print(f"Token Perplexity:         {s['token_perplexity']:.2f}")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
