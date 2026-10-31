"""XQuAD Marathi extractive question-answering evaluation."""

import re
import json
import string
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from huggingface_hub import hf_hub_download

from marathi_llm.generate import load_model_and_tokenizer, generate_batch
from marathi_llm.eval.stats import bootstrap_ci


XQUAD_PROMPT_TEMPLATE = (
    "खाली दिलेल्या परिच्छेदाच्या आधारे विचारलेल्या प्रश्नाचे संक्षिप्त उत्तर द्या.\n"
    "परिच्छेद: {context}\n"
    "प्रश्न: {question}\n"
    "उत्तर:"
)


def normalize_answer(s: str) -> str:
    """Lowercases, strips punctuation, and removes excess whitespace."""
    s = s.lower().strip()
    # Strip common punctuation
    exclude = set(string.punctuation).union({"।", "॥", "‘", "’", "“", "”", "—", "–"})
    s = "".join(ch for ch in s if ch not in exclude)
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Computes exact match (1.0 or 0.0) against candidate ground truth answers."""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if norm_pred == normalize_answer(gt):
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, ground_truths: List[str]) -> float:
    """Computes max token-level F1 score against candidate ground truth answers."""
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 1.0 if any(not normalize_answer(gt).split() for gt in ground_truths) else 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)

    return float(best_f1)


def load_xquad_data(split: str = "dev") -> List[Dict[str, Any]]:
    """Loads XQuAD Marathi data from local raw cache or Hugging Face Hub."""
    filename = f"xquad_mr_{split}.json"
    local_path = Path("data/raw/IndicGenBench_xquad_in") / filename

    if local_path.exists():
        target_path = str(local_path)
    else:
        target_path = hf_hub_download(
            repo_id="google/IndicGenBench_xquad_in",
            filename=filename,
            repo_type="dataset",
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    items = []
    for ex in examples:
        ctx = ex.get("context", "").strip()
        q = ex.get("question", "").strip()
        ans_list = [a["text"] for a in ex.get("answers", []) if "text" in a]
        items.append({
            "id": ex.get("id", str(len(items))),
            "prompt": XQUAD_PROMPT_TEMPLATE.format(context=ctx, question=q),
            "context": ctx,
            "question": q,
            "answers": ans_list,
        })

    return items


def evaluate_xquad(
    model,
    tokenizer,
    split: str = "dev",
    batch_size: int = 4,
    prompt_format: str = "chat",
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Runs XQuAD evaluation on Marathi."""
    items = load_xquad_data(split=split)
    print(f"Loaded {len(items)} questions from XQuAD {split}...")

    generations = generate_batch(
        model=model,
        tokenizer=tokenizer,
        items=items,
        batch_size=batch_size,
        max_new_tokens=32,
        prompt_format=prompt_format,
        device=device,
    )

    em_scores = []
    f1_scores = []
    records = []

    for g in generations:
        pred = g["output"]
        gts = g["answers"]
        em = compute_exact_match(pred, gts)
        f1 = compute_token_f1(pred, gts)

        em_scores.append(em)
        f1_scores.append(f1)
        records.append({
            "id": g["id"],
            "question": g["question"],
            "prediction": pred,
            "answers": gts,
            "em": em,
            "f1": f1,
        })

    em_ci = bootstrap_ci(em_scores)
    f1_ci = bootstrap_ci(f1_scores)

    summary = {
        "benchmark": "XQuAD-IN",
        "split": split,
        "n_samples": len(items),
        "exact_match": float(np.mean(em_scores) * 100.0),
        "em_ci_95": [float(em_ci["ci_low"] * 100.0), float(em_ci["ci_high"] * 100.0)],
        "f1": float(np.mean(f1_scores) * 100.0),
        "f1_ci_95": [float(f1_ci["ci_low"] * 100.0), float(f1_ci["ci_high"] * 100.0)],
    }

    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="XQuAD Marathi QA evaluation")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-format", default="chat", choices=["chat", "alpaca_v0", "raw"])
    parser.add_argument("--output-dir", default="results/xquad")
    args = parser.parse_args()

    model, tokenizer, prequantized = load_model_and_tokenizer(args.model, args.adapter)
    results = evaluate_xquad(
        model,
        tokenizer,
        split=args.split,
        batch_size=args.batch_size,
        prompt_format=args.prompt_format,
    )

    s = results["summary"]
    print("\n--- XQuAD Evaluation Summary ---")
    print(f"Exact Match (EM): {s['exact_match']:.2f}% (95% CI: [{s['em_ci_95'][0]:.2f}%, {s['em_ci_95'][1]:.2f}%])")
    print(f"F1 Score:         {s['f1']:.2f}% (95% CI: [{s['f1_ci_95'][0]:.2f}%, {s['f1_ci_95'][1]:.2f}%])")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"xquad_{args.split}_summary.json", "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    with open(out_dir / f"xquad_{args.split}_records.json", "w", encoding="utf-8") as f:
        json.dump(results["records"], f, indent=2)


if __name__ == "__main__":
    main()
