"""Flores-IN translation evaluation using chrF++ (word_order=2)."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import sacrebleu
from huggingface_hub import hf_hub_download

from marathi_llm.generate import load_model_and_tokenizer, generate_batch
from marathi_llm.eval.stats import bootstrap_ci, bootstrap_corpus_chrf
from marathi_llm.eval.auto_metrics import compute_script_purity


EN_MR_PROMPT = "खालील इंग्रजी मजकुराचे मराठीत भाषांतर करा.\nइंग्रजी: {source}\nमराठी:"
MR_EN_PROMPT = "Translate the following Marathi text to English.\nMarathi: {source}\nEnglish:"


def load_flores_data(direction: str = "en_mr", split: str = "dev") -> List[Dict[str, Any]]:
    """Loads Flores-IN direction JSON from cache or Hugging Face Hub."""
    filename = f"flores_{direction}_{split}.json"
    local_path = Path("data/raw/IndicGenBench_flores_in") / filename

    if local_path.exists():
        target_path = str(local_path)
    else:
        target_path = hf_hub_download(
            repo_id="google/IndicGenBench_flores_in",
            filename=filename,
            repo_type="dataset",
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", [])
    items = []
    template = EN_MR_PROMPT if direction == "en_mr" else MR_EN_PROMPT

    for idx, ex in enumerate(examples):
        src = ex.get("source", "").strip()
        tgt = ex.get("target", "").strip()
        items.append({
            "id": f"flores_{direction}_{split}_{idx}",
            "prompt": template.format(source=src),
            "source": src,
            "target": tgt,
            "direction": direction,
            "split": split,
        })

    return items


def evaluate_flores_direction(
    model,
    tokenizer,
    direction: str = "en_mr",
    split: str = "dev",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    prompt_format: str = "chat",
    limit: Optional[int] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Evaluates a single Flores-IN direction."""
    items = load_flores_data(direction=direction, split=split)
    if limit is not None:
        items = items[:limit]
    print(f"Evaluating Flores-IN ({direction}, {split}, n={len(items)})...")

    generations = generate_batch(
        model=model,
        tokenizer=tokenizer,
        items=items,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        prompt_format=prompt_format,
        device=device,
    )

    hypotheses = [g["output"] for g in generations]
    references = [g["target"] for g in generations]

    # Sentence-level chrF++ scores
    sentence_scores = []
    purities = []
    for h, r in zip(hypotheses, references):
        score = sacrebleu.sentence_chrf(h, [r], word_order=2).score
        sentence_scores.append(score)
        if direction == "en_mr":
            purities.append(compute_script_purity(h))

    # Bootstrap corpus chrF++ directly
    chrf_res = bootstrap_corpus_chrf(hypotheses, references)
    sentence_ci = bootstrap_ci(sentence_scores)

    summary = {
        "benchmark": "Flores-IN",
        "direction": direction,
        "split": split,
        "n_samples": len(items),
        "corpus_chrf_pp": float(chrf_res["corpus_chrf"]),
        "ci_95": [float(chrf_res["ci_low"]), float(chrf_res["ci_high"])],
        "mean_sentence_chrf_pp": float(sentence_ci["mean"]),
        "sentence_ci_95": [float(sentence_ci["ci_low"]), float(sentence_ci["ci_high"])],
    }

    if purities:
        summary["mean_devanagari_purity"] = float(np.mean(purities))

    records = []
    for g, score in zip(generations, sentence_scores):
        records.append({
            "id": g["id"],
            "source": g["source"],
            "target": g["target"],
            "output": g["output"],
            "generation": g["output"],
            "finish_reason": g["finish_reason"],
            "n_new_tokens": g["n_new_tokens"],
            "chrf_pp": float(score),
        })

    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="Flores-IN translation evaluation")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--direction", default="both", choices=["en_mr", "mr_en", "both"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-format", default="chat", choices=["chat", "alpaca_v0", "raw"])
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    parser.add_argument("--output-dir", default="results/flores")
    args = parser.parse_args()

    model, tokenizer, prequantized = load_model_and_tokenizer(args.model, args.adapter)
    directions = ["en_mr", "mr_en"] if args.direction == "both" else [args.direction]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in directions:
        res = evaluate_flores_direction(
            model=model,
            tokenizer=tokenizer,
            direction=d,
            split=args.split,
            batch_size=args.batch_size,
            prompt_format=args.prompt_format,
            limit=args.limit,
        )
        s = res["summary"]
        print(f"\n[{d.upper()}] Corpus chrF++: {s['corpus_chrf_pp']:.2f} | Mean: {s['mean_sentence_chrf_pp']:.2f} (95% CI: [{s['ci_95'][0]:.2f}, {s['ci_95'][1]:.2f}])")

        with open(out_dir / f"flores_{d}_{args.split}_summary.json", "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
        with open(out_dir / f"flores_{d}_{args.split}_records.json", "w", encoding="utf-8") as f:
            json.dump(res["records"], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
