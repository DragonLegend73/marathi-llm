"""MILU Marathi multiple-choice evaluation via next-token log-likelihood scoring."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from marathi_llm.eval.stats import bootstrap_ci


OPTION_MAP = {
    "option1": "A",
    "option2": "B",
    "option3": "C",
    "option4": "D",
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
}

CHOICE_LETTERS = ["A", "B", "C", "D"]


def format_milu_prompt(item: Dict[str, Any]) -> str:
    """Formats a MILU question into the standardized Marathi MCQ prompt."""
    q = item.get("question", "").strip()
    opt1 = item.get("option1", "").strip()
    opt2 = item.get("option2", "").strip()
    opt3 = item.get("option3", "").strip()
    opt4 = item.get("option4", "").strip()

    return (
        f"खालील प्रश्नाचे अचूक उत्तर निवडा.\n"
        f"प्रश्न: {q}\n"
        f"A. {opt1}\n"
        f"B. {opt2}\n"
        f"C. {opt3}\n"
        f"D. {opt4}\n"
        f"उत्तर:"
    )


def get_choice_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Retrieves token IDs for choice letters A-D, both with and without leading space."""
    token_ids = {}
    for letter in CHOICE_LETTERS:
        id_nospace = tokenizer.encode(letter, add_special_tokens=False)[-1]
        id_space = tokenizer.encode(" " + letter, add_special_tokens=False)[-1]
        token_ids[letter] = list(set([id_nospace, id_space]))
    return token_ids


def evaluate_milu(
    model,
    tokenizer,
    dataset_name: str = "ai4bharat/MILU",
    split: str = "validation",
    batch_size: int = 2,
    use_chat_template: bool = True,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Evaluates a model on MILU Marathi split using next-token logits over A-D."""
    print(f"Loading {dataset_name} (Marathi, split={split})...")
    ds = load_dataset(dataset_name, "Marathi", split=split)
    print(f"Total questions in {split}: {len(ds)}")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f"Option token IDs: {choice_token_ids}")

    records = []
    correct_list = []
    predicted_letters = []

    model.eval()

    # Process in batches
    n_items = len(ds)
    for start_idx in range(0, n_items, batch_size):
        batch_slice = [ds[i] for i in range(start_idx, min(start_idx + batch_size, n_items))]
        prompts = []

        for item in batch_slice:
            raw_prompt = format_milu_prompt(item)
            if use_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": raw_prompt}]
                chat_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(chat_prompt)
            else:
                prompts.append(raw_prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            # Memory optimization: project only the final position hidden state through lm_head
            if hasattr(model, "model") and hasattr(model, "lm_head"):
                hidden = model.model(**inputs).last_hidden_state[:, -1:, :]
                final_logits = model.lm_head(hidden)[:, 0, :]
            else:
                outputs = model(**inputs)
                final_logits = outputs.logits[:, -1, :]

        for i, item in enumerate(batch_slice):
            sample_logits = final_logits[i]
            letter_scores = {}
            for letter in CHOICE_LETTERS:
                tids = choice_token_ids[letter]
                # Take max logit across candidate token IDs
                letter_scores[letter] = float(max(sample_logits[tid].item() for tid in tids))

            predicted = max(CHOICE_LETTERS, key=lambda c: letter_scores[c])
            target_raw = str(item.get("target", "")).strip()
            target = OPTION_MAP.get(target_raw, target_raw.upper())
            is_correct = bool(predicted == target)

            correct_list.append(1.0 if is_correct else 0.0)
            predicted_letters.append(predicted)

            records.append({
                "idx": start_idx + i,
                "domain": item.get("domain", "Unknown"),
                "subject": item.get("subject", "Unknown"),
                "target": target,
                "predicted": predicted,
                "correct": is_correct,
                "scores": letter_scores,
            })

        if (start_idx // batch_size) % 20 == 0:
            print(f"Processed {len(records)}/{n_items} questions...")

    # Calculate statistics
    overall_ci = bootstrap_ci(correct_list)
    total = len(correct_list)
    accuracy = float(np.mean(correct_list) * 100.0)

    # Letter distribution
    letter_counts = Counter(predicted_letters)
    letter_dist = {
        letter: float((letter_counts.get(letter, 0) / total) * 100.0)
        for letter in CHOICE_LETTERS
    }

    # Domain breakdown
    domain_results = {}
    domains = set(r["domain"] for r in records)
    for dom in sorted(domains):
        dom_records = [r for r in records if r["domain"] == dom]
        dom_correct = [1.0 if r["correct"] else 0.0 for r in dom_records]
        dom_ci = bootstrap_ci(dom_correct)
        domain_results[dom] = {
            "n": len(dom_records),
            "accuracy": float(np.mean(dom_correct) * 100.0),
            "ci_95": [float(dom_ci["ci_low"] * 100.0), float(dom_ci["ci_high"] * 100.0)],
        }

    summary = {
        "dataset": dataset_name,
        "split": split,
        "n_samples": total,
        "accuracy": accuracy,
        "ci_95": [float(overall_ci["ci_low"] * 100.0), float(overall_ci["ci_high"] * 100.0)],
        "chance_level": 25.0,
        "predicted_letter_distribution": letter_dist,
        "domain_accuracy": domain_results,
    }

    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Marathi model on MILU MCQ")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", help="Base model ID or path")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--split", default="validation", choices=["validation", "test"], help="MILU split")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--no-chat-template", action="store_true", help="Disable chat template wrapper")
    parser.add_argument("--output-dir", default="results/milu", help="Directory to save results")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    if args.adapter:
        print(f"Attaching LoRA adapter from: {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    results = evaluate_milu(
        model=model,
        tokenizer=tokenizer,
        split=args.split,
        batch_size=args.batch_size,
        use_chat_template=not args.no_chat_template,
    )

    summary = results["summary"]
    print("\n--- MILU Evaluation Summary ---")
    print(f"Split: {summary['split']} (n={summary['n_samples']})")
    print(f"Overall Accuracy: {summary['accuracy']:.2f}% (95% CI: [{summary['ci_95'][0]:.2f}%, {summary['ci_95'][1]:.2f}%])")
    print(f"Chance level: {summary['chance_level']:.1f}%")
    print(f"Letter Distribution: {summary['predicted_letter_distribution']}")
    print("\nDomain Accuracy:")
    for dom, dinfo in summary["domain_accuracy"].items():
        print(f"  {dom} (n={dinfo['n']}): {dinfo['accuracy']:.2f}% (95% CI: [{dinfo['ci_95'][0]:.2f}%, {dinfo['ci_95'][1]:.2f}%])")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"milu_{args.split}_summary.json"
    records_path = out_dir / f"milu_{args.split}_records.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(results["records"], f, indent=2)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved records to: {records_path}")


if __name__ == "__main__":
    main()
