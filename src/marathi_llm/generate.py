"""Deterministic batched generation runner for base models and LoRA adapters."""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_and_tokenizer(
    model_id: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda:0",
    prequantized: bool = True,
):
    """Loads base model (in 4-bit) and optional LoRA adapter with left-padding tokenizer."""
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}...")
    # Note: If checkpoint is pre-quantized (e.g. *-bnb-4bit), do not pass load_in_4bit=True
    kwargs = {"device_map": device, "trust_remote_code": True}
    if not prequantized:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter_path and os.path.exists(adapter_path):
        print(f"Attaching LoRA adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def prepare_prompt(
    raw_prompt: str,
    tokenizer,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
) -> str:
    """Formats prompt with the model's native chat template if enabled."""
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template") or not tokenizer.chat_template:
        return raw_prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": raw_prompt})

    # Disable thinking tokens for Qwen3 if applicable
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def generate_batch(
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    device: str = "cuda:0",
) -> List[Dict[str, Any]]:
    """
    Executes greedy batched generation with deterministic length-based sorting.
    Each item must be a dict with 'id' and 'prompt' keys.
    Returns items with 'output', 'finish_reason', and 'n_new_tokens'.
    """
    if not items:
        return []

    # Prepare formatted prompts
    formatted = []
    for idx, it in enumerate(items):
        raw_p = it.get("prompt", "")
        p_text = prepare_prompt(raw_p, tokenizer, use_chat_template, system_prompt)
        formatted.append({
            "orig_idx": idx,
            "id": it.get("id", str(idx)),
            "raw_prompt": raw_p,
            "input_text": p_text,
            "meta": {k: v for k, v in it.items() if k not in ["id", "prompt"]},
        })

    # Sort deterministically by prompt character length with stable tie-break
    sorted_items = sorted(formatted, key=lambda x: (len(x["input_text"]), x["orig_idx"]))

    results_in_order = [None] * len(items)
    eos_id = tokenizer.eos_token_id

    n_total = len(sorted_items)
    for start in range(0, n_total, batch_size):
        chunk = sorted_items[start : min(start + batch_size, n_total)]
        batch_texts = [c["input_text"] for c in chunk]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        input_lengths = [int(mask.sum().item()) for mask in inputs["attention_mask"]]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=eos_id,
            )

        for i, c in enumerate(chunk):
            full_seq = outputs[i]
            # Since inputs are left-padded, new tokens begin at inputs["input_ids"].shape[1]
            prefix_len = inputs["input_ids"].shape[1]
            generated_ids = full_seq[prefix_len:]

            # Remove padding tokens from the end if any
            if eos_id is not None and (generated_ids == eos_id).any():
                finish_reason = "eos"
                # Slice up to first eos token
                eos_indices = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
                first_eos = int(eos_indices[0].item())
                trimmed_ids = generated_ids[: first_eos + 1]
                n_new = len(trimmed_ids)
            else:
                finish_reason = "length" if len(generated_ids) >= max_new_tokens else "eos"
                trimmed_ids = generated_ids
                n_new = len(trimmed_ids)

            output_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

            record = {
                "id": c["id"],
                "prompt": c["raw_prompt"],
                "output": output_text,
                "finish_reason": finish_reason,
                "n_new_tokens": n_new,
                **c["meta"],
            }
            results_in_order[c["orig_idx"]] = record

        if (start // batch_size) % 25 == 0:
            print(f"Generated {min(start + batch_size, n_total)}/{n_total} items...")

    return results_in_order


def main():
    parser = argparse.ArgumentParser(description="Batched greedy generation runner")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file with 'prompt' key")
    parser.add_argument("--output", default="results/generations.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    # Load input items
    in_path = Path(args.input)
    items = []
    if in_path.suffix == ".jsonl":
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("items", [])

    print(f"Loaded {len(items)} items from {args.input}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.adapter)
    results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        items=items,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=not args.no_chat_template,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} generations to {args.output}")


if __name__ == "__main__":
    main()
