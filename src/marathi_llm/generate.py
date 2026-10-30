"""Deterministic batched generation runner for base models and LoRA adapters."""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_id: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda:0",
    prequantized: Optional[bool] = None,
    trust_remote_code: bool = False,
):
    """
    Loads base model and optional LoRA adapter with left-padding tokenizer.
    If adapter_path is provided, loads from local disk or Hugging Face Hub and verifies attachment.
    """
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check whether the checkpoint is pre-quantized (e.g. *-bnb-4bit)
    if prequantized is None:
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            prequantized = getattr(cfg, "quantization_config", None) is not None
        except Exception:
            prequantized = "bnb-4bit" in model_id.lower() or "4bit" in model_id.lower()

    print(f"Loading model: {model_id} (prequantized={prequantized})...")
    kwargs = {"device_map": device, "trust_remote_code": trust_remote_code}

    if not prequantized:
        # Standardized 4-bit NF4 configuration matching Unsloth prequantized weights
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    adapter_loaded = False
    if adapter_path:
        print(f"Attaching LoRA adapter: {adapter_path}...")
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            if not isinstance(model, PeftModel):
                raise RuntimeError(f"Model was not wrapped as PeftModel after loading from {adapter_path}")
            adapter_loaded = True
            print(f"Successfully attached LoRA adapter from {adapter_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA adapter from '{adapter_path}': {e}") from e

    model.eval()
    return model, tokenizer


def prepare_prompt(
    raw_prompt: str,
    tokenizer,
    prompt_format: str = "chat",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Formats prompt according to prompt_format:
    - 'chat': native chat template (with enable_thinking=False for Qwen3)
    - 'alpaca_v0': legacy Alpaca training template
    - 'raw': unchanged string
    """
    if prompt_format == "alpaca_v0":
        return f"### सूचना:\n{raw_prompt}\n\n### उत्तर:\n"

    if prompt_format == "raw" or not hasattr(tokenizer, "apply_chat_template") or not tokenizer.chat_template:
        return raw_prompt

    # Default: 'chat' format
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": raw_prompt})

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Explicitly disable Qwen3 thinking mode to avoid wasting generation tokens on <think>
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def get_all_eos_token_ids(model, tokenizer) -> Set[int]:
    """Retrieves all valid EOS token IDs from generation config and tokenizer."""
    eos_ids = set()
    gen_config = getattr(model, "generation_config", None)
    if gen_config and gen_config.eos_token_id is not None:
        if isinstance(gen_config.eos_token_id, list):
            eos_ids.update(gen_config.eos_token_id)
        else:
            eos_ids.add(gen_config.eos_token_id)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    return eos_ids


def generate_batch(
    model,
    tokenizer,
    items: List[Dict[str, Any]],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    prompt_format: str = "chat",
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

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    eos_token_ids = get_all_eos_token_ids(model, tokenizer)
    primary_eos_id = tokenizer.eos_token_id

    # Format prompts
    formatted = []
    for idx, it in enumerate(items):
        raw_p = it.get("prompt", "")
        p_text = prepare_prompt(raw_p, tokenizer, prompt_format=prompt_format, system_prompt=system_prompt)
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
    n_total = len(sorted_items)
    truncated_count = 0

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

        # Check if left-truncation occurred
        for t_idx, txt in enumerate(batch_texts):
            raw_len = len(tokenizer.encode(txt, add_special_tokens=False))
            if raw_len > 2048:
                truncated_count += 1

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=primary_eos_id,
            )

        prefix_len = inputs["input_ids"].shape[1]

        for i, c in enumerate(chunk):
            full_seq = outputs[i]
            generated_ids = full_seq[prefix_len:].tolist()

            # Find earliest occurrence of any valid EOS token ID
            earliest_eos_idx = None
            for idx_tok, tok_id in enumerate(generated_ids):
                if tok_id in eos_token_ids:
                    earliest_eos_idx = idx_tok
                    break

            if earliest_eos_idx is not None:
                finish_reason = "eos"
                trimmed_ids = generated_ids[:earliest_eos_idx]
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

    if truncated_count > 0:
        print(f"Warning: {truncated_count} prompts were left-truncated at max_length=2048.")

    return results_in_order


def main():
    parser = argparse.ArgumentParser(description="Batched greedy generation runner")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file with 'prompt' key")
    parser.add_argument("--output", default="results/generations.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-format", default="chat", choices=["chat", "alpaca_v0", "raw"])
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
        prompt_format=args.prompt_format,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} generations to {args.output}")


if __name__ == "__main__":
    main()
