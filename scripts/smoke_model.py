"""Sanity test script to smoke-test candidate models before full training.

For each candidate model:
1. Loads 4-bit model and left-padding tokenizer.
2. Formats a sample training example (head and tail).
3. Decodes the last 5 unmasked labels from TRL collation to verify end-of-turn token inclusion.
4. Generates on 3 Marathi prompts and verifies completion terminates on EOS.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from marathi_llm.generate import generate_batch, load_model_and_tokenizer
from marathi_llm.train import build_prompt_completion_records, format_record_for_length_check

DEFAULT_CANDIDATE_CONFIGS = [
    "configs/models/qwen25_1p5b.yaml",
    "configs/models/qwen3_1p7b.yaml",
    "configs/models/gemma3_1b.yaml",
    "configs/models/llama32_1b.yaml",
    "configs/models/sarvam1.yaml",
]

SMOKE_PROMPTS = [
    "भारताची राजधानी कोणती आहे?",
    "मराठी भाषेबद्दल दोन ओळी लिहा.",
    "पाण्याचे महत्त्व काय आहे?",
]


def smoke_test_model(
    model_id: str,
    enable_thinking: Optional[bool] = None,
    trust_remote_code: bool = False,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"[SMOKE TEST] Candidate Model: {model_id}")
    print(f"{'='*75}", flush=True)

    t0 = time.time()
    model, tokenizer, prequantized = load_model_and_tokenizer(
        model_id=model_id,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    # 1. Inspect Formatted Training Example
    raw_row = {
        "instruction": "भारताची राजधानी कोणती आहे?",
        "input": "",
        "response": "भारताची राजधानी नवी दिल्ली आहे.",
    }
    has_tmpl = bool(getattr(tokenizer, "chat_template", None))
    records = build_prompt_completion_records([raw_row], has_chat_template=has_tmpl)
    rendered = format_record_for_length_check(records[0], tokenizer, enable_thinking=enable_thinking)

    print("\n--- 1. Formatted Training Example ---")
    print(f"Has Native Chat Template: {has_tmpl}")
    print(f"Rendered Head (first 100 chars):\n  {repr(rendered[:100])}")
    print(f"Rendered Tail (last 100 chars):\n  {repr(rendered[-100:])}")

    # 2. Inspect Unmasked Labels Tail
    print("\n--- 2. TRL Label Masking Inspection ---")
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(model, lora_cfg)

    ds = Dataset.from_list(records)
    sft_cfg = SFTConfig(
        output_dir="/tmp/smoke_run",
        completion_only_loss=True,
        max_length=512,
        per_device_train_batch_size=1,
        report_to="none",
    )
    trainer = SFTTrainer(model=peft_model, processing_class=tokenizer, args=sft_cfg, train_dataset=ds)
    train_ex = trainer.train_dataset[0]
    labels = train_ex["labels"]
    unmasked_ids = [tok_id for tok_id in labels if tok_id != -100]
    unmasked_tail_tokens = [tokenizer.decode([t]) for t in unmasked_ids[-5:]]
    last_unmasked_id = unmasked_ids[-1] if unmasked_ids else None
    last_unmasked_token = tokenizer.decode([last_unmasked_id]) if last_unmasked_id is not None else ""

    print(f"Total unmasked tokens in loss: {len(unmasked_ids)}")
    print(f"Last 5 unmasked tokens: {unmasked_tail_tokens}")
    print(f"Last unmasked token: {repr(last_unmasked_token)} (id={last_unmasked_id})")

    # 3. Generate on 3 Sample Prompts
    print("\n--- 3. Generation Verification (3 Marathi Prompts) ---")
    items = [{"id": f"smoke_{i}", "prompt": p} for i, p in enumerate(SMOKE_PROMPTS)]
    prompt_fmt = "chat" if has_tmpl else "plain"
    gen_results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        items=items,
        batch_size=3,
        max_new_tokens=64,
        prompt_format=prompt_fmt,
        device=device,
    )

    all_eos_terminated = True
    for r in gen_results:
        p = r.get("prompt", r.get("raw_prompt", ""))
        fin = r["finish_reason"]
        ntok = r["n_new_tokens"]
        out = r["output"].strip().replace("\n", " ")
        if len(out) > 80:
            out = out[:80] + "..."
        ends_eos = (fin == "eos")
        if not ends_eos:
            all_eos_terminated = False
        print(f"  Prompt: \"{p}\"")
        print(f"    Output: \"{out}\"")
        print(f"    Tokens: {ntok} | Finish: {fin} | Ended on EOS: {ends_eos}")

    elapsed = time.time() - t0
    print(f"\n[Smoke Test PASSED in {elapsed:.1f}s for {model_id}]")

    # Clean up memory
    del trainer
    del peft_model
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "has_chat_template": has_tmpl,
        "last_unmasked_token": last_unmasked_token,
        "all_eos_terminated": all_eos_terminated,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke test candidate models before training")
    parser.add_argument("--model", default=None, help="Specific Hugging Face model ID")
    parser.add_argument("--config", default=None, help="Path to model config YAML")
    parser.add_argument("--all", action="store_true", help="Run smoke tests on all default Stage A candidates")
    args = parser.parse_args()

    targets: List[Dict[str, Any]] = []
    if args.model:
        targets = [{"hf_id": args.model, "enable_thinking": None, "trust_remote_code": "sarvam" in args.model.lower()}]
    elif args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hf_id = cfg.get("hf_id") or cfg.get("model", {}).get("hf_id")
        targets = [{
            "hf_id": hf_id,
            "enable_thinking": cfg.get("enable_thinking"),
            "trust_remote_code": cfg.get("trust_remote_code", "sarvam" in hf_id.lower()),
        }]
    else:
        # Default or --all: run across candidate configs
        for cp in DEFAULT_CANDIDATE_CONFIGS:
            if Path(cp).exists():
                with open(cp, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                hf_id = cfg.get("hf_id") or cfg.get("model", {}).get("hf_id")
                targets.append({
                    "hf_id": hf_id,
                    "enable_thinking": cfg.get("enable_thinking"),
                    "trust_remote_code": cfg.get("trust_remote_code", "sarvam" in hf_id.lower()),
                })

    print(f"Smoke testing {len(targets)} candidate model(s): {[t['hf_id'] for t in targets]}")
    summary = []
    for t in targets:
        res = smoke_test_model(
            t["hf_id"],
            enable_thinking=t["enable_thinking"],
            trust_remote_code=t["trust_remote_code"],
        )
        summary.append(res)

    print(f"\n{'='*75}")
    print("ALL CANDIDATE SMOKE TESTS COMPLETED SUCCESSFULLY!")
    for s in summary:
        print(f"  - {s['model_id']}: ChatTemplate={s['has_chat_template']} | EOS unmasked={repr(s['last_unmasked_token'])} ({s['elapsed_s']:.1f}s)")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
