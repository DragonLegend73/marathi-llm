"""
scripts/rescore_v0.py — Task 0.5: Re-score base model vs v0 adapter on clean S0 test split (200 prompts)
--------------------------------------------------------------------------------------------------------
Evaluates:
- Corpus chrF++ against clean test ground-truth responses
- Script purity (% of letters that are Devanagari)
- EOS rate (% of outputs that terminated with EOS vs max_new_tokens)
- Mean output length
"""

import os
import json
import re
import torch
import unicodedata
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from unsloth import FastLanguageModel
from peft import PeftModel

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
ADAPTER_NAME = "DragonLegend/marathi-qwen2.5-lora"
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 256

PROMPT_PREFIX = (
    "खाली एक सूचना आहे जी एखाद्या कार्याचे वर्णन करते. "
    "विनंती योग्य रितीने पूर्ण करणारा प्रतिसाद लिहा.\n\n"
    "### सूचना:\n{instruction}\n\n### प्रतिसाद:\n"
)

def is_devanagari_char(c):
    return '\u0900' <= c <= '\u097F'

def compute_script_purity(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    deva_count = sum(1 for c in letters if is_devanagari_char(c))
    return deva_count / len(letters)

def main():
    print("Loading test split IDs from data/splits/s0_test_200.json...")
    with open("data/splits/s0_test_200.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    test_ids = set(meta["row_ids"])
    print(f"Total test prompts: {len(test_ids)}")

    print("Loading dataset to extract prompt instructions and references...")
    ds = load_dataset("rachittshah/alpaca-marahti", split="train")
    
    test_examples = []
    for idx in meta["row_ids"]:
        row = ds[idx]
        trans = (row.get("translated_text") or "").strip()
        inst = ""
        resp = ""
        if "### सूचना:" in trans and "### प्रतिसाद:" in trans:
            parts = trans.split("### प्रतिसाद:")
            resp = parts[1].strip() if len(parts) > 1 else ""
            inst_part = parts[0]
            if "### सूचना:" in inst_part:
                inst = inst_part.split("### सूचना:")[1].strip()
                if "### इनपुट:" in inst:
                    inst = inst.split("### इनपुट:")[0].strip()
        else:
            inst = row.get("instruction") or ""
            resp = row.get("output") or ""
        
        if inst and resp:
            test_examples.append({
                "id": idx,
                "instruction": inst,
                "reference": resp
            })

    print(f"Valid test examples extracted: {len(test_examples)}")

    chrf = CHRF(word_order=2)

    def evaluate_model(adapter_id=None):
        tag = "FINE-TUNED (v0 adapter)" if adapter_id else "BASE MODEL"
        print(f"\n{'='*70}\nLoading {tag}...\n{'='*70}")
        
        torch.cuda.empty_cache()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        if adapter_id:
            print(f"Loading adapter: {adapter_id}")
            model = PeftModel.from_pretrained(model, adapter_id)
        
        FastLanguageModel.for_inference(model)
        eos_id = tokenizer.eos_token_id

        generations = []
        stopped_eos_count = 0
        total_purity = 0.0

        for i, item in enumerate(test_examples):
            prompt = PROMPT_PREFIX.format(instruction=item["instruction"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,  # greedy decoding per working principles
                    pad_token_id=eos_id,
                )
            
            in_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[0][in_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            
            # Check if last token is EOS
            hit_eos = (gen_ids[-1].item() == eos_id) if len(gen_ids) > 0 else False
            if hit_eos or len(gen_ids) < MAX_NEW_TOKENS:
                stopped_eos_count += 1
            
            purity = compute_script_purity(gen_text)
            total_purity += purity
            
            generations.append({
                "id": item["id"],
                "instruction": item["instruction"],
                "reference": item["reference"],
                "output": gen_text,
                "token_len": len(gen_ids),
                "hit_eos": hit_eos or (len(gen_ids) < MAX_NEW_TOKENS),
                "script_purity": purity
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(test_examples):
                print(f"  Processed {i+1}/{len(test_examples)}...")

        # Calculate metrics
        refs = [[item["reference"]] for item in test_examples]
        hyps = [g["output"] for g in generations]
        corpus_score = chrf.corpus_score(hyps, refs).score
        eos_rate = (stopped_eos_count / len(test_examples)) * 100.0
        mean_purity = (total_purity / len(test_examples)) * 100.0
        mean_len = sum(g["token_len"] for g in generations) / len(generations)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        return {
            "corpus_chrf": corpus_score,
            "eos_rate": eos_rate,
            "script_purity": mean_purity,
            "mean_tokens": mean_len,
            "generations": generations
        }

    # 1. Base model
    base_res = evaluate_model(adapter_id=None)

    # 2. Fine-tuned model
    ft_res = evaluate_model(adapter_id=ADAPTER_NAME)

    # Summary table
    print("\n" + "="*80)
    print("TASK 0.5: HONEST v0 RE-EVALUATION (200 UNSEEN TEST PROMPTS)")
    print("="*80)
    print(f"{'Metric':<25} | {'Base Model':<15} | {'Fine-Tuned (v0)':<15} | {'Delta':<10}")
    print("-" * 75)
    
    delta_chrf = ft_res["corpus_chrf"] - base_res["corpus_chrf"]
    delta_eos = ft_res["eos_rate"] - base_res["eos_rate"]
    delta_purity = ft_res["script_purity"] - base_res["script_purity"]
    delta_len = ft_res["mean_tokens"] - base_res["mean_tokens"]

    print(f"{'Corpus chrF++':<25} | {base_res['corpus_chrf']:<15.2f} | {ft_res['corpus_chrf']:<15.2f} | {delta_chrf:+.2f}")
    print(f"{'Script Purity (%)':<25} | {base_res['script_purity']:<15.2f} | {ft_res['script_purity']:<15.2f} | {delta_purity:+.2f}%")
    print(f"{'EOS Rate (%)':<25} | {base_res['eos_rate']:<15.2f} | {ft_res['eos_rate']:<15.2f} | {delta_eos:+.2f}%")
    print(f"{'Mean Output Tokens':<25} | {base_res['mean_tokens']:<15.1f} | {ft_res['mean_tokens']:<15.1f} | {delta_len:+.1f}")
    print("="*80)

    # Save to results/v0_rescore.json
    os.makedirs("results", exist_ok=True)
    out_path = "results/v0_rescore.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_prompts_count": len(test_examples),
            "base": {
                "corpus_chrf": round(base_res["corpus_chrf"], 2),
                "eos_rate": round(base_res["eos_rate"], 2),
                "script_purity": round(base_res["script_purity"], 2),
                "mean_tokens": round(base_res["mean_tokens"], 1),
            },
            "finetuned_v0": {
                "corpus_chrf": round(ft_res["corpus_chrf"], 2),
                "eos_rate": round(ft_res["eos_rate"], 2),
                "script_purity": round(ft_res["script_purity"], 2),
                "mean_tokens": round(ft_res["mean_tokens"], 1),
            },
            "delta": {
                "chrf": round(delta_chrf, 2),
                "eos_rate": round(delta_eos, 2),
                "script_purity": round(delta_purity, 2),
            },
            "base_generations_sample": base_res["generations"][:5],
            "ft_generations_sample": ft_res["generations"][:5],
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
