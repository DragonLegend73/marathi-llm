"""
infer.py — Compare base model vs fine-tuned LoRA adapter
---------------------------------------------------------
Runs a set of Marathi prompts through both models and saves
a side-by-side comparison to evals/comparison.csv

Usage:  python infer.py
"""

import os, csv, torch
from unsloth import FastLanguageModel
from config import MODEL_NAME, MAX_SEQ_LEN, ADAPTER_DIR, MAX_NEW_TOKENS, TEMPERATURE

# ── Prompts (Marathi instructions) ───────────────────────────
# Format mirrors translated_text: the model completes after ### प्रतिसाद:
PROMPTS = [
    "निरोगी राहण्यासाठी तीन टिप्स द्या.",
    "तीन मुख्य रंग कोणते आहेत?",
    "पाण्याचे महत्त्व सांगा.",
    "भारताची राजधानी कोणती आहे?",
    "सूर्यमाला म्हणजे काय?",
    "मित्राला पत्र कसे लिहावे?",
    "झाडे लावणे का महत्त्वाचे आहे?",
    "संगणक म्हणजे काय आणि त्याचा उपयोग काय?",
    "वेळेचे व्यवस्थापन कसे करावे?",
    "चांगल्या विद्यार्थ्याचे गुण कोणते असतात?",
]

PROMPT_PREFIX = (
    "खाली एक सूचना आहे जी एखाद्या कार्याचे वर्णन करते. "
    "विनंती योग्य रितीने पूर्ण करणारा प्रतिसाद लिहा.\n\n"
    "### सूचना:\n{instruction}\n\n### प्रतिसाद:\n"
)

def build_prompt(instruction):
    return PROMPT_PREFIX.format(instruction=instruction)

def load_model(adapter_dir=None):
    """Load model, optionally with a LoRA adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = MODEL_NAME,
        max_seq_length= MAX_SEQ_LEN,
        dtype         = torch.float16,
        load_in_4bit  = True,
    )
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"  Loaded adapter from {adapter_dir}")
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = TEMPERATURE,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── Run base model ────────────────────────────────────────────
print("=" * 60)
print("Loading BASE model …")
base_model, tokenizer = load_model(adapter_dir=None)

base_outputs = []
for i, instruction in enumerate(PROMPTS):
    prompt = build_prompt(instruction)
    out = generate(base_model, tokenizer, prompt)
    base_outputs.append(out)
    print(f"  [{i+1}/{len(PROMPTS)}] {instruction[:40]}…")

del base_model
torch.cuda.empty_cache()

# ── Run fine-tuned model ──────────────────────────────────────
print("\nLoading FINE-TUNED model …")
ft_model, tokenizer = load_model(adapter_dir=ADAPTER_DIR)

ft_outputs = []
for i, instruction in enumerate(PROMPTS):
    prompt = build_prompt(instruction)
    out = generate(ft_model, tokenizer, prompt)
    ft_outputs.append(out)
    print(f"  [{i+1}/{len(PROMPTS)}] {instruction[:40]}…")

del ft_model
torch.cuda.empty_cache()

# ── Save comparison CSV ───────────────────────────────────────
os.makedirs("evals", exist_ok=True)
out_path = "evals/comparison.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["prompt", "base_output", "finetuned_output"])
    writer.writeheader()
    for prompt, base, ft in zip(PROMPTS, base_outputs, ft_outputs):
        writer.writerow({
            "prompt"          : prompt,
            "base_output"     : base,
            "finetuned_output": ft,
        })

print(f"\nComparison saved → {out_path}")

# ── Print summary to terminal ─────────────────────────────────
print("\n" + "=" * 60)
for i, (prompt, base, ft) in enumerate(zip(PROMPTS, base_outputs, ft_outputs)):
    print(f"\n[{i+1}] Prompt   : {prompt}")
    print(f"    Base     : {base[:120]}")
    print(f"    FineTuned: {ft[:120]}")
