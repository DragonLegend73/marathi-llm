# Interactive demo tool for chatting with fine-tuned models
"""
chat.py — Interactive terminal chat with the fine-tuned Marathi model
----------------------------------------------------------------------
Loads the LoRA adapter and starts a conversation loop.
Type 'quit' or 'exit' to stop.  Type 'reset' to clear history.

Usage:  python chat.py
"""

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from config import MODEL_NAME, MAX_SEQ_LEN, ADAPTER_DIR, MAX_NEW_TOKENS, TEMPERATURE

PROMPT_PREFIX = (
    "खाली एक सूचना आहे जी एखाद्या कार्याचे वर्णन करते. "
    "विनंती योग्य रितीने पूर्ण करणारा प्रतिसाद लिहा.\n\n"
    "### सूचना:\n{instruction}\n\n### प्रतिसाद:\n"
)

def build_prompt(instruction):
    return PROMPT_PREFIX.format(instruction=instruction)

# ── Load model ────────────────────────────────────────────────
print("Loading fine-tuned model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = MODEL_NAME,
    max_seq_length= MAX_SEQ_LEN,
    dtype         = torch.float16,
    load_in_4bit  = True,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
FastLanguageModel.for_inference(model)
print("Model ready. Type your instruction in Marathi (or English).")
print("Commands: 'reset' to clear context | 'quit' to exit\n")

# ── Chat loop ─────────────────────────────────────────────────
while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break

    if user_input.lower() == "reset":
        print("[Context reset]\n")
        continue

    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            max_length     = None,
            temperature    = TEMPERATURE,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"\nModel: {response}\n")
