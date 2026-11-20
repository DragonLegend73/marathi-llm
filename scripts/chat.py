"""
chat.py — Interactive terminal chat with fine-tuned Marathi models
----------------------------------------------------------------------
Usage:
    python scripts/chat.py --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit --adapter results/runs/A_qwen25_1p5b_768/adapter
"""

import argparse
import torch
from marathi_llm.generate import load_model_and_tokenizer, prepare_prompt


def main():
    parser = argparse.ArgumentParser(description="Interactive terminal chat for Marathi LLM")
    parser.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", help="Base model ID")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--prompt-format", default="chat", choices=["chat", "alpaca_v0", "raw", "plain"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model: {args.model} (adapter={args.adapter})...")
    model, tokenizer, _ = load_model_and_tokenizer(
        model_id=args.model,
        adapter_path=args.adapter,
        device=args.device,
    )
    print("Model ready. Type your instruction in Marathi (or English).")
    print("Commands: 'quit' or 'exit' to quit.\n")

    uses_template = (args.prompt_format == "chat") and bool(getattr(tokenizer, "chat_template", None))
    add_special = not uses_template

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

        prompt = prepare_prompt(user_input, tokenizer, prompt_format=args.prompt_format)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special).to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        print(f"\nModel: {response}\n")


if __name__ == "__main__":
    main()
