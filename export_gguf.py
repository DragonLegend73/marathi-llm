"""
export_gguf.py — Merge LoRA adapter into base model and export to GGUF
----------------------------------------------------------------------
Usage:
    python export_gguf.py --adapter results/runs/A_qwen25_1p5b_768/adapter --output-dir outputs

Then load into Ollama:
    cd outputs
    ollama create marathi-llm -f Modelfile
    ollama run marathi-llm
"""

import os
import argparse
from pathlib import Path
import torch
from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF for Ollama")
    parser.add_argument("--adapter", default="outputs/lora_adapter", help="Path to trained LoRA adapter directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for GGUF model and Modelfile")
    parser.add_argument("--max-seq-len", type=int, default=768, help="Maximum sequence length")
    parser.add_argument("--quantization-method", default="q4_k_m", help="GGUF quantization method (e.g. q4_k_m, q8_0, f16)")
    args = parser.parse_args()

    adapter_dir = args.adapter
    output_dir = args.output_dir
    max_seq_len = args.max_seq_len
    quant_method = args.quantization_method

    merged_dir = os.path.join(output_dir, "merged_model")
    gguf_path  = os.path.join(output_dir, f"marathi-llm-{quant_method}.gguf")
    modelfile  = os.path.join(output_dir, "Modelfile")

    # ── Step 1: Load base + adapter ───────────────────────────────
    print(f"Loading LoRA adapter from {adapter_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = adapter_dir,   # Unsloth can load adapter dir directly
        max_seq_length= max_seq_len,
        dtype         = torch.float16,
        load_in_4bit  = True,
    )
    FastLanguageModel.for_inference(model)

    # ── Step 2: Export to GGUF ────────────────────────────────────
    print(f"Exporting to GGUF ({quant_method}) → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method = quant_method,
    )
    print("GGUF export complete ✓")

    # ── Step 3: Write Modelfile for Ollama ────────────────────────
    gguf_files = [f for f in os.listdir(output_dir) if f.endswith(".gguf")]
    if not gguf_files:
        print("Warning: no .gguf file found in output dir. Check export above.")
        gguf_filename = f"marathi-llm-{quant_method}.gguf"
    else:
        # Use latest gguf
        gguf_files_sorted = sorted(gguf_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
        gguf_filename = gguf_files_sorted[-1]
        print(f"Found GGUF file: {gguf_filename}")

    modelfile_content = f"""FROM ./{gguf_filename}

# System prompt — tells the model to respond in Marathi
SYSTEM "तुम्ही एक मराठी भाषेतील सहाय्यक आहात. नेहमी मराठीत उत्तर द्या."

# Generation parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
"""

    with open(modelfile, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"Modelfile written → {modelfile}")

    # ── Step 4: Print Ollama instructions ────────────────────────
    print("\n" + "=" * 60)
    print("Next steps to run in Ollama:")
    print("=" * 60)
    print(f"\n  cd {output_dir}")
    print( "  ollama create marathi-llm -f Modelfile")
    print( "  ollama run marathi-llm")
    print("\nOr use the OpenAI-compatible API:")
    print( "  ollama serve   # starts on http://localhost:11434")
    print("=" * 60)


if __name__ == "__main__":
    main()
