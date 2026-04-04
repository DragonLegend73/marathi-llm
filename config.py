# config.py — Single source of truth for all hyperparameters

MODEL_NAME      = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DATASET_NAME    = "rachittshah/alpaca-marahti"
OUTPUT_DIR      = "./outputs"
ADAPTER_DIR     = "./outputs/lora_adapter"

# ── Model / LoRA ──────────────────────────────────────────────
# RTX 3050 Ti 4 GB — safe ceiling values
MAX_SEQ_LEN     = 512

LORA_R          = 16
LORA_ALPHA      = 32          # keep alpha = 2×r

# MUST be 0 — Unsloth disables fast patching when dropout > 0,
# which causes "0 QKV/O/MLP layers patched" and kills LoRA effectiveness.
LORA_DROPOUT    = 0

# TARGET_MODULES intentionally omitted.
# Passing Llama-style names ("q_proj" etc.) to a Qwen2 model causes
# Unsloth to match 0 layers. Omitting lets Unsloth auto-select the
# correct layer names for whatever model is loaded.

# ── Training ──────────────────────────────────────────────────
NUM_EPOCHS      = 1.0         # Forces exactly 1 full pass over the dataset
BATCH_SIZE      = 1           # Hardware limit
GRAD_ACCUM      = 8           # Effective batch = 8
LEARNING_RATE   = 2e-4
LOGGING_STEPS   = 50          # Scaled up to avoid console spam over 6000 steps
SAVE_STEPS      = 1000        # Save checkpoint every ~16% of the epoch
EVAL_STEPS      = 500

# ── Inference ─────────────────────────────────────────────────
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.7
