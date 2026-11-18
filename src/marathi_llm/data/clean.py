"""Text cleaning, script verification, and filtering for Marathi instruction datasets."""

import re
import unicodedata
from typing import Dict, Any, Tuple, Optional
from marathi_llm.eval.auto_metrics import (
    compute_script_purity,
    compute_english_leak_ratio,
    compute_repetition_metrics,
    get_lingua_detector,
)

# Zero-width spaces, joiners, BOM
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")
NEWLINE_PATTERN = re.compile(r"\n{3,}")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
ARTIFACT_PATTERN = re.compile(r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|###\s*(?:Instruction|Input|Response|प्रतिसाद|सूचना))", re.IGNORECASE)


def normalize_marathi_text(text: str) -> str:
    """Applies Unicode NFC, strips zero-width chars, removes HTML, and normalizes whitespace."""
    if not text:
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Strip zero-width chars
    text = ZERO_WIDTH_PATTERN.sub("", text)
    # 3. Strip HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)
    # 4. Normalize spaces and multiple newlines
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def count_training_tokens(
    instruction: str,
    response: str,
    tokenizer,
    input_text: str = "",
) -> int:
    """
    Counts exact training tokens using the tokenizer's chat template (matching
    `marathi_llm.train`), falling back to plain prompt-response + EOS if the
    tokenizer defines no chat template.
    """
    user_content = f"{instruction.strip()}\n\n{input_text.strip()}" if (input_text and input_text.strip()) else instruction.strip()
    resp_text = response.strip()

    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": resp_text},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return len(tokenizer.encode(rendered, add_special_tokens=False))
        except Exception:
            pass

    formatted = f"### सूचना:\n{user_content}\n\n### प्रतिसाद:\n{resp_text}"
    eos = getattr(tokenizer, "eos_token", "") or ""
    if eos:
        formatted = f"{formatted}{eos}"
    return len(tokenizer.encode(formatted, add_special_tokens=False))


def validate_example(
    instruction: str,
    response: str,
    input_text: str = "",
    min_devanagari_purity: float = 60.0,
    max_latin_ratio: float = 0.15,
    min_response_chars: int = 10,
    max_repeat_ngram_frac: float = 0.25,
    require_mr_lang_id: bool = False,
    tokenizer=None,
    max_seq_len: int = 768,
) -> Tuple[bool, Optional[str]]:
    """
    Validates a cleaned (instruction, input, response) triple against quality thresholds.
    Returns (is_valid, reject_reason).
    """
    if not instruction or not response:
        return False, "empty_instruction_or_response"

    if len(response) < min_response_chars:
        return False, "response_too_short"

    # Template artifact check
    if ARTIFACT_PATTERN.search(instruction) or ARTIFACT_PATTERN.search(response):
        return False, "template_artifact_leak"

    # Devanagari purity check on instruction and response
    inst_purity = compute_script_purity(instruction)
    resp_purity = compute_script_purity(response)
    if inst_purity < min_devanagari_purity:
        return False, f"low_instruction_script_purity ({inst_purity:.1f}%)"
    if resp_purity < min_devanagari_purity:
        return False, f"low_response_script_purity ({resp_purity:.1f}%)"

    # Latin ratio check (detect excessive English leakage)
    inst_latin = compute_english_leak_ratio(instruction)
    resp_latin = compute_english_leak_ratio(response)
    if inst_latin > max_latin_ratio or resp_latin > max_latin_ratio:
        return False, f"high_latin_ratio (inst={inst_latin:.2f}, resp={resp_latin:.2f})"

    # Repetition check on response
    rep_metrics = compute_repetition_metrics(response)
    if rep_metrics.get("is_repetitive", 0.0) > 0.5:
        return False, "degenerate_repetition"

    # Optional Language ID check
    if require_mr_lang_id:
        detector = get_lingua_detector()
        if detector:
            from lingua import Language
            detected = detector.detect_language_of(response)
            if detected is not None and detected != Language.MARATHI:
                return False, f"language_id_mismatch ({detected.name})"

    # Token length bounds (chat-template counter matching marathi_llm.train)
    if tokenizer is not None:
        n_toks = count_training_tokens(instruction, response, tokenizer, input_text=input_text)
        if n_toks > max_seq_len:
            return False, f"exceeds_max_seq_len ({n_toks} > {max_seq_len})"

    return True, None
