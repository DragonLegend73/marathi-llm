"""Unit tests verifying BOS token count and prompt templates across all Stage A candidate tokenizers."""

import pytest
from transformers import AutoTokenizer
from marathi_llm.generate import prepare_prompt, PLAIN_PROMPT, format_plain_prompt

STAGE_A_MODELS = [
    "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "sarvamai/sarvam-1",
]


@pytest.mark.parametrize("model_id", STAGE_A_MODELS)
def test_exactly_one_bos_in_encoded_prompt(model_id: str):
    """
    Verifies that when prepare_prompt and tokenization occur:
    1. For chat format (with chat template), exactly one BOS is present if the model uses BOS.
    2. For plain format (or template-less models where chat_template is empty),
       exactly one BOS is present if the model uses BOS.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    bos_id = tok.bos_token_id
    raw_prompt = "भारताची राजधानी कोणती आहे?"

    # Case 1: Chat format (with native chat template if available)
    chat_prompt = prepare_prompt(raw_prompt, tok, prompt_format="chat")
    uses_template = bool(getattr(tok, "chat_template", None))
    add_special_chat = not uses_template
    enc_chat = tok.encode(chat_prompt, add_special_tokens=add_special_chat)

    if bos_id is not None:
        assert enc_chat.count(bos_id) == 1, (
            f"{model_id} (chat format) had {enc_chat.count(bos_id)} BOS tokens (expected 1)"
        )
    else:
        # Models without BOS (e.g. Qwen) use turn delimiters
        assert enc_chat.count(bos_id) == 0

    # Case 2: Plain format (instruction wrapper)
    plain_prompt = prepare_prompt(raw_prompt, tok, prompt_format="plain")
    assert plain_prompt == format_plain_prompt(raw_prompt)
    enc_plain = tok.encode(plain_prompt, add_special_tokens=True)

    if bos_id is not None:
        assert enc_plain.count(bos_id) == 1, (
            f"{model_id} (plain format) had {enc_plain.count(bos_id)} BOS tokens (expected 1)"
        )
        assert enc_plain[0] == bos_id, f"{model_id} first token should be BOS ({bos_id})"

    # Case 3: Template-less fallback: chat_template is None but prompt_format="chat"
    tok.chat_template = None
    fallback_prompt = prepare_prompt(raw_prompt, tok, prompt_format="chat")
    # Must automatically fall back to plain prompt wrapper
    assert fallback_prompt == format_plain_prompt(raw_prompt)
    uses_template_fallback = bool(getattr(tok, "chat_template", None))
    add_special_fallback = not uses_template_fallback
    assert add_special_fallback is True

    enc_fallback = tok.encode(fallback_prompt, add_special_tokens=add_special_fallback)
    if bos_id is not None:
        assert enc_fallback.count(bos_id) == 1, (
            f"{model_id} (fallback plain format) had {enc_fallback.count(bos_id)} BOS tokens (expected 1)"
        )
        assert enc_fallback[0] == bos_id
