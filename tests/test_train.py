"""Unit tests for config-driven QLoRA training helpers."""

import csv
import json
from pathlib import Path
from transformers import AutoTokenizer

from marathi_llm.train import (
    ALL_LINEAR_MODULES,
    ATTENTION_ONLY_MODULES,
    build_prompt_completion_records,
    filter_by_max_length,
    find_latest_checkpoint,
    format_user_content,
    infer_eval_data_path,
    resolve_target_modules,
    write_loss_csv,
)


def test_resolve_target_modules():
    assert resolve_target_modules("all-linear") == ALL_LINEAR_MODULES
    assert resolve_target_modules("attention-only") == ATTENTION_ONLY_MODULES
    assert resolve_target_modules("q_proj, v_proj") == ["q_proj", "v_proj"]
    assert resolve_target_modules(["q_proj", "o_proj"]) == ["q_proj", "o_proj"]


def test_format_user_content():
    assert format_user_content("प्रश्न", "") == "प्रश्न"
    assert format_user_content("प्रश्न", "   ") == "प्रश्न"
    assert format_user_content("प्रश्न", "संदर्भ") == "प्रश्न\n\nसंदर्भ"


def test_build_prompt_completion_records():
    rows = [
        {"instruction": "मराठीत उत्तर द्या.", "input": "", "response": "नमस्कार!"},
        {"instruction": "सारांश लिहा.", "input": "हा एक उतारा आहे.", "response": "थोडक्यात सारांश."},
        {"instruction": "", "input": "", "response": "skipped"},
    ]

    chat_recs = build_prompt_completion_records(rows, has_chat_template=True)
    assert len(chat_recs) == 2
    assert chat_recs[0]["prompt"] == [{"role": "user", "content": "मराठीत उत्तर द्या."}]
    assert chat_recs[0]["completion"] == [{"role": "assistant", "content": "नमस्कार!"}]
    assert chat_recs[1]["prompt"][0]["content"] == "सारांश लिहा.\n\nहा एक उतारा आहे."

    plain_recs = build_prompt_completion_records(rows, has_chat_template=False)
    assert len(plain_recs) == 2
    assert "### सूचना:" in plain_recs[0]["prompt"]
    assert plain_recs[0]["completion"] == "नमस्कार!"


def test_filter_by_max_length_with_qwen_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", local_files_only=True
    )
    rows = [
        {"instruction": "नमस्कार", "input": "", "response": "नमस्कार! कसे आहात?"},
        {"instruction": "मराठी भाषा " * 200, "input": "", "response": "विस्तृत उत्तर " * 200},
    ]
    recs = build_prompt_completion_records(rows, has_chat_template=True)
    kept, stats = filter_by_max_length(recs, tok, max_seq_len=128)
    assert len(kept) == 1
    assert stats["total"] == 2
    assert stats["kept"] == 1
    assert stats["dropped"] == 1
    assert stats["pct_examples_kept"] == 50.0


def test_find_latest_checkpoint(tmp_path: Path):
    assert find_latest_checkpoint(tmp_path / "missing") is None
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-250").mkdir()
    (tmp_path / "not-a-ckpt").mkdir()
    assert find_latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-500")


def test_write_loss_csv(tmp_path: Path):
    history = [
        {"step": 25, "epoch": 0.05, "loss": 1.54321, "learning_rate": 1e-4},
        {"step": 50, "epoch": 0.10, "eval_loss": 1.32109},
        {"step": 50, "train_runtime": 12.0},  # ignored
    ]
    out = tmp_path / "loss.csv"
    write_loss_csv(history, out)
    with open(out, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["step"] == "25"
    assert rows[0]["loss"] == "1.54321"
    assert rows[1]["eval_loss"] == "1.32109"


def test_infer_eval_data_path(tmp_path: Path):
    train_p = tmp_path / "s0_alpaca_12k.jsonl"
    dev_p = tmp_path / "s0_alpaca_dev500.jsonl"
    train_p.write_text("{}\n")
    assert infer_eval_data_path(str(train_p)) is None
    dev_p.write_text("{}\n")
    assert infer_eval_data_path(str(train_p)) == str(dev_p)
