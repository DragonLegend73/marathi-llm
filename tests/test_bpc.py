"""Tests for held-out BPC data loading and functions."""

import json
from pathlib import Path
from marathi_llm.eval.bpc import load_bpc_reference_texts


def test_load_bpc_reference_texts_aya(tmp_path):
    # Test loading from Aya format
    p = tmp_path / "bpc_responses.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "aya_0", "response": "मराठी ही समृद्ध भाषा आहे."}) + "\n")
        f.write(json.dumps({"id": "aya_1", "response": "पुणे हे विद्येचे माहेरघर आहे."}) + "\n")

    items = load_bpc_reference_texts(source="aya", custom_path=str(p))
    assert len(items) == 2
    assert items[0]["id"] == "aya_0"
    assert items[0]["text"] == "मराठी ही समृद्ध भाषा आहे."

    # Test limit
    limited = load_bpc_reference_texts(source="aya", limit=1, custom_path=str(p))
    assert len(limited) == 1
    assert limited[0]["id"] == "aya_0"
