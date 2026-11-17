import pytest
import unicodedata
from marathi_llm.eval.auto_metrics import compute_script_purity, evaluate_generations
from marathi_llm.eval.stats import bootstrap_corpus_chrf, mcnemar_test
from marathi_llm.generate import prepare_prompt


def test_marathi_matras_preserved():
    """Verifies that Marathi words with vowel signs (matras) are preserved and recognized as Devanagari."""
    text = "भारताची राजधानी नवी दिल्ली आहे."
    chars = [c for c in text if c.isspace() or unicodedata.category(c)[0] in ("L", "M", "N")]
    cleaned = "".join(chars)
    assert cleaned == "भारताची राजधानी नवी दिल्ली आहे"
    purity = compute_script_purity(text)
    assert purity > 98.0


def test_evaluate_generations_flores_format():
    """Verifies evaluate_generations handles Flores records containing 'output' or 'generation'."""
    records = [
        {
            "id": "flores_1",
            "output": "सूर्य पूर्व दिशेला उगवतो.",
            "finish_reason": "eos",
            "n_new_tokens": 8,
        },
        {
            "id": "flores_2",
            "generation": "पृथ्वी सूर्याभोवती फिरते.",
            "finish_reason": "eos",
            "n_new_tokens": 7,
        },
    ]
    stats = evaluate_generations(records)
    assert stats["n_samples"] == 2
    assert stats["mean_script_purity"] > 90.0
    assert stats["eos_rate"] == 100.0
    assert stats["mean_new_tokens"] == 7.5


def test_prepare_prompt_formats():
    """Verifies prepare_prompt produces expected structure for chat, alpaca_v0, and raw formats."""
    raw = "महाराष्ट्राची राजधानी कोणती?"

    # alpaca_v0 format
    alpaca = prepare_prompt(raw, tokenizer=None, prompt_format="alpaca_v0")
    assert "### सूचना:" in alpaca
    assert "### प्रतिसाद:" in alpaca
    assert "खाली एक सूचना आहे" in alpaca
    assert raw in alpaca

    # raw format
    raw_res = prepare_prompt(raw, tokenizer=None, prompt_format="raw")
    assert raw_res == raw


def test_mcnemar_exact_binomial():
    """Verifies McNemar test correctly uses exact binomial test when discordant pairs < 25."""
    a = [True] * 5 + [False] * 5
    b = [True] * 8 + [False] * 2  # discordant: b01 = 3, b10 = 0 -> total 3 < 25
    res = mcnemar_test(a, b)
    assert res["exact"] is True
    assert res["total_discordant"] if "total_discordant" in res else (res["b01"] + res["b10"] == 3)
    assert res["p_value"] < 0.3


def test_bootstrap_corpus_chrf_basic():
    """Verifies bootstrap_corpus_chrf produces valid bounds."""
    hyps = ["सूर्य पूर्व दिशेला उगवतो.", "चंद्र रात्री चमकतो."]
    refs = ["सूर्य पूर्व दिशेला उगवतो.", "चंद्र रात्री चमकतो."]
    res = bootstrap_corpus_chrf(hyps, refs, n_bootstraps=50)
    assert pytest.approx(res["corpus_chrf"], 0.1) == 100.0
    assert pytest.approx(res["ci_low"], 0.1) == 100.0
    assert pytest.approx(res["ci_high"], 0.1) == 100.0


def test_generate_batch_checkpoint_resume(tmp_path):
    """Verifies that generate_batch resumes seamlessly from existing checkpoint file."""
    import json
    from unittest.mock import MagicMock
    import torch
    from marathi_llm.generate import generate_batch

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.chat_template = None
    mock_tokenizer.encode.return_value = [10, 20]
    mock_tokenizer.decode.return_value = "नमुना उत्तर"

    # Pre-populate a checkpoint file with 1 of 2 items
    ckpt_file = tmp_path / "checkpoint.jsonl"
    with open(ckpt_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "item_1", "output": "पहिला उत्तर", "finish_reason": "eos", "n_new_tokens": 5}) + "\n")

    items = [
        {"id": "item_1", "prompt": "प्रश्‍न १"},
        {"id": "item_2", "prompt": "प्रश्‍न २"},
    ]

    mock_model.generate.return_value = torch.tensor([[10, 20, 30]])
    mock_tokenizer.return_value = MagicMock(
        to=MagicMock(return_value={"input_ids": torch.tensor([[10, 20]])}),
        __getitem__=lambda self, key: torch.tensor([[10, 20]]),
    )

    results = generate_batch(
        mock_model,
        mock_tokenizer,
        items,
        batch_size=1,
        checkpoint_path=str(ckpt_file),
    )

    assert len(results) == 2
    assert results[0]["id"] == "item_1"
    assert results[0]["output"] == "पहिला उत्तर"  # restored from checkpoint
    assert results[1]["id"] == "item_2"
    # Checkpoint should be unlinked upon successful completion of all items
    assert not ckpt_file.exists()


def test_invalid_adapter_path_raises(monkeypatch):
    """Verifies D1: invalid adapter path raises RuntimeError instead of silently falling back to base."""
    from unittest.mock import MagicMock
    import transformers
    import peft
    from marathi_llm.generate import load_model_and_tokenizer

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: mock_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: mock_model)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", MagicMock(side_effect=OSError("Adapter directory or file not found")))

    with pytest.raises(RuntimeError, match="Failed to load LoRA adapter"):
        load_model_and_tokenizer(
            model_id="mock_model",
            adapter_path="nonexistent/fake-adapter-path-xyz",
            prequantized=True,
        )


def test_alpaca_v0_preamble_matches_dataset_translated_text():
    """Asserts that eval-time ALPACAV0_PROMPT_* templates match raw training preambles character-for-character."""
    from marathi_llm.generate import prepare_prompt
    # Case 1: with input
    inst = "विचित्र ओळखा."
    inp = "ट्विटर, इन्स्टाग्राम, टेलिव्हिजन"
    p1 = prepare_prompt(inst, tokenizer=None, prompt_format="alpaca_v0", input_text=inp)
    assert p1.startswith("खाली एक सूचना आहे जी एखाद्या कार्याचे वर्णन करते, पुढील संदर्भ प्रदान करणार्या इनपुटसह जोडली जाते. विनंती योग्य रितीने पूर्ण करणारा प्रतिसाद लिहा.")
    assert "### सूचना:\nविचित्र ओळखा.\n\n### इनपुट:\nट्विटर, इन्स्टाग्राम, टेलिव्हिजन\n\n### प्रतिसाद:\n" in p1

    # Case 2: without input
    p2 = prepare_prompt(inst, tokenizer=None, prompt_format="alpaca_v0", input_text="")
    assert p2.startswith("खाली एक सूचना आहे जी एखाद्या कार्याचे वर्णन करते. विनंती योग्य रितीने पूर्ण करणारा प्रतिसाद लिहा.")
    assert "### सूचना:\nविचित्र ओळखा.\n\n### प्रतिसाद:\n" in p2


