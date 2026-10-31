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
