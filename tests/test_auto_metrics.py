import pytest
from marathi_llm.eval.auto_metrics import compute_script_purity, compute_repetition_metrics


def test_script_purity_with_numbers_and_punctuation():
    # Marathi with numbers and punctuation should have 100% letter/mark purity
    text = "भारताची राजधानी १००० रुपये (२०२६ रोजी)."
    purity = compute_script_purity(text)
    assert purity == 100.0


def test_evaluate_generations_schema():
    from marathi_llm.eval.auto_metrics import evaluate_generations
    items = [
        {"output": "महाराष्ट्राची राजधानी मुंबई आहे.", "finish_reason": "eos", "n_new_tokens": 12},
        {"output": "This is leaked english response.", "finish_reason": "eos", "n_new_tokens": 8},
        {"output": "", "finish_reason": "length", "n_new_tokens": 0},
    ]
    res = evaluate_generations(items)
    assert res["n_samples"] == 3
    assert 0.0 < res["script_purity"] < 100.0
    assert "lang_id_mr" in res
    assert "repeat_4gram" in res
    assert "distinct_4grams_ratio" in res
    assert "eos_rate" in res
    assert "english_leak_ratio" in res
    assert "empty_output_rate" in res
    assert res["empty_output_rate"] == pytest.approx(33.33, abs=0.1)



def test_script_purity_pure_english():
    english_text = "This is an English sentence."
    purity = compute_script_purity(english_text)
    assert purity == 0.0


def test_script_purity_code_mixed():
    mixed_text = "मी Python programming शिकत आहे."
    purity = compute_script_purity(mixed_text)
    assert 25.0 < purity < 80.0


def test_repetition_metrics():
    # Degenerate repetitive text
    repeated_phrase = "हे एक सुंदर वाक्य आहे " * 10
    res = compute_repetition_metrics(repeated_phrase)
    assert res["is_repetitive"] == 1.0
    assert res["max_ngram_repeat"] >= 4

    # Natural text without 4-gram repeats
    natural_text = "सौर पेशी सिलिकॉन पासून बनतात आणि त्या सूर्यप्रकाशाचे ऊर्जेत रूपांतर करतात."
    res_nat = compute_repetition_metrics(natural_text)
    assert res_nat["is_repetitive"] == 0.0
