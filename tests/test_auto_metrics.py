import pytest
from marathi_llm.eval.auto_metrics import compute_script_purity, compute_repetition_metrics


def test_script_purity_pure_marathi():
    marathi_text = "महाराष्ट्र हे भारतातील एक राज्य आहे."
    purity = compute_script_purity(marathi_text)
    assert purity > 98.0


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
