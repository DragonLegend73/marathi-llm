import pytest
from marathi_llm.data.decontaminate import BenchmarkDecontaminator

def test_devanagari_words_import():
    from marathi_llm.data.text import words
    assert words("भारताची राजधानी कोणती आहे") == ["भारताची", "राजधानी", "कोणती", "आहे"]

def test_danda_is_not_part_of_word():
    from marathi_llm.data.text import words
    assert words("भारत माझा देश आहे।") == ["भारत", "माझा", "देश", "आहे"]
    assert words("हे वाक्य संपले॥") == ["हे", "वाक्य", "संपले"]

def test_three_or_four_shared_words_do_not_trigger_8gram():
    d = BenchmarkDecontaminator(ngram_size=8)
    # Eval string has 10 words:
    d.add_eval_text("भारताची राजधानी कोणती आहे आणि ती कुठे आहे हे सांगा", "test_source")
    # Candidate shares only 4 words ("भारताची राजधानी कोणती आहे")
    # Under \w+, "भारताची राजधानी कोणती आहे" produces 9 fragments and incorrectly triggers an 8-gram match!
    # Under true word tokenization, it is only 4 words and must not trigger.
    is_contam, tag = d.check_contamination("भारताची राजधानी कोणती आहे पण उत्तर वेगळे आहे")
    assert is_contam is False, f"Incorrectly flagged as contaminated: {tag}"

def test_zwj_normalization_matches():
    from marathi_llm.data.text import normalize_for_match, words
    t1 = "महाराष्ट्र\u200Dात"
    t2 = "महाराष्ट्रात"
    assert normalize_for_match(t1) == normalize_for_match(t2)
    assert words(t1) == words(t2)
