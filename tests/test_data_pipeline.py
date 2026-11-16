import pytest
from marathi_llm.data.clean import normalize_marathi_text, validate_example
from marathi_llm.data.dedup import DatasetDeduplicator
from marathi_llm.data.decontaminate import BenchmarkDecontaminator
from marathi_llm.data.stats import FunnelTracker


def test_normalize_marathi_text():
    # Zero width chars, HTML tags, irregular whitespace
    raw = "<b>सूर्य</b>\u200b\u200c पूर्व दिशेला   उगवतो.\n\n\n\nआणि पश्चिमेला मावळतो."
    cleaned = normalize_marathi_text(raw)
    assert "\u200b" not in cleaned
    assert "\u200c" not in cleaned
    assert "<b>" not in cleaned
    assert "  " not in cleaned
    assert "सूर्य पूर्व दिशेला उगवतो." in cleaned


def test_validate_example_quality_rules():
    # 1. Valid pure Marathi
    valid, reason = validate_example(
        instruction="महाराष्ट्राची राजधानी कोणती आहे?",
        response="महाराष्ट्राची राजधानी मुंबई आहे.",
    )
    assert valid is True
    assert reason is None

    # 2. English / Latin leakage rejected
    valid, reason = validate_example(
        instruction="What is the capital?",
        response="The capital of Maharashtra is Mumbai.",
    )
    assert valid is False
    assert "script_purity" in reason or "latin" in reason

    # 3. Degenerate repetition rejected
    valid, reason = validate_example(
        instruction="सांगा.",
        response="हे एक वाक्य आहे. " * 20,
    )
    assert valid is False
    assert "repetition" in reason

    # 4. Template artifact leak rejected
    valid, reason = validate_example(
        instruction="प्रश्न",
        response="उत्तर <|im_end|>",
    )
    assert valid is False
    assert "artifact" in reason


def test_dataset_deduplicator():
    dedup = DatasetDeduplicator(threshold=0.8)

    # First item
    t1 = "महाराष्ट्राची राजधानी मुंबई ही भारताची आर्थिक राजधानी आहे आणि तिथे बॉम्बे स्टॉक एक्सचेंज आहे. हे शहर अरबी समुद्राच्या किनाऱ्यावर वसलेले एक अत्यंत मोठे महानगर आहे."
    assert dedup.is_duplicate(t1, key_id="1") is False

    # Exact duplicate
    assert dedup.is_duplicate(t1, key_id="2") is True

    # Near twin (paraphrase)
    t2 = "महाराष्ट्राची राजधानी मुंबई ही भारताची आर्थिक राजधानी आहे आणि तिथे बॉम्बे स्टॉक एक्सचेंज स्थित आहे. हे शहर अरबी समुद्राच्या किनाऱ्यावर वसलेले एक अत्यंत मोठे महानगर आहे."
    assert dedup.is_duplicate(t2, key_id="3") is True

    # Completely different item
    t3 = "सूर्य हा आपल्या सूर्यमालेचा केंद्रबिंदू असलेला एक तारा आहे."
    assert dedup.is_duplicate(t3, key_id="4") is False


def test_benchmark_decontaminator():
    decontam = BenchmarkDecontaminator(minhash_threshold=0.7, ngram_size=8)
    eval_text = "भारताची राजधानी नवी दिल्ली आहे आणि तिथे संसद भवन आहे."
    decontam.add_eval_text(eval_text, source_tag="milu_test")

    # Candidate with 8-word exact match
    cand_contaminated = "मला सांगा भारताची राजधानी नवी दिल्ली आहे आणि तिथे संसद भवन आहे का?"
    is_contam, tag = decontam.check_contamination(cand_contaminated)
    assert is_contam is True
    assert "8gram_match" in tag

    # Clean candidate
    cand_clean = "पुणे हे महाराष्ट्रातील एक महत्त्वाचे ऐतिहासिक व सांस्कृतिक शहर आहे."
    is_contam, tag = decontam.check_contamination(cand_clean)
    assert is_contam is False
    assert tag is None


def test_funnel_tracker():
    tracker = FunnelTracker("test_source")
    tracker.counts["raw_ingested"] = 100
    tracker.log_rejection("clean", "low_purity")
    tracker.log_rejection("dedup", "duplicate")
    assert tracker.counts["failed_clean"] == 1
    assert tracker.counts["failed_dedup"] == 1
    assert tracker.rejection_details["low_purity"] == 1
