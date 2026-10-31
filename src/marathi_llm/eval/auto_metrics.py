"""Automatic quality and generation metrics for text evaluation in Marathi."""

import re
import unicodedata
from typing import List, Dict, Any, Optional
import numpy as np

# Cache lingua detector for language-ID
_LINGUA_DETECTOR = None


def get_lingua_detector():
    """Lazily initializes lingua LanguageDetector for Marathi, Hindi, English."""
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            _LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(
                Language.MARATHI, Language.HINDI, Language.ENGLISH
            ).build()
        except ImportError:
            _LINGUA_DETECTOR = False
    return _LINGUA_DETECTOR


def compute_script_purity(text: str) -> float:
    """
    Computes percentage of letters and combining marks (categories L and M)
    that belong to the Devanagari Unicode block (U+0900 - U+097F).
    Digits, punctuation, whitespace, and symbols are excluded from the denominator.
    """
    if not text:
        return 0.0

    chars = [c for c in text if unicodedata.category(c).startswith(("L", "M"))]
    if not chars:
        return 0.0

    devanagari_count = sum(1 for c in chars if "\u0900" <= c <= "\u097F")
    return float((devanagari_count / len(chars)) * 100.0)


def compute_english_leak_ratio(text: str) -> float:
    """
    Computes proportion of letters and marks that are Latin characters (A-Z, a-z).
    """
    if not text:
        return 0.0

    chars = [c for c in text if unicodedata.category(c).startswith(("L", "M"))]
    if not chars:
        return 0.0

    latin_count = sum(1 for c in chars if ("a" <= c <= "z" or "A" <= c <= "Z"))
    return float(latin_count / len(chars))


def detect_is_marathi(text: str) -> Optional[bool]:
    """
    Detects if text is Marathi using lingua-language-detector.
    Returns True if detected as Marathi, False if other language, None if empty or detector unavailable.
    """
    if not text or not text.strip():
        return None

    detector = get_lingua_detector()
    if not detector:
        return None

    try:
        from lingua import Language
        res = detector.detect_language_of(text.strip())
        return res == Language.MARATHI
    except Exception:
        return None


def extract_ngrams(tokens: List[str], n: int = 4) -> List[tuple]:
    """Extracts n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_repetition_metrics(text: str) -> Dict[str, float]:
    """
    Computes 4-gram repetition metrics:
    - distinct_4grams_ratio: unique 4-grams / total 4-grams
    - max_ngram_repeat: maximum frequency of any 4-gram
    - is_repetitive: 1.0 if max_ngram_repeat >= 4 else 0.0
    """
    words = text.split()
    if len(words) < 4:
        return {
            "distinct_4grams_ratio": 1.0,
            "max_ngram_repeat": 1.0 if words else 0.0,
            "is_repetitive": 0.0,
        }

    ngrams = extract_ngrams(words, n=4)
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    from collections import Counter
    counts = Counter(ngrams)
    max_repeat = counts.most_common(1)[0][1] if counts else 0

    return {
        "distinct_4grams_ratio": float(unique_ngrams / total_ngrams),
        "max_ngram_repeat": float(max_repeat),
        "is_repetitive": 1.0 if max_repeat >= 4 else 0.0,
    }


def evaluate_generations(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a list of generation records according to Appendix B schema:
      - script_purity (% letters/marks in Devanagari)
      - lang_id_mr (% non-empty outputs detected as Marathi)
      - repeat_4gram (% repetitive outputs)
      - distinct_4grams_ratio (mean unique / total 4-grams)
      - eos_rate (% terminated with EOS)
      - mean_len_tokens (mean new tokens generated)
      - english_leak_ratio (mean share of Latin letters)
      - empty_output_rate (% empty / whitespace-only outputs)
    """
    if not items:
        return {}

    purities = []
    distinct_ratios = []
    repetitive_flags = []
    eos_counts = 0
    token_lengths = []
    english_leaks = []
    empty_flags = []
    mr_langs = []

    for item in items:
        out = item.get("output") or item.get("generation") or ""
        is_empty = 1.0 if not out.strip() else 0.0
        empty_flags.append(is_empty)

        purities.append(compute_script_purity(out))
        english_leaks.append(compute_english_leak_ratio(out))

        rep = compute_repetition_metrics(out)
        distinct_ratios.append(rep["distinct_4grams_ratio"])
        repetitive_flags.append(rep["is_repetitive"])

        is_mr = detect_is_marathi(out)
        if is_mr is not None:
            mr_langs.append(1.0 if is_mr else 0.0)

        finish_reason = item.get("finish_reason")
        if finish_reason == "eos":
            eos_counts += 1

        if "n_new_tokens" in item:
            token_lengths.append(item["n_new_tokens"])

    total = len(items)
    results = {
        "n_samples": total,
        "script_purity": float(np.mean(purities)),
        "mean_script_purity": float(np.mean(purities)),
        "distinct_4grams_ratio": float(np.mean(distinct_ratios)),
        "mean_distinct_4grams_ratio": float(np.mean(distinct_ratios)),
        "repeat_4gram": float(np.mean(repetitive_flags) * 100.0),
        "repetition_rate": float(np.mean(repetitive_flags) * 100.0),
        "eos_rate": float((eos_counts / total) * 100.0) if any("finish_reason" in it for it in items) else 0.0,
        "english_leak_ratio": float(np.mean(english_leaks)),
        "empty_output_rate": float(np.mean(empty_flags) * 100.0),
        "lang_id_mr": float(np.mean(mr_langs) * 100.0) if mr_langs else 0.0,
    }

    if token_lengths:
        results["mean_new_tokens"] = float(np.mean(token_lengths))
        results["mean_len_tokens"] = float(np.mean(token_lengths))

    return results
