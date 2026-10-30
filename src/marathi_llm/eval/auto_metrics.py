"""Automatic quality and generation metrics for text evaluation in Marathi."""

import re
import unicodedata
from typing import List, Dict, Any
import numpy as np


def compute_script_purity(text: str) -> float:
    """
    Computes percentage of non-whitespace, non-punctuation characters
    belonging to the Devanagari Unicode block (U+0900 - U+097F).
    """
    if not text:
        return 0.0

    chars = [c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P")]
    if not chars:
        return 0.0

    devanagari_count = sum(1 for c in chars if "\u0900" <= c <= "\u097F")
    return float((devanagari_count / len(chars)) * 100.0)


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
    Evaluates a list of generation records.
    Each item must contain:
      - 'output': generated text string
      - optional 'finish_reason': 'eos' or 'length'
      - optional 'n_new_tokens': int
    """
    if not items:
        return {}

    purities = []
    distinct_ratios = []
    repetitive_flags = []
    eos_counts = 0
    token_lengths = []

    for item in items:
        out = item.get("output", "")
        purities.append(compute_script_purity(out))

        rep = compute_repetition_metrics(out)
        distinct_ratios.append(rep["distinct_4grams_ratio"])
        repetitive_flags.append(rep["is_repetitive"])

        finish_reason = item.get("finish_reason")
        if finish_reason == "eos":
            eos_counts += 1
        elif finish_reason is None:
            # Fallback heuristic: check if output didn't end abruptly without punctuation
            # or check if explicit eos was logged
            pass

        if "n_new_tokens" in item:
            token_lengths.append(item["n_new_tokens"])

    total = len(items)
    results = {
        "n_samples": total,
        "mean_script_purity": float(np.mean(purities)),
        "mean_distinct_4grams_ratio": float(np.mean(distinct_ratios)),
        "repetition_rate": float(np.mean(repetitive_flags) * 100.0),
        "eos_rate": float((eos_counts / total) * 100.0) if any("finish_reason" in it for it in items) else None,
    }

    if token_lengths:
        results["mean_new_tokens"] = float(np.mean(token_lengths))

    return results
