"""Paired bootstrap comparison routines for MILU, Flores, and BPC."""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import sacrebleu
from scipy.stats import binomtest


def paired_milu_bootstrap(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired bootstrap & exact McNemar test on MILU accuracy (B - A).
    Asserts exact matching record IDs across both arms.
    """
    assert len(recs_a) == len(recs_b), f"MILU length mismatch: {len(recs_a)} vs {len(recs_b)}"
    for idx, (ra, rb) in enumerate(zip(recs_a, recs_b)):
        id_a = ra.get("id", str(idx))
        id_b = rb.get("id", str(idx))
        assert id_a == id_b, f"MILU ID mismatch at index {idx}: {id_a} vs {id_b}"

    a = np.array([1.0 if r.get("correct") else 0.0 for r in recs_a], dtype=np.float64)
    b = np.array([1.0 if r.get("correct") else 0.0 for r in recs_b], dtype=np.float64)
    n = len(a)

    acc_a = float(a.mean() * 100.0)
    acc_b = float(b.mean() * 100.0)
    delta = acc_b - acc_a

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = (b[idx].mean(axis=1) - a[idx].mean(axis=1)) * 100.0
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    # Exact McNemar on discordant pairs
    n_01 = int(((a == 0) & (b == 1)).sum())
    n_10 = int(((a == 1) & (b == 0)).sum())
    p_mcnemar = (
        float(binomtest(min(n_01, n_10), n_01 + n_10, 0.5).pvalue)
        if (n_01 + n_10) > 0
        else 1.0
    )

    return {
        "n": n,
        "a": round(acc_a, 2),
        "b": round(acc_b, 2),
        "delta": round(delta, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "mcnemar_b_wins": n_01,
        "mcnemar_a_wins": n_10,
        "mcnemar_p": round(p_mcnemar, 4),
    }


def paired_flores_bootstrap(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    limit: Optional[int] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired corpus chrF++ bootstrap (B - A) on identical resampled sentence indices.
    Truncates both arms to limit (if specified) and asserts exact record ID matching.
    """
    if limit is not None:
        recs_a = recs_a[:limit]
        recs_b = recs_b[:limit]

    assert len(recs_a) == len(recs_b), f"Flores length mismatch: {len(recs_a)} vs {len(recs_b)}"
    for idx, (ra, rb) in enumerate(zip(recs_a, recs_b)):
        id_a = ra.get("id", str(idx))
        id_b = rb.get("id", str(idx))
        assert id_a == id_b, f"Flores ID mismatch at index {idx}: {id_a} vs {id_b}"

    n = len(recs_a)
    hyps_a = [r.get("output", r.get("generation", "")) for r in recs_a]
    hyps_b = [r.get("output", r.get("generation", "")) for r in recs_b]
    refs = [r["target"] for r in recs_a]

    chrf = sacrebleu.CHRF(word_order=2)
    score_a = float(chrf.corpus_score(hyps_a, [refs]).score)
    score_b = float(chrf.corpus_score(hyps_b, [refs]).score)
    delta = score_b - score_a

    stats_a = np.array(chrf._extract_corpus_statistics(hyps_a, [refs]))
    stats_b = np.array(chrf._extract_corpus_statistics(hyps_b, [refs]))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sa = chrf._compute_score_from_stats(stats_a[idx].sum(axis=0).tolist()).score
        sb = chrf._compute_score_from_stats(stats_b[idx].sum(axis=0).tolist()).score
        diffs[i] = sb - sa

    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {
        "n": n,
        "a": round(score_a, 2),
        "b": round(score_b, 2),
        "delta": round(delta, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


def paired_bpc_bootstrap(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    limit: Optional[int] = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Paired corpus BPC bootstrap (B - A; negative delta means B is better).
    Asserts exact matching record IDs across both arms.
    """
    if limit is not None:
        recs_a = recs_a[:limit]
        recs_b = recs_b[:limit]

    assert len(recs_a) == len(recs_b), f"BPC length mismatch: {len(recs_a)} vs {len(recs_b)}"
    for idx, (ra, rb) in enumerate(zip(recs_a, recs_b)):
        id_a = ra.get("id", str(idx))
        id_b = rb.get("id", str(idx))
        assert id_a == id_b, f"BPC ID mismatch at index {idx}: {id_a} vs {id_b}"

    n = len(recs_a)
    nll_a = np.array([r["nll"] for r in recs_a], dtype=np.float64)
    nll_b = np.array([r["nll"] for r in recs_b], dtype=np.float64)
    chars = np.array([r["chars"] for r in recs_a], dtype=np.float64)
    ln2 = math.log(2.0)

    bpc_a = float(nll_a.sum() / (ln2 * chars.sum()))
    bpc_b = float(nll_b.sum() / (ln2 * chars.sum()))
    delta = bpc_b - bpc_a

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    c_sum = chars[idx].sum(axis=1)
    diffs = (nll_b[idx].sum(axis=1) - nll_a[idx].sum(axis=1)) / (ln2 * c_sum)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    return {
        "n": n,
        "a": round(bpc_a, 4),
        "b": round(bpc_b, 4),
        "delta": round(delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }
