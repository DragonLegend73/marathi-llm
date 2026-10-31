"""Statistical significance and bootstrap confidence interval helpers for model comparisons."""

from typing import List, Tuple, Dict, Any
import numpy as np


def bootstrap_ci(
    values: List[float] | np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes empirical mean and non-parametric percentile bootstrap confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_bootstraps, len(arr)))
    boot_means = np.mean(arr[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))

    return {
        "mean": float(np.mean(arr)),
        "ci_low": low,
        "ci_high": high,
    }


def bootstrap_corpus_chrf(
    hypotheses: List[str],
    references: List[str],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes corpus-level chrF++ and its non-parametric bootstrap confidence interval."""
    import sacrebleu

    if not hypotheses or len(hypotheses) != len(references):
        return {"corpus_chrf": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    n = len(hypotheses)
    base_score = float(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score)

    rng = np.random.default_rng(seed)
    boot_scores = []
    hyps_arr = np.array(hypotheses, dtype=object)
    refs_arr = np.array(references, dtype=object)

    for _ in range(n_bootstraps):
        idxs = rng.integers(0, n, size=n)
        sub_hyps = hyps_arr[idxs].tolist()
        sub_refs = refs_arr[idxs].tolist()
        s = sacrebleu.corpus_chrf(sub_hyps, [sub_refs], word_order=2).score
        boot_scores.append(s)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_scores, 100 * alpha))
    high = float(np.percentile(boot_scores, 100 * (1.0 - alpha)))

    return {
        "corpus_chrf": base_score,
        "ci_low": low,
        "ci_high": high,
    }


def paired_bootstrap_corpus_chrf(
    hypotheses_a: List[str],
    hypotheses_b: List[str],
    references: List[str],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired difference in corpus-level chrF++ (B - A) using shared resample indices.
    """
    import sacrebleu

    if len(hypotheses_a) != len(hypotheses_b) or len(hypotheses_a) != len(references):
        raise ValueError("Lengths of hypotheses_a, hypotheses_b, and references must match.")

    n = len(references)
    if n == 0:
        return {"diff_corpus_chrf": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "is_significant": False}

    score_a = float(sacrebleu.corpus_chrf(hypotheses_a, [references], word_order=2).score)
    score_b = float(sacrebleu.corpus_chrf(hypotheses_b, [references], word_order=2).score)
    diff = score_b - score_a

    rng = np.random.default_rng(seed)
    hyps_a_arr = np.array(hypotheses_a, dtype=object)
    hyps_b_arr = np.array(hypotheses_b, dtype=object)
    refs_arr = np.array(references, dtype=object)

    boot_diffs = []
    for _ in range(n_bootstraps):
        idxs = rng.integers(0, n, size=n)
        s_a = sacrebleu.corpus_chrf(hyps_a_arr[idxs].tolist(), [refs_arr[idxs].tolist()], word_order=2).score
        s_b = sacrebleu.corpus_chrf(hyps_b_arr[idxs].tolist(), [refs_arr[idxs].tolist()], word_order=2).score
        boot_diffs.append(s_b - s_a)

    boot_diffs = np.array(boot_diffs)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_diffs, 100 * alpha))
    high = float(np.percentile(boot_diffs, 100 * (1.0 - alpha)))

    centered = boot_diffs - diff
    p_value = float(np.mean(np.abs(centered) >= np.abs(diff)))
    is_significant = (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)

    return {
        "diff_corpus_chrf": diff,
        "score_a": score_a,
        "score_b": score_b,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "is_significant": is_significant,
    }


def paired_bootstrap_ci(
    scores_a: List[float] | np.ndarray,
    scores_b: List[float] | np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired difference bootstrap CI: diff = scores_b - scores_a.
    Both systems must be evaluated on identical items in identical order.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match for paired bootstrap: {len(a)} vs {len(b)}")

    diffs = b - a
    diff_mean = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(n_bootstraps, len(diffs)))
    boot_diffs = np.mean(diffs[indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_diffs, 100 * alpha))
    high = float(np.percentile(boot_diffs, 100 * (1.0 - alpha)))

    # Two-sided empirical p-value for hypothesis that diff == 0
    # centered around 0
    centered = boot_diffs - diff_mean
    p_value = float(np.mean(np.abs(centered) >= np.abs(diff_mean)))

    is_significant = (low > 0.0 and high > 0.0) or (low < 0.0 and high < 0.0)

    return {
        "diff_mean": diff_mean,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "is_significant": is_significant,
    }


def mcnemar_test(
    correct_a: List[int | bool] | np.ndarray,
    correct_b: List[int | bool] | np.ndarray,
    continuity_correction: bool = True,
) -> Dict[str, float]:
    """
    McNemar's test for paired binary (0/1 or True/False) outcomes.
    Tests null hypothesis that both classifiers have equal error rates.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match: {len(a)} vs {len(b)}")

    # Contingency table
    # b01: A wrong (False), B right (True)
    # b10: A right (True), B wrong (False)
    b01 = int(np.sum((~a) & b))
    b10 = int(np.sum(a & (~b)))

    total_discordant = b01 + b10
    if total_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b01": 0, "b10": 0}

    # For small sample sizes (discordant pairs < 25), use exact binomial test
    if total_discordant < 25:
        try:
            from scipy.stats import binomtest
            res = binomtest(b01, n=total_discordant, p=0.5, alternative="two-sided")
            p_val = float(res.pvalue)
            stat = float(min(b01, b10))
            return {
                "statistic": stat,
                "p_value": p_val,
                "b01": b01,
                "b10": b10,
                "exact": True,
            }
        except ImportError:
            pass

    diff = abs(b01 - b10)
    if continuity_correction:
        diff = max(0.0, diff - 1.0)

    stat = (diff ** 2) / total_discordant

    # p-value from chi2 with 1 degree of freedom: P(chi2 >= stat)
    try:
        from scipy.stats import chi2
        p_val = float(chi2.sf(stat, df=1))
    except ImportError:
        import math
        z = math.sqrt(stat)
        p_val = math.erfc(z / math.sqrt(2))

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "b01": b01,
        "b10": b10,
        "exact": False,
    }
