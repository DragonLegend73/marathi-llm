import pytest
import numpy as np
from marathi_llm.eval.stats import bootstrap_ci, paired_bootstrap_ci, mcnemar_test


def test_bootstrap_ci_basic():
    data = [1.0] * 50 + [0.0] * 50
    res = bootstrap_ci(data, n_bootstraps=500, ci=0.95, seed=42)
    assert res["mean"] == 0.5
    assert 0.35 < res["ci_low"] < 0.5
    assert 0.5 < res["ci_high"] < 0.65


def test_paired_bootstrap_ci_difference():
    # Model B consistently outscores Model A
    scores_a = [0.2, 0.3, 0.4, 0.5] * 25
    scores_b = [0.4, 0.5, 0.6, 0.7] * 25
    res = paired_bootstrap_ci(scores_a, scores_b, n_bootstraps=500, ci=0.95, seed=42)
    assert pytest.approx(res["diff_mean"], 0.01) == 0.2
    assert res["ci_low"] > 0.0
    assert res["is_significant"] is True


def test_mcnemar_test():
    # Both agree on all cases
    correct_a = [1, 1, 0, 0, 1]
    correct_b = [1, 1, 0, 0, 1]
    res = mcnemar_test(correct_a, correct_b)
    assert res["statistic"] == 0.0
    assert res["p_value"] == 1.0

    # Model B gets cases right where Model A is wrong
    correct_a = [0] * 20 + [1] * 30
    correct_b = [1] * 20 + [1] * 30
    res = mcnemar_test(correct_a, correct_b)
    assert res["b01"] == 20
    assert res["b10"] == 0
    assert res["p_value"] < 0.001
