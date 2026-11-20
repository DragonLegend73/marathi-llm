"""Tests for paired bootstrap comparisons."""

import pytest
from marathi_llm.eval.paired import (
    paired_milu_bootstrap,
    paired_flores_bootstrap,
    paired_bpc_bootstrap,
)


def test_paired_milu_bootstrap():
    recs_a = [{"id": f"q_{i}", "correct": i % 2 == 0} for i in range(20)]
    recs_b = [{"id": f"q_{i}", "correct": i % 3 == 0} for i in range(20)]

    res = paired_milu_bootstrap(recs_a, recs_b, n_resamples=50)
    assert res["n"] == 20
    assert "delta" in res
    assert "ci95" in res
    assert "mcnemar_p" in res


def test_paired_milu_id_mismatch():
    recs_a = [{"id": "q_1", "correct": True}]
    recs_b = [{"id": "q_2", "correct": True}]
    with pytest.raises(AssertionError):
        paired_milu_bootstrap(recs_a, recs_b)


def test_paired_flores_bootstrap():
    recs_a = [
        {"id": f"f_{i}", "output": "भारताची राजधानी नवी दिल्ली आहे.", "target": "नवी दिल्ली ही भारताची राजधानी आहे."}
        for i in range(10)
    ]
    recs_b = [
        {"id": f"f_{i}", "output": "नवी दिल्ली ही राजधानी आहे.", "target": "नवी दिल्ली ही भारताची राजधानी आहे."}
        for i in range(10)
    ]
    res = paired_flores_bootstrap(recs_a, recs_b, limit=5, n_resamples=50)
    assert res["n"] == 5
    assert "delta" in res
    assert "ci95" in res


def test_paired_bpc_bootstrap():
    recs_a = [{"id": f"b_{i}", "nll": 10.0 + i, "chars": 15 + i} for i in range(10)]
    recs_b = [{"id": f"b_{i}", "nll": 9.5 + i, "chars": 15 + i} for i in range(10)]
    res = paired_bpc_bootstrap(recs_a, recs_b, n_resamples=50)
    assert res["n"] == 10
    assert "delta" in res
    assert "ci95" in res
