import json
from pathlib import Path
import pytest


def test_split_manifest_no_overlap():
    splits_dir = Path("data/splits")
    train_path = splits_dir / "s0_train_ids.json"
    dev_path = splits_dir / "s0_dev_ids.json"
    test_path = splits_dir / "s0_test_ids.json"

    assert train_path.exists(), "Train split manifest missing"
    assert dev_path.exists(), "Dev split manifest missing"
    assert test_path.exists(), "Test split manifest missing"

    with open(train_path, "r", encoding="utf-8") as f:
        train_ids = set(json.load(f)["row_ids"])
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_ids = set(json.load(f)["row_ids"])
    with open(test_path, "r", encoding="utf-8") as f:
        test_ids = set(json.load(f)["row_ids"])

    # Verify counts
    assert len(dev_ids) == 500
    assert len(test_ids) == 1000
    assert len(train_ids) > 50000

    # Verify zero disjoint intersection
    assert len(train_ids.intersection(dev_ids)) == 0, "Train and dev splits overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and test splits overlap!"
    assert len(dev_ids.intersection(test_ids)) == 0, "Dev and test splits overlap!"
