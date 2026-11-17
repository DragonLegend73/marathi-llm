"""Test ensuring evaluation-time prompt templates match training data preambles character-for-character."""

import os
from datasets import load_dataset
from marathi_llm.generate import ALPACAV0_PROMPT_NO_INPUT, ALPACAV0_PROMPT_WITH_INPUT

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def test_templates_match_training_data():
    ds = load_dataset("rachittshah/alpaca-marahti", split="train")
    no_in = None
    with_in = None
    for row in ds:
        t = row["translated_text"]
        if "### इनपुट:" in t and with_in is None:
            with_in = t
        elif "### इनपुट:" not in t and no_in is None:
            no_in = t
        if no_in and with_in:
            break
    assert no_in.startswith(ALPACAV0_PROMPT_NO_INPUT.split("{")[0])
    assert with_in.startswith(ALPACAV0_PROMPT_WITH_INPUT.split("{")[0])
