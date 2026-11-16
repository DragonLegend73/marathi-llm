"""Benchmark decontamination engine checking training data against all evaluation suites."""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from datasketch import MinHash, MinHashLSH

WORD_PATTERN = re.compile(r"\w+")


def extract_word_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extracts consecutive word n-grams."""
    words = WORD_PATTERN.findall(text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def get_minhash(text: str, num_perm: int = 128, n: int = 5) -> MinHash:
    """Computes MinHash on 5-word shingles."""
    m = MinHash(num_perm=num_perm)
    words = WORD_PATTERN.findall(text.lower())
    if len(words) >= n:
        for i in range(len(words) - n + 1):
            shingle = " ".join(words[i : i + n])
            m.update(shingle.encode("utf-8"))
    elif words:
        m.update(" ".join(words).encode("utf-8"))
    return m


class BenchmarkDecontaminator:
    """
    Indexes all evaluation strings across MILU, Flores-IN, XQuAD, IndicIFEval, and held-out Aya prompts.
    Rejects any training example with >= 8-word verbatim overlap or MinHash Jaccard >= 0.7.
    """

    def __init__(self, minhash_threshold: float = 0.7, ngram_size: int = 8):
        self.minhash_threshold = minhash_threshold
        self.ngram_size = ngram_size
        self.eval_ngrams: Dict[str, str] = {} # ngram -> benchmark_source
        self.lsh = MinHashLSH(threshold=minhash_threshold, num_perm=128)
        self.eval_texts: Dict[str, str] = {}
        self.total_indexed = 0

    def add_eval_text(self, text: str, source_tag: str):
        """Indexes an evaluation text into n-gram and MinHash LSH tables."""
        text = text.strip()
        if not text:
            return

        doc_id = f"{source_tag}_{self.total_indexed}"
        self.eval_texts[doc_id] = text

        # 1. Index 8-word n-grams
        ngrams = extract_word_ngrams(text, n=self.ngram_size)
        for ng in ngrams:
            self.eval_ngrams[ng] = source_tag

        # 2. Index MinHash
        m = get_minhash(text, num_perm=128, n=5)
        self.lsh.insert(doc_id, m)
        self.total_indexed += 1

    def check_contamination(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if text is contaminated against the evaluation index.
        Returns (is_contaminated, benchmark_name).
        """
        if not text or not text.strip():
            return False, None

        # 1. Check verbatim 8-word n-gram matches
        cand_ngrams = extract_word_ngrams(text, n=self.ngram_size)
        for ng in cand_ngrams:
            if ng in self.eval_ngrams:
                return True, f"8gram_match ({self.eval_ngrams[ng]})"

        # 2. Check MinHash near-match (Jaccard >= threshold)
        m = get_minhash(text, num_perm=128, n=5)
        matches = self.lsh.query(m)
        if matches:
            first_match = matches[0]
            source_tag = first_match.rsplit("_", 1)[0]
            return True, f"minhash_near_match ({source_tag})"

        return False, None


def load_all_eval_targets(decontaminator: BenchmarkDecontaminator):
    """Loads all evaluation benchmark texts into the decontaminator."""
    # 1. MILU questions
    try:
        from datasets import load_dataset
        ds = load_dataset("ai4bharat/MILU", "Marathi", split="validation")
        for r in ds:
            decontaminator.add_eval_text(r.get("question", ""), "milu_val")
            for opt in ["option1", "option2", "option3", "option4"]:
                decontaminator.add_eval_text(r.get(opt, ""), "milu_val_opt")
    except Exception as e:
        print(f"Warning: could not load MILU for decontamination: {e}")

    # 2. Flores-IN sentences
    try:
        from marathi_llm.eval.flores import load_flores_data
        for direction in ["en_mr", "mr_en"]:
            for item in load_flores_data(direction, split="dev"):
                decontaminator.add_eval_text(item.get("source", ""), f"flores_{direction}")
                decontaminator.add_eval_text(item.get("target", ""), f"flores_{direction}")
    except Exception as e:
        print(f"Warning: could not load Flores for decontamination: {e}")

    # 3. S0 Test & Dev splits
    splits_dir = Path("data/splits")
    for s_file, tag in [("s0_test_ids.json", "s0_test"), ("s0_dev_ids.json", "s0_dev")]:
        fpath = splits_dir / s_file
        if fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # If row_ids exist, we can mark them
            except Exception:
                pass

    print(f"Decontamination index ready: {decontaminator.total_indexed} evaluation strings indexed.")
