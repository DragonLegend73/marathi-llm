"""Benchmark decontamination engine checking training data against all evaluation suites."""

import collections
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from datasketch import MinHash, MinHashLSH

from marathi_llm.data.text import words, normalize_for_match


def extract_word_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extracts consecutive word n-grams using clean Devanagari word tokenization."""
    w_list = words(text)
    if len(w_list) < n:
        return set()
    return {" ".join(w_list[i : i + n]) for i in range(len(w_list) - n + 1)}


def get_minhash(text: str, num_perm: int = 128, n: int = 5) -> MinHash:
    """Computes MinHash on 5-word shingles using clean Devanagari word tokenization."""
    m = MinHash(num_perm=num_perm)
    w_list = words(text)
    if len(w_list) >= n:
        for i in range(len(w_list) - n + 1):
            shingle = " ".join(w_list[i : i + n])
            m.update(shingle.encode("utf-8"))
    elif w_list:
        m.update(" ".join(w_list).encode("utf-8"))
    return m


class BenchmarkDecontaminator:
    """
    Indexes all evaluation strings across MILU (val+test), Flores-IN (dev+test),
    XQuAD-IN (dev+test), IndicIFEval (ground+trans), and held-out Aya prompts.
    Rejects any training example with >= n-word verbatim overlap or MinHash Jaccard >= threshold.
    """

    def __init__(self, minhash_threshold: float = 0.7, ngram_size: int = 8):
        self.minhash_threshold = minhash_threshold
        self.ngram_size = ngram_size
        # Multi-tag attribution: ngram -> Set[benchmark_source]
        self.eval_ngrams: Dict[str, Set[str]] = collections.defaultdict(set)
        self.lsh = MinHashLSH(threshold=minhash_threshold, num_perm=128)
        self.eval_texts: Dict[str, str] = {}
        self.tag_counts: Dict[str, int] = collections.Counter()
        self.total_indexed = 0

    def add_eval_text(self, text: str, source_tag: str):
        """Indexes an evaluation text into n-gram and MinHash LSH tables."""
        norm_text = normalize_for_match(text)
        if not norm_text:
            return

        doc_id = f"{source_tag}_{self.total_indexed}"
        self.eval_texts[doc_id] = norm_text
        self.tag_counts[source_tag] += 1

        # 1. Index word n-grams
        ngrams = extract_word_ngrams(norm_text, n=self.ngram_size)
        for ng in ngrams:
            self.eval_ngrams[ng].add(source_tag)

        # 2. Index MinHash
        m = get_minhash(norm_text, num_perm=128, n=5)
        self.lsh.insert(doc_id, m)
        self.total_indexed += 1

    def check_contamination(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks if text is contaminated against the evaluation index.
        Returns (is_contaminated, benchmark_name).
        """
        norm_text = normalize_for_match(text)
        if not norm_text:
            return False, None

        # 1. Check verbatim word n-gram matches
        cand_ngrams = extract_word_ngrams(norm_text, n=self.ngram_size)
        for ng in cand_ngrams:
            if ng in self.eval_ngrams:
                tags = sorted(list(self.eval_ngrams[ng]))
                return True, f"{self.ngram_size}gram_match ({','.join(tags)})"

        # 2. Check MinHash near-match (Jaccard >= threshold)
        m = get_minhash(norm_text, num_perm=128, n=5)
        matches = self.lsh.query(m)
        if matches:
            first_match = matches[0]
            source_tag = first_match.rsplit("_", 1)[0]
            return True, f"minhash_near_match ({source_tag})"

        return False, None


def load_all_eval_targets(decontaminator: BenchmarkDecontaminator):
    """Loads all evaluation benchmark texts into the decontaminator."""
    from datasets import load_dataset

    # 1. MILU (validation + test)
    for split in ["validation", "test"]:
        try:
            ds = load_dataset("ai4bharat/MILU", "Marathi", split=split)
            for r in ds:
                q = r.get("question", "")
                decontaminator.add_eval_text(q, f"milu_{split}")
                for opt in ["option1", "option2", "option3", "option4"]:
                    opt_val = r.get(opt, "")
                    if opt_val:
                        decontaminator.add_eval_text(opt_val, f"milu_{split}_opt")
        except Exception as e:
            print(f"Warning: could not load MILU {split} for decontamination: {e}")

    # 2. Flores-IN (dev + test, source and target in en_mr and mr_en)
    flores_hub_dir = Path("/home/dragon/.cache/huggingface/hub/datasets--google--IndicGenBench_flores_in/snapshots/f8650438298df086750ff4973661bb58a201a5ee")
    for direction in ["en_mr", "mr_en"]:
        for split in ["dev", "test"]:
            f_path = flores_hub_dir / f"flores_{direction}_{split}.json"
            if f_path.exists():
                try:
                    with open(f_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data.get("examples", []):
                        decontaminator.add_eval_text(item.get("source", ""), f"flores_{direction}_{split}")
                        decontaminator.add_eval_text(item.get("target", ""), f"flores_{direction}_{split}")
                except Exception as e:
                    print(f"Warning: could not read Flores {direction} {split}: {e}")

    # 3. XQuAD-IN (dev + test)
    xquad_hub_dir = Path("/home/dragon/.cache/huggingface/hub/datasets--google--IndicGenBench_xquad_in/snapshots/32026f0bcb0eed9aea0418694448c143c98acb0a")
    for split in ["dev", "test"]:
        x_path = xquad_hub_dir / f"xquad_mr_{split}.json"
        if x_path.exists():
            try:
                with open(x_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for ex in data.get("examples", []):
                    if ex.get("context"):
                        decontaminator.add_eval_text(ex.get("context"), f"xquad_{split}_context")
                    if ex.get("question"):
                        decontaminator.add_eval_text(ex.get("question"), f"xquad_{split}_q")
                    for a in ex.get("answers", []):
                        if a.get("text"):
                            decontaminator.add_eval_text(a.get("text"), f"xquad_{split}_ans")
            except Exception as e:
                print(f"Warning: could not read XQuAD {split}: {e}")

    # 4. IndicIFEval (ground + trans)
    ifeval_base = Path("/home/dragon/.cache/huggingface/hub/datasets--ai4bharat--IndicIFEval/snapshots/4343c1b174322e32cc785ef89fe117038fdad7a9")
    for cfg in ["indicifeval-ground", "indicifeval-trans"]:
        p_path = ifeval_base / cfg / "mr-00000-of-00001.parquet"
        if p_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(p_path)
                for _, row in df.iterrows():
                    prompt = row.get("prompt", "")
                    if prompt:
                        decontaminator.add_eval_text(prompt, f"ifeval_{cfg}")
            except Exception as e:
                print(f"Warning: could not read IndicIFEval {cfg}: {e}")

    # 5. Held-out Aya prompts & BPC responses if existing
    native_eval_path = Path("data/eval/native_prompts.jsonl")
    if native_eval_path.exists():
        try:
            with open(native_eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("prompt"):
                            decontaminator.add_eval_text(obj.get("prompt"), "aya_native_prompt")
                        if obj.get("reference"):
                            decontaminator.add_eval_text(obj.get("reference"), "aya_native_ref")
        except Exception as e:
            print(f"Warning: could not read native_prompts.jsonl: {e}")

    bpc_eval_path = Path("data/eval/bpc_responses.jsonl")
    if bpc_eval_path.exists():
        try:
            with open(bpc_eval_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if obj.get("response"):
                            decontaminator.add_eval_text(obj.get("response"), "aya_bpc_resp")
        except Exception as e:
            print(f"Warning: could not read bpc_responses.jsonl: {e}")

    print(f"Decontamination index ready: {decontaminator.total_indexed:,} evaluation strings indexed.")
