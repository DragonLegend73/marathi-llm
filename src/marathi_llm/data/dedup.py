"""Exact and near-duplicate deduplication using MinHash LSH (datasketch)."""

import hashlib
from typing import List, Dict, Any, Tuple, Set, Generator
from datasketch import MinHash, MinHashLSH

from marathi_llm.data.text import words


def get_word_shingles(text: str, n: int = 5) -> List[str]:
    """Extracts word n-grams (shingles) from text, adapting to 3-grams for short texts."""
    w_list = words(text)
    if not w_list:
        return []
    actual_n = n if len(w_list) >= 20 else max(1, min(3, len(w_list)))
    if len(w_list) < actual_n:
        return [" ".join(w_list)]
    return [" ".join(w_list[i : i + actual_n]) for i in range(len(w_list) - actual_n + 1)]


def compute_minhash(text: str, num_perm: int = 128, shingle_size: int = 5) -> MinHash:
    """Computes a datasketch MinHash object for a given string."""
    m = MinHash(num_perm=num_perm)
    shingles = get_word_shingles(text, n=shingle_size)
    for shingle in shingles:
        m.update(shingle.encode("utf-8"))
    return m


class DatasetDeduplicator:
    """
    Manages exact and MinHash near-duplicate deduplication.
    Ensures that neither exact twins nor near-paraphrases (> threshold Jaccard) are accepted.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_size: int = 5):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.exact_hashes: Set[str] = set()
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_count = 0

    def is_duplicate(self, text: str, key_id: str) -> bool:
        """Checks if text is an exact or near duplicate of any previously seen item."""
        norm = " ".join(text.strip().split())
        if not norm:
            return True

        # 1. Exact hash check
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in self.exact_hashes:
            return True

        # 2. MinHash LSH check
        m = compute_minhash(norm, num_perm=self.num_perm, shingle_size=self.shingle_size)
        matches = self.lsh.query(m)
        if matches:
            return True

        # Add to index
        self.exact_hashes.add(h)
        self.lsh.insert(key_id, m)
        self.seen_count += 1
        return False
