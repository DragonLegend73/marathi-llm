"""Text normalization and tokenization utilities for Marathi NLP."""

import re
import unicodedata
from typing import List

# Zero-width spaces, joiners, non-joiners, and BOM
ZW = re.compile(r"[\u200B-\u200D\uFEFF]")

# Word chars + all Devanagari letters/matras/virama/digits,
# strictly excluding danda (U+0964) and double danda (U+0965)
WORD_PATTERN = re.compile(r"[\w\u0900-\u0963\u0966-\u097F]+")


def normalize_for_match(text: str) -> str:
    """Normalizes text for matching by stripping zero-width characters, normalizing NFC, and lowercasing."""
    if not text:
        return ""
    return ZW.sub("", unicodedata.normalize("NFC", str(text))).lower()


def words(text: str) -> List[str]:
    """Tokenizes text into words, respecting Devanagari characters, matras, and viramas while excluding dandas."""
    return WORD_PATTERN.findall(normalize_for_match(text))
