"""Deterministic text matching for augmentation leakage checks."""
import unicodedata
def norm(text):
    return ' '.join(unicodedata.normalize('NFKC',str(text)).casefold().split())
