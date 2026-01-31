# demo_text.py
from core import tokenize, choose_best_span, rank_db, token_hamming_distance
from db import DB

def g2p_mock(span_text: str) -> str | None:
    mapping = {
        "cool-io": "k u l o",
        "very cool-io": "v ɛ r i k u l o",
    }
    return mapping.get(span_text)

def match_fn(ipa: str) -> float:
    # similarity based on best DB distance
    best = min(token_hamming_distance(ipa, e["ipa"]) for e in DB)
    return 1.0 / (best + 1.0)

tokens = tokenize("the very cool-io thing")
score, span, ipa = choose_best_span(tokens, g2p_mock, match_fn)
print("best span:", span, "ipa:", ipa)

ranked = rank_db(ipa, DB, token_hamming_distance)
print("top match:", ranked[0][1])