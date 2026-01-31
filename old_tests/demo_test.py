# demo_text.py
from core import tokenize, choose_best_span, rank_db, token_hamming_distance
from g2p import text_to_ipa

DB = [
    {"word": "culo", "lang": "es", "ipa": "k u l o", "meaning": "vulgar slang", "severity": 2},
    {"word": "cola", "lang": "es", "ipa": "k o l a", "meaning": "neutral word", "severity": 0},
]

def match_fn(ipa: str) -> float:
    # similarity = inverse distance to best DB entry
    best = min(token_hamming_distance(ipa, e["ipa"]) for e in DB)
    return 1.0 / (best + 1.0)

def g2p_fn(span_text: str):
    # use espeak IPA for English input
    return text_to_ipa(span_text, lang="en")

tokens = tokenize("the very cool-io thing")
score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)
print("best span:", span)
print("ipa:", ipa)

ranked = rank_db(ipa, DB, token_hamming_distance)
print("top match:", ranked[0][1])
