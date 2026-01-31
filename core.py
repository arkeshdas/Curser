# core.py
import math
from typing import Callable, Iterable, Optional, Tuple, List, Dict, Any

STOPWORDS = {
    "i", "am", "a", "the", "is", "it", "you",
    "to", "and", "or", "in", "on", "at", "of",
    "for", "with", "uh", "um"
}

def tokenize(text: str) -> List[str]:
    """
    MVP tokenizer: lowercase, split on whitespace.
    """
    return [t for t in text.lower().split() if t.strip()]

def choose_best_span(
    tokens: List[str],
    g2p_fn: Callable[[str], Optional[str]],
    match_fn: Callable[[str], float],
    stopwords: set = STOPWORDS,
    max_len: int = 8,
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    Scores all contiguous spans and returns the best one.
    Returns: (best_score, best_span_text, best_ipa)
    """
    best_score = -math.inf
    best_span = None
    best_ipa = None

    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, min(n, i + max_len) + 1):
            span_tokens = tokens[i:j]

            # Skip spans that are only stopwords
            if all(t in stopwords for t in span_tokens):
                continue

            span_text = " ".join(span_tokens)
            ipa = g2p_fn(span_text)

            # --- scoring ---
            len_bonus = math.sqrt(len(span_tokens))
            g2p_quality = 1.0 if ipa else 0.0

            penalty = 0.0
            if len(span_tokens) < 2:
                penalty += 1.0

            stop_ratio = sum(t in stopwords for t in span_tokens) / len(span_tokens)
            penalty += 1.25 * stop_ratio
            penalty += 0.40 * (len(span_tokens) - 1)

            match_sim = match_fn(ipa) if ipa else 0.0

            score = len_bonus + g2p_quality + 2.0 * match_sim - penalty

            if score > best_score:
                best_score = score
                best_span = span_text
                best_ipa = ipa

    return best_score, best_span, best_ipa

def rank_db(user_ipa: str, db: List[Dict[str, Any]], distance_fn: Callable[[str, str], float]):
    """
    Ranks DB entries by distance (smaller is better).
    Returns list of tuples: (distance, entry)
    """
    scored = []
    for entry in db:
        d = distance_fn(user_ipa, entry["ipa"])
        scored.append((d, entry))
    scored.sort(key=lambda x: x[0])
    return scored

def token_hamming_distance(a: str, b: str) -> int:
    ta, tb = a.split(), b.split()
    m = max(len(ta), len(tb))
    ta += ["_"] * (m - len(ta))
    tb += ["_"] * (m - len(tb))
    return sum(x != y for x, y in zip(ta, tb))