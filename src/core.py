# core.py
import math
import re
import unicodedata
from typing import Callable, Optional, Tuple, List, Dict, Any

import panphon.distance

STOPWORDS = {
    "i", "am", "a", "the", "is", "it", "you",
    "to", "and", "or", "in", "on", "at", "of",
    "for", "with", "uh", "um"
}

# Create once, reused everywhere
_PANPHON_DST = panphon.distance.Distance()


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s'-]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return [t for t in text.split(" ") if t]


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

            # Regularize length: extra tokens must earn their keep
            penalty += 0.40 * (len(span_tokens) - 1)

            match_sim = match_fn(ipa) if ipa else 0.0

            score = len_bonus + g2p_quality + 2.0 * match_sim - penalty

            if score > best_score:
                best_score = score
                best_span = span_text
                best_ipa = ipa

    return best_score, best_span, best_ipa


def rank_db(
    user_ipa: str,
    db: List[Dict[str, Any]],
    distance_fn: Callable[[str, str], float],
    ipa_key: str = "ipa",
):
    """
    Ranks DB entries by distance (smaller is better).
    Returns list of tuples: (distance, entry)
    """
    scored = []
    for entry in db:
        d = distance_fn(user_ipa, entry[ipa_key])
        scored.append((d, entry))
    scored.sort(key=lambda x: x[0])
    return scored


def token_hamming_distance(a: str, b: str) -> int:
    ta, tb = a.split(), b.split()
    m = max(len(ta), len(tb))
    ta += ["_"] * (m - len(ta))
    tb += ["_"] * (m - len(tb))
    return sum(x != y for x, y in zip(ta, tb))


def normalize_ipa(s: str) -> str:
    """
    Normalize espeak-ish IPA so panphon does not get wrecked by markers and invisible chars.
    """
    if not s:
        return ""

    s = s.strip()

    # remove wrappers
    s = s.replace("/", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    # remove invisible chars (ZWJ/ZWNJ/ZWSP/BOM, etc)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)

    # remove stress + length markers
    s = re.sub(r"[ˈˌːˑ:]", "", s)

    # remove tie bars
    s = s.replace("͡", "").replace("͜", "")

    # normalize whitespace
    s = re.sub(r"\s+", " ", s)

    # drop combining diacritics
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

    # remove any embedded language tags like en-us
    s = re.sub(r"\b[a-z]{2,3}-[a-z]{2,4}\b", "", s, flags=re.IGNORECASE)

    return s.strip()


def similarity_from_distance(dist: float) -> float:
    """
    Convert a distance to a stable similarity score for ranking.
    """
    if dist == float("inf"):
        return 0.0
    return 1.0 / (1.0 + dist)


def ipa_tokens(s: str) -> List[str]:
    """
    Turn normalized IPA string into tokens.
    Prefer whitespace tokens, fallback to per-character if no spaces.
    """
    s = normalize_ipa(s)
    if not s:
        return []

    toks = [t for t in s.split(" ") if t]
    if len(toks) >= 2:
        return toks

    compact = s.replace(" ", "")
    return list(compact) if compact else []


def panphon_distance_tokens(a_tokens: List[str], b_tokens: List[str]) -> float:
    """
    Panphon feature edit distance on token lists.
    """
    if not a_tokens or not b_tokens:
        return float("inf")

    a = " ".join(a_tokens)
    b = " ".join(b_tokens)

    try:
        return _PANPHON_DST.weighted_feature_edit_distance(a, b)
    except Exception:
        return float("inf")


def windowed_panphon_distance_tokens(
    user_ipa: str,
    cand_ipa: str,
    window_slack: int = 2,
) -> Tuple[float, str, str]:
    """
    Window match candidate inside each user word-token separately.

    Returns:
      (best_distance_normalized, best_window_substring, best_user_token)
    where best_window_substring is the substring taken from best_user_token.
    """
    u_norm = normalize_ipa(user_ipa)
    c_norm = normalize_ipa(cand_ipa)
    if not u_norm or not c_norm:
        return float("inf"), "", ""

    user_tokens = [t for t in u_norm.split() if t]
    c = c_norm.replace(" ", "")
    if not user_tokens or not c:
        return float("inf"), "", ""

    c_len = len(c)

    best = float("inf")
    best_window = ""
    best_tok = ""

    for tok in user_tokens:
        u = tok.replace(" ", "")
        if not u:
            continue

        Lu = len(u)
        w_min = max(1, c_len - window_slack)
        w_max = min(Lu, c_len + window_slack)

        for wlen in range(w_min, w_max + 1):
            for start in range(0, Lu - wlen + 1):
                u_sub = u[start:start + wlen]
                try:
                    dist = _PANPHON_DST.weighted_feature_edit_distance(u_sub, c)
                except Exception:
                    continue

                dist_norm = dist / max(len(u_sub), len(c), 1)

                if dist_norm < best:
                    best = dist_norm
                    best_window = u_sub
                    best_tok = u

    return best, best_window, best_tok
