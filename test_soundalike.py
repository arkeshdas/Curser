# test_soundalike.py
import math

# ----------------------------
# Core logic (temporary)
# ----------------------------

STOPWORDS = {
    "i", "am", "a", "the", "is", "it", "you",
    "to", "and", "or", "in", "on", "at", "of",
    "for", "with", "uh", "um"
}

def tokenize(text: str):
    """
    MVP tokenizer: lowercase, split on whitespace.
    """
    return [t for t in text.lower().split() if t.strip()]

def choose_best_span(tokens, g2p_fn, match_fn, stopwords=STOPWORDS, max_len=8):
    """
    Scores all contiguous spans and returns the best one.
    Returns: (score, span_text, ipa)
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

def rank_db(user_ipa, db, distance_fn):
    """
    Ranks DB entries by distance (smaller is better).
    """
    scored = []
    for entry in db:
        d = distance_fn(user_ipa, entry["ipa"])
        scored.append((d, entry))
    scored.sort(key=lambda x: x[0])
    return scored

# ----------------------------
# Tests (CSE 331 style)
# ----------------------------

def test_tokenize_basic():
    tokens = tokenize("Hello world")
    assert tokens == ["hello", "world"]

def test_span_prefers_content_word():
    tokens = tokenize("it is the cool-io")

    # Mock G2P
    g2p_map = {
        "cool-io": "k u l o",
        "the cool-io": "ð ə k u l o",
        "it is": "ɪ t ɪ z",
    }

    def g2p_fn(s):
        return g2p_map.get(s)

    # Mock similarity
    def match_fn(ipa):
        return 0.95 if ipa == "k u l o" else 0.1

    score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)

    assert span == "cool-io"
    assert ipa == "k u l o"

def test_shorter_clean_span_beats_longer_noisy_span():
    tokens = tokenize("the very cool-io thing")

    g2p_map = {
        "cool-io": "k u l o",
        "very cool-io": "v ɛ r i k u l o",
        "the very cool-io": "ð ə v ɛ r i k u l o",
    }

    def g2p_fn(s):
        return g2p_map.get(s)

    def match_fn(ipa):
        if ipa == "k u l o":
            return 0.9
        if ipa == "v ɛ r i k u l o":
            return 0.3
        if ipa == "ð ə v ɛ r i k u l o":
            return 0.2
        return 0.0

    score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)

    assert span == "cool-io"
    assert ipa == "k u l o"

def test_db_ranking_orders_by_distance():
    db = [
        {"word": "culo", "lang": "es", "ipa": "k u l o"},
        {"word": "kulo", "lang": "xx", "ipa": "k u l u"},
        {"word": "cola", "lang": "es", "ipa": "k o l a"},
    ]

    def distance_fn(a, b):
        # Simple deterministic distance
        ta, tb = a.split(), b.split()
        m = max(len(ta), len(tb))
        ta += ["_"] * (m - len(ta))
        tb += ["_"] * (m - len(tb))
        return sum(x != y for x, y in zip(ta, tb))

    ranked = rank_db("k u l o", db, distance_fn)

    assert ranked[0][1]["word"] == "culo"
    assert ranked[-1][1]["word"] == "cola"

def test_no_valid_span_returns_none():
    tokens = tokenize("uh um uh")

    def g2p_fn(_):
        return None

    def match_fn(_):
        return 0.0

    score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)

    assert span is None
    assert ipa is None