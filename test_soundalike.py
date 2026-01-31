# test_soundalike.py
from core import tokenize, choose_best_span, rank_db

def test_tokenize_basic():
    tokens = tokenize("Hello world")
    assert tokens == ["hello", "world"]

def test_span_prefers_content_word():
    tokens = tokenize("it is the cool-io")

    g2p_map = {
        "cool-io": "k u l o",
        "the cool-io": "ð ə k u l o",
        "it is": "ɪ t ɪ z",
    }

    def g2p_fn(s):
        return g2p_map.get(s)

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