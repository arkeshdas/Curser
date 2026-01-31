# app.py
import json
import streamlit as st

from core import tokenize, choose_best_span, panphon_distance, similarity_from_distance, normalize_ipa
from g2p import text_to_ipa

@st.cache_data
def load_db():
    with open("db.json", "r", encoding="utf-8") as f:
        return json.load(f)

DB = load_db()

st.title("Curserio, text prototype")

text = st.text_input("Type a phrase")
input_lang = st.selectbox("Input language for G2P", ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "hi"], index=0)

if text:
    tokens = tokenize(text)

    def g2p_fn(span_text: str):
        return text_to_ipa(span_text, lang=input_lang)

    # match_fn needs to return a similarity score (higher is better)
    def match_fn(user_ipa: str) -> float:
        if not user_ipa:
            return 0.0
        user_norm = normalize_ipa(user_ipa)
        best_dist = float("inf")
        for e in DB:
            dist = panphon_distance(user_norm, e["ipa_norm"])
            if dist < best_dist:
                best_dist = dist
        return similarity_from_distance(best_dist)

    score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)

    st.write("Best span:", span)
    st.write("IPA:", ipa)

    if ipa:
        user_norm = normalize_ipa(ipa)

        scored = []
        for e in DB:
            dist = panphon_distance(user_norm, e["ipa_norm"])
            sim = similarity_from_distance(dist)
            scored.append((dist, sim, e))

        scored.sort(key=lambda x: x[0])  # smallest distance first
        top = scored[:10]

        st.subheader("Top matches")
        st.table([
            {
                "word": e["word"],
                "lang": e["lang"],
                "meaning": e.get("meaning", ""),
                "severity": e.get("severity", ""),
                "ipa": e["ipa"],
                "distance": round(dist, 3),
                "similarity": round(sim, 3),
            }
            for dist, sim, e in top
        ])
    else:
        st.warning("No IPA produced, try a different input or language.")