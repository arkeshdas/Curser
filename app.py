# app.py
import json
import tempfile
import streamlit as st

from asr import transcribe_audio_file
from core import (
    tokenize,
    choose_best_span,
    sliding_window_panphon_distance,
    similarity_from_distance,
    normalize_ipa,
)
from g2p import text_to_ipa
from langdetect import detect


def infer_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


@st.cache_data
def load_db():
    with open("db.json", "r", encoding="utf-8") as f:
        return json.load(f)


DB = load_db()

st.title("Curserio, text prototype")

# Persistent text box state
if "text_phrase" not in st.session_state:
    st.session_state["text_phrase"] = ""

# Persistent auto language from audio (optional)
if "audio_lang" not in st.session_state:
    st.session_state["audio_lang"] = None

st.subheader("Optionally upload audio")
audio_file = st.file_uploader("Upload a .wav or .mp3", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix="." + audio_file.name.split(".")[-1]
    ) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing..."):
        text_from_audio, lang_from_audio = transcribe_audio_file(tmp_path, model_name="base")

    st.write("Transcribed text:", text_from_audio)
    st.write("Whisper language:", lang_from_audio)

    # Push into the text input, store lang, and rerun so the pipeline runs automatically
    st.session_state["text_phrase"] = text_from_audio
    st.session_state["audio_lang"] = lang_from_audio
    st.rerun()

# Text input uses session_state
text = st.text_input("Type a phrase", key="text_phrase")

lang = st.selectbox(
    "Input language for G2P",
    ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "hi", "ko"],
    index=0
)

if text:
    tokens = tokenize(text)

    def g2p_fn(span_text: str):
        if lang != "auto":
            chosen_lang = lang
        else:
            # Prefer Whisper lang if we have it from an audio upload
            chosen_lang = st.session_state.get("audio_lang") or infer_lang(span_text) or "en"
        return text_to_ipa(span_text, lang=chosen_lang)

    def match_fn(user_ipa: str) -> float:
        if not user_ipa:
            return 0.0
        user_norm = normalize_ipa(user_ipa)
        best_dist = float("inf")
        for e in DB:
            ipa_norm = e.get("ipa_norm") or normalize_ipa(e.get("ipa", ""))
            dist = sliding_window_panphon_distance(user_norm, ipa_norm)
            if dist < best_dist:
                best_dist = dist
        return similarity_from_distance(best_dist)

    score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)

    st.write("Best span:", span)
    st.write("IPA:", ipa)
    st.write("IPA (norm):", normalize_ipa(ipa) if ipa else "")

    if ipa:
        user_norm = normalize_ipa(ipa)
        scored = []
        for e in DB:
            ipa_norm = e.get("ipa_norm") or normalize_ipa(e.get("ipa", ""))
            dist = sliding_window_panphon_distance(user_norm, ipa_norm)
            sim = similarity_from_distance(dist)
            scored.append((dist, sim, e))

        scored.sort(key=lambda x: x[0])
        top = scored[:10]

        st.subheader("Top matches")
        st.table([
            {
                "word": e.get("word"),
                "lang": e.get("lang"),
                "meaning": e.get("meaning", ""),
                "severity": e.get("severity", ""),
                "ipa": e.get("ipa", ""),
                "distance": round(dist, 3),
                "similarity": round(sim, 3),
            }
            for dist, sim, e in top
        ])
    else:
        st.warning("No IPA produced, try a different input or language.")