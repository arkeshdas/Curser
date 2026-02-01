# app.py
import sys
from pathlib import Path

# Add the project root (parent of /app) to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import tempfile
import time
from datetime import datetime
import subprocess

import streamlit as st
import numpy as np
from langdetect import detect
from scipy.io.wavfile import write as wav_write

from src.asr import transcribe_audio_file
from src.core import (
    tokenize,
    choose_best_span,
    windowed_panphon_distance_tokens,
    similarity_from_distance,
    normalize_ipa,
)
from src.g2p import text_to_ipa


# -----------------------
# Optional mic support (often unavailable on Streamlit Cloud)
# -----------------------
SD_AVAILABLE = False
sd = None
try:
    import sounddevice as _sd  # type: ignore

    sd = _sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False
    sd = None


def infer_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


@st.cache_data
def load_db():
    db_path = ROOT / "db" / "db.json"
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    for e in db:
        if "ipa_norm" not in e:
            e["ipa_norm"] = normalize_ipa(e.get("ipa", ""))
    return db


DB = load_db()

# -----------------------
# Page config + CSS theme
# -----------------------
st.set_page_config(page_title="Curser", page_icon="🖱️", layout="centered")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

      :root {
        --curser-font: 'Press Start 2P', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }

      /* Pixel font for normal text and headings */
      html, body, .stApp {
        font-family: var(--curser-font) !important;
      }
      h1, h2, h3, h4, h5, h6, p, label, span, small, li {
        font-family: var(--curser-font) !important;
      }

      /* Background gradient: top to bottom */
      .stApp {
        background: linear-gradient(180deg, #ff4646 0%, #bdbdbd 100%);
      }

      /* Center column */
      .block-container {
        max-width: 980px;
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
      }

      header[data-testid="stHeader"] {
        background: transparent;
      }

      /* Crisp logo rendering */
      img {
        image-rendering: pixelated;
        image-rendering: crisp-edges;
      }

      /* Header text */
      .curser-title {
        font-family: var(--curser-font) !important;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin: 0;
        line-height: 1.05;
        color: rgba(255,255,255,0.96);
        text-shadow: 0 2px 10px rgba(0,0,0,0.35);
      }
      .curser-subtitle {
        font-family: var(--curser-font) !important;
        font-size: 10px;
        letter-spacing: 0.25px;
        margin-top: 6px;
        color: rgba(255,255,255,0.85);
      }

      /* Dark translucent cards */
      .curser-card {
        background: rgba(10, 10, 12, 0.58);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 16px 16px 12px 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.24);
        backdrop-filter: blur(7px);
        -webkit-backdrop-filter: blur(7px);
        margin: 12px 0;
      }

      /* Make text readable on dark cards */
      .curser-card label,
      .curser-card p,
      .curser-card span,
      .curser-card div,
      .curser-card h1,
      .curser-card h2,
      .curser-card h3 {
        color: rgba(255,255,255,0.92) !important;
      }

      /* Inputs */
      .stTextInput input,
      .stSelectbox div[data-baseweb="select"] > div,
      .stTextArea textarea {
        font-family: var(--curser-font) !important;
        background: rgba(15,15,18,0.92) !important;
        color: rgba(255,255,255,0.96) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 12px !important;
      }

      /* Buttons */
      .stButton > button {
        font-family: var(--curser-font) !important;
        background: rgba(15,15,18,0.92) !important;
        color: rgba(255,255,255,0.96) !important;
        border: 1px solid rgba(255,70,70,0.78) !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        box-shadow: 0 7px 20px rgba(0,0,0,0.28);
      }
      .stButton > button:hover {
        border-color: rgba(255,70,70,1.0) !important;
        transform: translateY(-1px);
      }

      /* Sliders */
      div[data-testid="stSlider"] > div {
        color: rgba(255,255,255,0.92) !important;
      }
      div[data-testid="stSlider"] [role="slider"] {
        background: rgba(255,70,70,0.98) !important;
        border: 2px solid rgba(0,0,0,0.40) !important;
      }

      /* Dataframe */
      [data-testid="stDataFrame"] {
        background: rgba(15,15,18,0.62) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        overflow: hidden;
      }

      /* Headings spacing */
      h2, h3 {
        margin-top: 0.65rem !important;
      }

      /* --- File uploader fixes (pixel font causes overlap) --- */
      [data-testid="stFileUploader"] * {
        font-size: 12px !important;
        line-height: 1.25 !important;
      }
      [data-testid="stFileUploader"] label {
        margin-bottom: 6px !important;
      }
      [data-testid="stFileUploader"] section {
        padding: 10px 12px !important;
      }
      [data-testid="stFileUploader"] [data-testid="stFileDropzone"] * {
        white-space: normal !important;
      }
      [data-testid="stFileUploader"] small,
      [data-testid="stFileUploader"] [data-testid="stFileDropzone"] small {
        display: block !important;
        margin-top: 6px !important;
        opacity: 0.85 !important;
      }
      [data-testid="stFileUploader"] button {
        font-family: var(--curser-font) !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Helpers: card wrappers
# -----------------------
def card_open():
    st.markdown('<div class="curser-card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# Header: logo + title (no clipping)
# -----------------------
logo_path = Path(__file__).resolve().parent / "static" / "curser-logo.png"

col_logo, col_text = st.columns([0.12, 0.88], vertical_alignment="center")
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=48)
    else:
        st.write("")

with col_text:
    st.markdown(
        """
        <div>
          <div class="curser-title">Curser</div>
          <div class="curser-subtitle">Stop embarrassing names before they launch</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------
# TTS helpers (espeak -> wav bytes)
# -----------------------
ESPEAK_VOICE_MAP = {
    "en": "en-us",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ja": "ja",
    "hi": "hi",
    "ko": "ko",
}


def tts_espeak_wav_bytes(text: str, lang_code: str) -> bytes | None:
    if not text:
        return None
    voice = ESPEAK_VOICE_MAP.get(lang_code, "en-us")
    cmd = ["espeak", "-v", voice, "--stdout", text]
    try:
        p = subprocess.run(cmd, capture_output=True, check=False)
    except Exception:
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    return p.stdout


# -----------------------
# Session state
# -----------------------
if "text_phrase" not in st.session_state:
    st.session_state["text_phrase"] = ""
if "audio_lang" not in st.session_state:
    st.session_state["audio_lang"] = None
if "last_audio_sig" not in st.session_state:
    st.session_state["last_audio_sig"] = None
if "live_mode" not in st.session_state:
    st.session_state["live_mode"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_live_text" not in st.session_state:
    st.session_state["last_live_text"] = ""
if "current_result" not in st.session_state:
    st.session_state["current_result"] = None

do_rerun = False
rerun_sleep_s = 0.0


# -----------------------
# Level meter helpers
# -----------------------
def rms_level(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(x * x)))


def rms_to_meter(rms: float, floor: float = 1e-4, ceil: float = 0.2) -> float:
    if rms <= floor:
        return 0.0
    v = (rms - floor) / (ceil - floor)
    return float(np.clip(v, 0.0, 1.0))


def record_audio_with_meter(
    device_index: int,
    seconds: float,
    fs: int = 16000,
    block_ms: int = 50,
    meter_placeholder=None,
    meter_label: str = "Level",
) -> np.ndarray:
    if not SD_AVAILABLE or sd is None:
        return np.array([], dtype=np.float32)

    blocksize = max(1, int(fs * block_ms / 1000))
    frames = []
    t0 = time.time()

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=device_index,
        ) as stream:
            while True:
                elapsed = time.time() - t0
                if elapsed >= seconds:
                    break

                block, _overflowed = stream.read(blocksize)
                block = np.asarray(block).reshape(-1)
                frames.append(block)

                rms = rms_level(block)
                meter_val = rms_to_meter(rms)

                if meter_placeholder is not None:
                    meter_placeholder.progress(meter_val, text=f"{meter_label}: {meter_val:.2f}")
    except Exception:
        return np.array([], dtype=np.float32)

    if not frames:
        return np.array([], dtype=np.float32)

    return np.concatenate(frames).astype(np.float32)


# -----------------------
# Silence trimming
# -----------------------
def trim_silence(
    x: np.ndarray,
    fs: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
    rel_thresh: float = 0.08,
    abs_thresh: float = 1e-3,
    pad_ms: int = 80,
) -> np.ndarray:
    if x.size == 0:
        return x

    x = x.astype(np.float32, copy=False)
    x = x - float(np.mean(x))

    frame = max(1, int(fs * frame_ms / 1000))
    hop = max(1, int(fs * hop_ms / 1000))
    pad = int(fs * pad_ms / 1000)

    if len(x) < frame:
        return x

    rms = []
    idx = 0
    while idx + frame <= len(x):
        chunk = x[idx : idx + frame]
        rms.append(float(np.sqrt(np.mean(chunk * chunk))))
        idx += hop

    if not rms:
        return x

    rms = np.array(rms, dtype=np.float32)
    thr = max(abs_thresh, rel_thresh * float(rms.max()))

    voiced = np.where(rms >= thr)[0]
    if voiced.size == 0:
        return np.array([], dtype=np.float32)

    start_f = int(voiced[0])
    end_f = int(voiced[-1])

    start = max(0, start_f * hop - pad)
    end = min(len(x), end_f * hop + frame + pad)
    return x[start:end]


def write_temp_wav(x: np.ndarray, fs: int = 16000) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_write(tmp.name, fs, (x * 32767).astype(np.int16))
        return tmp.name


# -----------------------
# Core compute + history
# -----------------------
def resolve_lang(span_text: str, chosen_override_lang: str | None) -> str:
    if chosen_override_lang:
        return chosen_override_lang
    if st.session_state.get("audio_lang"):
        return st.session_state["audio_lang"]
    if len(span_text) < 15:
        return "en"
    return infer_lang(span_text) or "en"


def compute_results(text: str, chosen_override_lang: str | None) -> dict:
    tokens = tokenize(text)

    def g2p_fn(span_text: str):
        return text_to_ipa(span_text, lang=resolve_lang(span_text, chosen_override_lang))

    def match_fn(user_ipa: str) -> float:
        if not user_ipa:
            return 0.0
        best_dist = float("inf")
        for e in DB:
            cand_norm = e["ipa_norm"]
            dist, _win, _tok = windowed_panphon_distance_tokens(user_ipa, cand_norm)
            if dist < best_dist:
                best_dist = dist
        return similarity_from_distance(best_dist)

    _score, span, ipa = choose_best_span(tokens, g2p_fn, match_fn)

    if not ipa:
        return {
            "ok": False,
            "text": text,
            "best_span": span,
            "g2p_lang": resolve_lang(span, chosen_override_lang) if span else None,
            "ipa": "",
            "ipa_norm": "",
            "top": [],
            "why": None,
        }

    scored = []
    for e in DB:
        cand_norm = e["ipa_norm"]
        dist, win, tok = windowed_panphon_distance_tokens(ipa, cand_norm)
        sim = similarity_from_distance(dist)
        scored.append((dist, sim, win, tok, e))
    scored.sort(key=lambda x: x[0])
    top10 = scored[:10]

    best_dist, best_sim, best_win, best_tok, best_e = top10[0]
    
    why = {
        "user_ipa_raw": ipa,
        "user_ipa_norm": normalize_ipa(ipa),
        "candidate_word": best_e.get("display", best_e.get("word")),
        "candidate_word_raw": best_e.get("word"),
        "candidate_lang": best_e.get("lang"),
        "candidate_ipa_raw": best_e.get("ipa", ""),
        "candidate_ipa_norm": best_e.get("ipa_norm", ""),
        "source_token": best_tok,
        "best_window": best_win,
        "distance": float(best_dist),
        "similarity": float(best_sim),
    }

    top_rows = []
    for dist, sim, win, tok, e in top10:
        top_rows.append(
    {
        "display": e.get("display", e.get("word")),
        "word": e.get("word"),
        "lang": e.get("lang"),
        "meaning": e.get("meaning", ""),
        "severity": e.get("severity", ""),
        "ipa": e.get("ipa", ""),
        "source_token": tok,
        "best_window": win,
        "distance": float(dist),
        "similarity": float(sim),
    }
)


    return {
        "ok": True,
        "text": text,
        "best_span": span,
        "g2p_lang": resolve_lang(span, chosen_override_lang) if span else None,
        "ipa": ipa,
        "ipa_norm": normalize_ipa(ipa),
        "top": top_rows,
        "why": why,
    }


def push_history(result: dict):
    item = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "text": result.get("text", ""),
        "best_span": result.get("best_span"),
        "g2p_lang": result.get("g2p_lang"),
        "ipa": result.get("ipa", ""),
        "top1": (result.get("top") or [None])[0],
        "why": result.get("why"),
    }
    st.session_state["history"].insert(0, item)
    st.session_state["history"] = st.session_state["history"][:25]


# -----------------------
# Mode selector UI
# -----------------------
MODES_ALL = ["Listen (live-ish)", "Record (one-shot)", "Upload audio"]
MODES_NO_MIC = ["Upload audio"]

if SD_AVAILABLE:
    mode = st.selectbox("Mode", MODES_ALL, index=0)
else:
    st.warning("Mic input is unavailable in this environment, only Upload mode is enabled.")
    mode = st.selectbox("Mode", MODES_NO_MIC, index=0)

# Mic selector for listen + record
device_index = None
if mode in ("Listen (live-ish)", "Record (one-shot)"):
    if not SD_AVAILABLE or sd is None:
        st.error("Mic input is not available. Use Upload audio mode instead.")
        st.stop()

    st.subheader("Microphone")
    devices = sd.query_devices()
    input_devices = [
        (i, d["name"]) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0
    ]
    if not input_devices:
        st.error("No input devices found.")
        st.stop()

    default_idx = 0
    for k, (i, name) in enumerate(input_devices):
        if "macbook" in name.lower() or "built-in" in name.lower():
            default_idx = k
            break

    choice = st.selectbox(
        "Select microphone",
        input_devices,
        index=default_idx,
        format_func=lambda x: f"{x[0]}: {x[1]}",
    )
    device_index = choice[0]


# -----------------------
# G2P language override UI
# -----------------------
st.subheader("G2P language")
auto_lang = st.session_state.get("audio_lang")
st.write("Auto (from Whisper):", auto_lang if auto_lang else "None")

override = st.checkbox("Override language", value=False)
chosen_override_lang = None
if override:
    chosen_override_lang = st.selectbox(
        "Override language",
        ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "hi", "ko"],
        index=0,
    )


# -----------------------
# Mode: Upload
# -----------------------
if mode == "Upload audio":
    st.subheader("Upload audio")
    colU1, colU2 = st.columns([1, 1])
    with colU1:
        audio_file = st.file_uploader("Upload a .wav, .mp3, or .m4a", type=["wav", "mp3", "m4a"])
    with colU2:
        if st.button("Clear upload state"):
            st.session_state["last_audio_sig"] = None
            st.session_state["audio_lang"] = None
            st.session_state["text_phrase"] = ""
            st.session_state["current_result"] = None
            st.rerun()

    if audio_file is not None:
        audio_sig = (audio_file.name, audio_file.size)
        if st.session_state["last_audio_sig"] != audio_sig:
            st.session_state["last_audio_sig"] = audio_sig
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split(".")[-1]) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            with st.spinner("Transcribing upload..."):
                text_from_audio, lang_from_audio = transcribe_audio_file(tmp_path, model_name="base")

            st.session_state["text_phrase"] = (text_from_audio or "").strip()
            st.session_state["audio_lang"] = lang_from_audio
            st.session_state["current_result"] = None
            st.rerun()

    if st.session_state.get("text_phrase"):
        st.write("Transcribed text:", st.session_state["text_phrase"])
        st.write("Whisper language:", st.session_state.get("audio_lang"))


# -----------------------
# Mode: Record one-shot
# -----------------------
if mode == "Record (one-shot)":
    st.subheader("One-shot recording")
    dur_s = st.slider("Record seconds", 1.0, 8.0, 3.0, 0.5)
    rel_thr = st.slider("Silence sensitivity (relative)", 0.02, 0.20, 0.08, 0.01)
    abs_thr = st.slider("Silence floor (absolute)", 0.0003, 0.0050, 0.0010, 0.0001)

    if st.button("Record now"):
        fs = 16000
        st.info("Recording...")

        meter = st.progress(0.0, text="Level: 0.00")
        x = record_audio_with_meter(
            device_index=device_index,
            seconds=float(dur_s),
            fs=fs,
            meter_placeholder=meter,
            meter_label="Level",
        )
        meter.empty()

        if x.size == 0:
            st.error("No audio captured. Check mic selection and permissions.")
            st.stop()

        x_trim = trim_silence(x, fs, rel_thresh=float(rel_thr), abs_thresh=float(abs_thr))

        if x_trim.size == 0:
            st.error("Only silence detected. Try again or change mic.")
            st.stop()

        wav_path = write_temp_wav(x_trim, fs=fs)
        with st.spinner("Transcribing..."):
            text_from_audio, lang_from_audio = transcribe_audio_file(wav_path, model_name="base")

        st.session_state["text_phrase"] = (text_from_audio or "").strip()
        st.session_state["audio_lang"] = lang_from_audio
        st.session_state["current_result"] = None
        st.rerun()


# -----------------------
# Mode: Listen live-ish
# -----------------------
if mode == "Listen (live-ish)":
    st.subheader("Live-ish listening")
    colL, colR = st.columns([1, 1])
    with colL:
        chunk_s = st.slider("Chunk seconds", 1.0, 4.0, 2.0, 0.5)
    with colR:
        pause_s = st.slider("Pause between chunks", 0.0, 1.0, 0.2, 0.1)

    rel_thr = st.slider("Silence sensitivity (relative)", 0.02, 0.20, 0.08, 0.01, key="live_rel")
    abs_thr = st.slider("Silence floor (absolute)", 0.0003, 0.0050, 0.0010, 0.0001, key="live_abs")

    colA, colB, colC = st.columns([1, 1, 2])
    if colA.button("Start listening"):
        st.session_state["live_mode"] = True
        st.rerun()
    if colB.button("Stop listening"):
        st.session_state["live_mode"] = False
    if colC.button("Clear history"):
        st.session_state["history"] = []

    st.caption("Records short chunks repeatedly, trims silence, transcribes only if speech is present.")

    level_box = st.empty()

    if st.session_state.get("live_mode", False):
        fs = 16000

        meter = level_box.progress(0.0, text="Listening level: 0.00")
        x = record_audio_with_meter(
            device_index=device_index,
            seconds=float(chunk_s),
            fs=fs,
            meter_placeholder=meter,
            meter_label="Listening level",
        )
        level_box.empty()

        x_trim = trim_silence(x, fs, rel_thresh=float(rel_thr), abs_thresh=float(abs_thr))

        if x_trim.size > 0:
            wav_path = write_temp_wav(x_trim, fs=fs)
            text_from_audio, lang_from_audio = transcribe_audio_file(wav_path, model_name="base")
            text_from_audio = (text_from_audio or "").strip()

            if text_from_audio and text_from_audio != st.session_state.get("last_live_text", ""):
                st.session_state["last_live_text"] = text_from_audio
                st.session_state["text_phrase"] = text_from_audio
                st.session_state["audio_lang"] = lang_from_audio

                live_result = compute_results(text_from_audio, chosen_override_lang)
                st.session_state["current_result"] = live_result

                if live_result.get("ok"):
                    push_history(live_result)

        do_rerun = True
        rerun_sleep_s = float(pause_s)


# -----------------------
# Results section (always renders, all modes)
# -----------------------
st.subheader("Input text")
text = st.text_input("Type a phrase", key="text_phrase")

result = None
if text and text.strip():
    cached = st.session_state.get("current_result")
    if cached and cached.get("text", "") == text.strip():
        result = cached
    else:
        result = compute_results(text.strip(), chosen_override_lang)
        st.session_state["current_result"] = result

if result:
    st.subheader("Result")
    if not result["ok"]:
        st.warning("No IPA produced, try a different input or language.")
    else:
        top1 = result["top"][0]
        why = result["why"]

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"**Best span:** `{result['best_span']}`")
            st.markdown(f"**G2P lang:** `{result['g2p_lang']}`")
            st.markdown(f"**IPA:** `{result['ipa']}`")
        with c2:
            st.markdown(f"**Top match:** `{top1.get('display', top1['word'])}`")
            st.markdown(f"**Lang:** `{top1['lang']}`")
            st.markdown(f"**Severity:** `{top1.get('severity','')}`")
        with c3:
            st.markdown(f"**Similarity:** `{top1['similarity']:.3f}`")
            st.markdown(f"**Distance:** `{top1['distance']:.3f}`")

        # TTS for top match
        colT1, colT2 = st.columns([1, 2])
        with colT1:
            if st.button("Speak top match", key="speak_top_match"):
                wav_bytes = tts_espeak_wav_bytes(top1["word"], top1["lang"])
                if wav_bytes:
                    st.audio(wav_bytes, format="audio/wav")
                else:
                    st.warning("TTS failed for this word/voice.")
        with colT2:
            st.caption("Uses espeak voices, some languages may not be installed or may differ.")

        st.subheader("Top matches")
        rows = result["top"]

        # normalize types for display
        for r in rows:
            r["distance"] = float(r.get("distance", 0.0))
            r["similarity"] = float(r.get("similarity", 0.0))
            sev = r.get("severity", "")
            r["severity"] = int(sev) if str(sev).strip().isdigit() else 0
            if "display" not in r:
                r["display"] = r.get("word", "")

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "display": st.column_config.TextColumn("word"),
                "word": st.column_config.TextColumn("word_raw"),
                "distance": st.column_config.NumberColumn("distance", format="%.3f"),
                "similarity": st.column_config.NumberColumn("similarity", format="%.3f"),
                "severity": st.column_config.NumberColumn("severity", format="%d"),
            },
        )

        with st.expander("Speak a match"):
            st.write("Click Speak to hear espeak pronounce the word using its language voice.")
            for i, r in enumerate(rows[:10]):
                shown = r.get("display", r.get("word", ""))
                raw = r.get("word", "")
                cA, cB, cC, cD = st.columns([1.2, 0.8, 3.0, 1.0])
                with cA:
                    st.write(f"**{shown}**")
                with cB:
                    st.write(r.get("lang", ""))
                with cC:
                    st.write(r.get("meaning", ""))
                with cD:
                    if st.button("Speak", key=f"speak_row_{i}_{raw}_{r.get('lang','')}"):
                        wav_bytes = tts_espeak_wav_bytes(raw, r.get("lang", "en"))
                        if wav_bytes:
                            st.audio(wav_bytes, format="audio/wav")
                        else:
                            st.warning("TTS failed for this word/voice.")

        with st.expander("Why this match?"):
            st.write("User IPA (raw):", why["user_ipa_raw"])
            st.write("User IPA (norm):", why["user_ipa_norm"])
            st.write("Candidate:", f"{why['candidate_word']} [{why['candidate_lang']}]")
            st.write("Candidate IPA (raw):", why["candidate_ipa_raw"])
            st.write("Candidate IPA (norm):", why["candidate_ipa_norm"])
            st.write("Source token:", why["source_token"])
            st.write("Best window:", why["best_window"])
            st.write("Distance:", round(why["distance"], 3), "Similarity:", round(why["similarity"], 3))

        export_obj = {
            "text": result.get("text"),
            "best_span": result.get("best_span"),
            "g2p_lang": result.get("g2p_lang"),
            "ipa": result.get("ipa"),
            "ipa_norm": result.get("ipa_norm"),
            "top": result.get("top"),
            "why": result.get("why"),
        }
        export_json = json.dumps(export_obj, ensure_ascii=False, indent=2)

        colD1, colD2 = st.columns([1, 2])
        with colD1:
            st.download_button(
                "Download JSON",
                data=export_json,
                file_name="curserio_result.json",
                mime="application/json",
            )
        with colD2:
            with st.expander("Show JSON"):
                st.code(export_json, language="json")


# -----------------------
# History
# -----------------------
st.subheader("History (latest first)")
if not st.session_state["history"]:
    st.write("No saved results yet.")
else:
    for item in st.session_state["history"]:
        top1 = item.get("top1") or {}
        shown = top1.get("display", top1.get("word", ""))
        title = f"{item['ts']} | {shown} [{top1.get('lang','')}] | {item.get('text','')}"
        with st.expander(title):
            st.write("Text:", item.get("text"))
            st.write("Best span:", item.get("best_span"))
            st.write("G2P lang:", item.get("g2p_lang"))
            st.write("IPA:", item.get("ipa"))
            st.write("Top1:", top1)

    hist_json = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2)
    st.download_button(
        "Download full history JSON",
        data=hist_json,
        file_name="curserio_history.json",
        mime="application/json",
    )


# -----------------------
# Rerun scheduling for Listen mode (must be last)
# -----------------------
if do_rerun and st.session_state.get("live_mode", False):
    time.sleep(rerun_sleep_s)
    st.rerun()
