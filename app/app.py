#app.py
# app.py
import sys
from pathlib import Path

# -----------------------
# Path setup so `import src.*` works no matter where you run from
# -----------------------
HERE = Path(__file__).resolve()

ROOT = None
for p in [HERE.parent] + list(HERE.parents):
    if (p / "src").is_dir():
        ROOT = p
        break

if ROOT is None:
    raise RuntimeError(f"Could not find project root containing `src/` starting from: {HERE}")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import json
import tempfile
import time
from datetime import datetime
import subprocess
import hashlib

import streamlit as st

# ElevenLabs is optional, keep app runnable even if not installed
try:
    from elevenlabs.client import ElevenLabs  # type: ignore

    ELEVENLABS_AVAILABLE = True
except Exception:
    ElevenLabs = None  # type: ignore
    ELEVENLABS_AVAILABLE = False

import numpy as np
import html
import pandas as pd
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
# Mic backends
# -----------------------
try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore

    BROWSER_MIC_AVAILABLE = True
except Exception:
    mic_recorder = None
    BROWSER_MIC_AVAILABLE = False

SD_AVAILABLE = False
sd = None
try:
    import sounddevice as _sd  # type: ignore

    sd = _sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False
    sd = None


def get_default_input_device_index() -> int | None:
    if not SD_AVAILABLE or sd is None:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    candidates: list[tuple[int, dict]] = [
        (i, d)
        for i, d in enumerate(devices)
        if int(d.get("max_input_channels", 0) or 0) > 0
    ]
    if not candidates:
        return None

    for i, d in candidates:
        name = (d.get("name") or "").lower()
        if ("macbook" in name) or ("built-in" in name):
            return int(i)

    return int(candidates[0][0])


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

:root{
  --curser-font:'Press Start 2P', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --ui-font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";

  --card-bg: rgba(10,10,12,0.58);
  --card-border: rgba(255,255,255,0.12);
  --control-bg: rgba(15,15,18,0.92);
  --control-border: rgba(255,255,255,0.18);

  --text: rgba(255,255,255,0.92);
  --text-dim: rgba(255,255,255,0.78);
}

/* Background */
.stApp{
  background: linear-gradient(180deg, #ff4646 0%, #bdbdbd 100%);
}

.block-container{
  max-width: 980px;
  padding-top: 1.1rem;
  padding-bottom: 2.5rem;
}

header[data-testid="stHeader"]{ background: transparent; }

/* Pixel font: apply to your content, not the entire app */
.curser-title,
.curser-subtitle,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stText"] *,
[data-testid="stCaptionContainer"] *,
label, small{
  font-family: var(--curser-font) !important;
}

/* Keep icons and SVGs on normal font rendering */
[data-testid="stIcon"],
[data-testid="stIcon"] *,
svg, svg *{
  font-family: var(--ui-font) !important;
}

/* Crisp logo rendering */
img{
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

/* Header text */
.curser-title{
  font-size: 26px;
  font-weight: 800;
  letter-spacing: 0.5px;
  margin: 0;
  line-height: 1.05;
  color: rgba(255,255,255,0.96);
  text-shadow: 0 2px 10px rgba(0,0,0,0.35);
}
.curser-subtitle{
  font-size: 10px;
  letter-spacing: 0.25px;
  margin-top: 6px;
  color: rgba(255,255,255,0.85);
}

/* Cards */
.curser-card{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.24);
  backdrop-filter: blur(7px);
  -webkit-backdrop-filter: blur(7px);
  margin: 12px 0;
}

/* Default text color */
h1,h2,h3,h4,h5,h6,p,span,small,li,div{
  color: var(--text) !important;
}

/* Inputs */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea{
  font-family: var(--curser-font) !important;
  background: var(--control-bg) !important;
  color: rgba(255,255,255,0.96) !important;
  border: 1px solid var(--control-border) !important;
  border-radius: 12px !important;
}

/* Buttons */
.stButton > button,
button[kind="secondary"],
button[kind="primary"]{
  font-family: var(--curser-font) !important;
  background: var(--control-bg) !important;
  color: rgba(255,255,255,0.96) !important;
  border: 1px solid rgba(255,70,70,0.78) !important;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  box-shadow: 0 7px 20px rgba(0,0,0,0.28);
}

/* Dataframe container only.
   Streamlit's dataframe body uses canvas in some versions, CSS cannot always recolor cells. */
[data-testid="stDataFrame"]{
  background: rgba(15,15,18,0.78) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  overflow: hidden;
}

/* File uploader fixes */
[data-testid="stFileUploader"] *{
  font-size: 12px !important;
  line-height: 1.25 !important;
}
[data-testid="stFileUploader"] section{
  padding: 10px 12px !important;
}
[data-testid="stFileUploader"] [data-testid="stFileDropzone"] *{
  white-space: normal !important;
}
[data-testid="stFileUploader"] small{
  display: block !important;
  margin-top: 6px !important;
  opacity: 0.85 !important;
  color: var(--text-dim) !important;
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
# Header: logo + title
# -----------------------
logo_path = Path(__file__).resolve().parent / "static" / "curser-logo.png"

col_logo, col_text = st.columns([0.10, 0.90], vertical_alignment="center")
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=44)
    else:
        st.error(f"Logo not found at: {logo_path}")

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

st.write("")

# -----------------------
# TTS helpers
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

# For demo, pick a single voice id and keep it stable.
# You can replace this with your chosen ElevenLabs voice id later.
DEFAULT_ELEVEN_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
DEFAULT_ELEVEN_MODEL_ID = "eleven_multilingual_v2"

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
def tts_elevenlabs_bytes(
    text: str, *, voice_id: str | None = None, model_id: str | None = None
) -> bytes | None:
    text = (text or "").strip()
    if not text:
        return None

    if not ELEVENLABS_AVAILABLE or ElevenLabs is None:
        return None

    api_key = st.secrets.get("ELEVENLABS_API_KEY", None)
    if not api_key:
        return None

    vid = voice_id or DEFAULT_ELEVEN_VOICE_ID
    mid = model_id or DEFAULT_ELEVEN_MODEL_ID

    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=vid,
            model_id=mid,
        )

        # ElevenLabs SDK may return bytes OR an iterator/generator of byte chunks.
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)

        # If it's an iterator/generator of chunks, join them.
        try:
            return b"".join(audio)
        except TypeError:
            # Last resort: try reading if it's file-like
            try:
                return audio.read()
            except Exception:
                return None

    except Exception:
        return None

def tts_bytes(
    text: str, lang_code: str, *, prefer_elevenlabs: bool
) -> tuple[bytes | None, str]:
    """
    Returns (audio_bytes, mime_format)
    """
    if prefer_elevenlabs:
        b = tts_elevenlabs_bytes(text)
        if b:
            return b, "audio/mpeg"  # ElevenLabs commonly returns mp3

    b2 = tts_espeak_wav_bytes(text, lang_code)
    if b2:
        return b2, "audio/wav"

    return None, "audio/wav"
# -----------------------
# Session state
# -----------------------
st.session_state.setdefault("text_phrase", "")
st.session_state.setdefault("audio_lang", None)
st.session_state.setdefault("last_audio_sig", None)
st.session_state.setdefault("live_mode", False)
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_live_text", "")
st.session_state.setdefault("current_result", None)
st.session_state.setdefault("sfw_mode", True)
st.session_state.setdefault("last_browser_live_sig", None)
st.session_state.setdefault("last_browser_oneshot_sig", None)

do_rerun = False
rerun_sleep_s = 0.0

# -----------------------
# Level meter helpers (sounddevice path only)
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
    device_index: int | None,
    seconds: float,
    fs: int = 16000,
    block_ms: int = 50,
    meter_placeholder=None,
    meter_label: str = "Level",
) -> np.ndarray:
    if (not SD_AVAILABLE) or (sd is None) or (device_index is None):
        return np.array([], dtype=np.float32)

    blocksize = max(1, int(fs * block_ms / 1000))
    frames: list[np.ndarray] = []
    t0 = time.time()

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=int(device_index),
        ) as stream:
            while True:
                elapsed = time.time() - t0
                if elapsed >= seconds:
                    break

                block, _overflowed = stream.read(blocksize)
                block = np.asarray(block).reshape(-1)
                frames.append(block)

                if meter_placeholder is not None:
                    meter_val = rms_to_meter(rms_level(block))
                    meter_placeholder.progress(
                        meter_val, text=f"{meter_label}: {meter_val:.2f}"
                    )
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

    def g2p_fn(span_text: str) -> str:
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

    top_rows: list[dict] = []
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


def compute_and_store(text: str, chosen_override_lang: str | None, *, push: bool = True):
    text = (text or "").strip()
    if not text:
        st.session_state["current_result"] = None
        return
    result = compute_results(text, chosen_override_lang)
    st.session_state["current_result"] = result
    if push and result.get("ok"):
        push_history(result)


# -----------------------
# Mode selector UI (only once)
# -----------------------
MODES_ALL = ["Listen (live-ish)", "Record (one-shot)", "Upload audio"]
MODES_UPLOAD_ONLY = ["Upload audio"]

MIC_ANY_AVAILABLE = bool(BROWSER_MIC_AVAILABLE) or bool(SD_AVAILABLE)
if MIC_ANY_AVAILABLE:
    mode = st.selectbox("Mode", MODES_ALL, index=0)
else:
    st.warning("Mic input is unavailable in this environment, only Upload mode is enabled.")
    mode = st.selectbox("Mode", MODES_UPLOAD_ONLY, index=0)

# Local mic selector (optional, local dev only)
device_index: int | None = None
if SD_AVAILABLE and sd is not None:
    device_index = get_default_input_device_index()
    with st.expander("Local mic settings (advanced)", expanded=False):
        try:
            devices = sd.query_devices()
            input_devices = [
                (i, d.get("name", f"Device {i}"))
                for i, d in enumerate(devices)
                if int(d.get("max_input_channels", 0) or 0) > 0
            ]
        except Exception:
            input_devices = []

        if input_devices:
            default_k = 0
            if device_index is not None:
                for k, (i, _name) in enumerate(input_devices):
                    if int(i) == int(device_index):
                        default_k = k
                        break

            choice = st.selectbox(
                "Select local microphone",
                input_devices,
                index=default_k,
                format_func=lambda x: f"{x[0]}: {x[1]}",
            )
            device_index = int(choice[0])
        else:
            st.info("No local input devices detected by sounddevice.")

# -----------------------
# G2P language override UI + SFW mode
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

st.session_state["sfw_mode"] = st.toggle(
    "SFW mode (hide severity 3+)",
    value=bool(st.session_state.get("sfw_mode", True)),
)

prefer_elevenlabs = st.toggle(
    "Use ElevenLabs for voice (fallback to espeak)",
    value=False,
    disabled=not ELEVENLABS_AVAILABLE,
)

if prefer_elevenlabs and not st.secrets.get("ELEVENLABS_API_KEY", None):
    st.warning("ElevenLabs toggle is on, but ELEVENLABS_API_KEY is missing in Streamlit secrets.")
# -----------------------
# Mode: Upload
# -----------------------
if mode == "Upload audio":
    st.subheader("Upload audio")
    colU1, colU2, colU3 = st.columns([2, 1, 1])

    with colU1:
        audio_file = st.file_uploader(
            "Upload a .wav, .mp3, or .m4a",
            type=["wav", "mp3", "m4a"],
            key="uploader_audio",
        )

    with colU2:
        process_clicked = st.button("Process upload", use_container_width=True)

    with colU3:
        if st.button("Clear upload state", use_container_width=True):
            st.session_state["last_audio_sig"] = None
            st.session_state["audio_lang"] = None
            st.session_state["text_phrase"] = ""
            st.session_state["current_result"] = None
            st.session_state["processed_upload_sig"] = None
            st.rerun()

    # track what we've already processed, across reruns
    st.session_state.setdefault("processed_upload_sig", None)

    if audio_file is not None:
        audio_sig = (audio_file.name, audio_file.size)

        # show the file we currently have loaded
        st.caption(f"Loaded: {audio_file.name} ({audio_file.size} bytes)")

        # only process when user clicks, and only once per unique file
        if process_clicked and st.session_state["processed_upload_sig"] != audio_sig:
            st.session_state["processed_upload_sig"] = audio_sig

            with tempfile.NamedTemporaryFile(
                delete=False, suffix="." + audio_file.name.split(".")[-1]
            ) as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = tmp.name

            with st.spinner("Transcribing upload..."):
                text_from_audio, lang_from_audio = transcribe_audio_file(tmp_path, model_name="base")

            text_from_audio = (text_from_audio or "").strip()

            st.session_state["text_phrase"] = text_from_audio
            st.session_state["audio_lang"] = lang_from_audio

            # compute immediately so UI updates without needing rerun loops
            compute_and_store(text_from_audio, chosen_override_lang, push=True)

            st.success("Processed.")
            st.rerun()

    # display current state (whether processed or typed)
    if st.session_state.get("text_phrase"):
        st.write("Transcribed text:", st.session_state["text_phrase"])
        st.write("Whisper language:", st.session_state.get("audio_lang"))

# -----------------------
# Mode: Record one-shot
# -----------------------
if mode == "Record (one-shot)":
    st.subheader("One-shot recording")

    backends: list[str] = []
    if BROWSER_MIC_AVAILABLE and mic_recorder is not None:
        backends.append("Browser mic (recommended)")
    if SD_AVAILABLE and device_index is not None:
        backends.append("Local mic (sounddevice)")

    if not backends:
        st.error(
            "No mic backend available. Install streamlit-mic-recorder or sounddevice."
        )
    else:
        backend = st.radio("Recording backend", backends, index=0, horizontal=True)

        if backend.startswith("Browser"):
            st.caption("Click Record, then Stop to process.")
            audio = mic_recorder(
            start_prompt="Record",
            stop_prompt="Stop and process",
            just_once=True,
            use_container_width=True,
            key="browser_mic_oneshot",
        )

        if audio and isinstance(audio, dict) and audio.get("bytes"):
            b = audio["bytes"]
            sig = hashlib.sha256(b).hexdigest()

            # only transcribe once per unique recording
            if st.session_state.get("last_browser_oneshot_sig") != sig:
                st.session_state["last_browser_oneshot_sig"] = sig

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(b)
                    wav_path = tmp.name

                with st.spinner("Transcribing..."):
                    text_from_audio, lang_from_audio = transcribe_audio_file(wav_path, model_name="base")

                text_from_audio = (text_from_audio or "").strip()
                st.session_state["text_phrase"] = text_from_audio
                st.session_state["audio_lang"] = lang_from_audio

                compute_and_store(text_from_audio, chosen_override_lang, push=True)
                st.rerun()

        else:
            dur_s = st.slider("Record seconds", 1.0, 8.0, 3.0, 0.5, key="oneshot_dur")
            rel_thr = st.slider(
                "Silence sensitivity (relative)", 0.02, 0.20, 0.08, 0.01, key="oneshot_rel"
            )
            abs_thr = st.slider(
                "Silence floor (absolute)", 0.0003, 0.0050, 0.0010, 0.0001, key="oneshot_abs"
            )

            if st.button("Record now", key="oneshot_record_btn"):
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
                else:
                    x_trim = trim_silence(x, fs, rel_thresh=float(rel_thr), abs_thresh=float(abs_thr))
                    if x_trim.size == 0:
                        st.error("Only silence detected. Try again or change mic.")
                    else:
                        wav_path = write_temp_wav(x_trim, fs=fs)
                        with st.spinner("Transcribing..."):
                            text_from_audio, lang_from_audio = transcribe_audio_file(
                                wav_path, model_name="base"
                            )

                        text_from_audio = (text_from_audio or "").strip()
                        st.session_state["text_phrase"] = text_from_audio
                        st.session_state["audio_lang"] = lang_from_audio

                        compute_and_store(text_from_audio, chosen_override_lang, push=True)
                        st.rerun()

# -----------------------
# Mode: Listen live-ish
# -----------------------
if mode == "Listen (live-ish)":
    st.subheader("Live-ish listening")

    backends: list[str] = []
    if BROWSER_MIC_AVAILABLE and mic_recorder is not None:
        backends.append("Browser mic (recommended)")
    if SD_AVAILABLE and device_index is not None:
        backends.append("Local mic (sounddevice)")

    if not backends:
        st.error("No mic backend available. Install streamlit-mic-recorder or sounddevice.")
    else:
        backend = st.radio("Listening backend", backends, index=0, horizontal=True)

        colA, colB, colC = st.columns([1, 1, 2])
        if colA.button("Start listening", key="live_start"):
            st.session_state["live_mode"] = True
            st.rerun()
        if colB.button("Stop listening", key="live_stop"):
            st.session_state["live_mode"] = False
        if colC.button("Clear history", key="live_clear_hist"):
            st.session_state["history"] = []

        if not st.session_state.get("live_mode", False):
            st.info("Click Start listening to begin.")
        else:
            fs = 16000

            if backend.startswith("Browser"):
                audio = mic_recorder(
                start_prompt="Record chunk",
                stop_prompt="Stop and process",
                just_once=False,
                use_container_width=True,
                key="browser_mic_live",
            )

            if audio and isinstance(audio, dict) and audio.get("bytes"):
                b = audio["bytes"]
                sig = hashlib.sha256(b).hexdigest()

                # only transcribe when the audio bytes change
                if st.session_state.get("last_browser_live_sig") != sig:
                    st.session_state["last_browser_live_sig"] = sig

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(b)
                        wav_path = tmp.name

                    with st.spinner("Transcribing..."):
                        text_from_audio, lang_from_audio = transcribe_audio_file(wav_path, model_name="base")

                    text_from_audio = (text_from_audio or "").strip()
                    if text_from_audio and text_from_audio != st.session_state.get("last_live_text", ""):
                        st.session_state["last_live_text"] = text_from_audio
                        st.session_state["text_phrase"] = text_from_audio
                        st.session_state["audio_lang"] = lang_from_audio

                        compute_and_store(text_from_audio, chosen_override_lang, push=True)
                        st.rerun()

            else:
                colL, colR = st.columns([1, 1])
                with colL:
                    chunk_s = st.slider("Chunk seconds", 1.0, 4.0, 2.0, 0.5, key="live_chunk_s")
                with colR:
                    pause_s = st.slider("Pause between chunks", 0.0, 1.0, 0.2, 0.1, key="live_pause_s")

                rel_thr = st.slider(
                    "Silence sensitivity (relative)", 0.02, 0.20, 0.08, 0.01, key="live_rel"
                )
                abs_thr = st.slider(
                    "Silence floor (absolute)", 0.0003, 0.0050, 0.0010, 0.0001, key="live_abs"
                )

                level_box = st.empty()
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
                    with st.spinner("Transcribing..."):
                        text_from_audio, lang_from_audio = transcribe_audio_file(
                            wav_path, model_name="base"
                        )

                    text_from_audio = (text_from_audio or "").strip()
                    if text_from_audio and text_from_audio != st.session_state.get("last_live_text", ""):
                        st.session_state["last_live_text"] = text_from_audio
                        st.session_state["text_phrase"] = text_from_audio
                        st.session_state["audio_lang"] = lang_from_audio

                        compute_and_store(text_from_audio, chosen_override_lang, push=True)

                do_rerun = True
                rerun_sleep_s = float(pause_s)

# -----------------------
# Results section (always renders)
# -----------------------
st.subheader("Input text")
text = st.text_input("Type a phrase", key="text_phrase")

# If user edits text manually, recompute (no history push for manual edits)
if text and text.strip():
    cached = st.session_state.get("current_result")
    if not (cached and cached.get("text", "") == text.strip()):
        compute_and_store(text.strip(), chosen_override_lang, push=False)

result = st.session_state.get("current_result")

if result:
    st.subheader("Result")
    if not result.get("ok"):
        st.warning("No IPA produced, try a different input or language.")
    else:
        why = result["why"]
        st.subheader("Top matches")

        rows_all = list(result.get("top") or [])

        # normalize types for display
        for r in rows_all:
            r["distance"] = float(r.get("distance", 0.0))
            r["similarity"] = float(r.get("similarity", 0.0))
            sev = r.get("severity", "")
            r["severity"] = int(sev) if str(sev).strip().isdigit() else 0
            if "display" not in r:
                r["display"] = r.get("word", "")

        sfw_on = bool(st.session_state.get("sfw_mode", True))
        if sfw_on:
            rows = [r for r in rows_all if int(r.get("severity", 0)) < 3]
        else:
            rows = rows_all

        if not rows:
            st.warning("SFW mode hid all matches for this input.")
        else:
            top1 = rows[0]

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

            # -----------------------
            # TTS + full width table (NOT inside the right column)
            # -----------------------
            colT1, colT2 = st.columns([1, 2])
            with colT1:
                if st.button("Speak top match", key="speak_top_match"):
                    audio_bytes, audio_fmt = tts_bytes(top1["word"], top1["lang"], prefer_elevenlabs=prefer_elevenlabs)
                    if audio_bytes:
                        st.audio(audio_bytes, format=audio_fmt)
                    else:
                        st.warning("TTS failed for this word/voice.")

            with colT2:
                st.caption("Uses espeak voices, some languages may not be installed or may differ.")

            # IMPORTANT: table is rendered OUTSIDE the columns, so it can span full width
            df = pd.DataFrame(rows)

            cols = [
                "display",
                "word",
                "lang",
                "meaning",
                "severity",
                "ipa",
                "source_token",
                "best_window",
                "distance",
                "similarity",
            ]
            cols = [c for c in cols if c in df.columns]
            df = df[cols]

            df = df.fillna("").astype(str).applymap(html.escape)
            table_html = df.to_html(index=False, escape=False)

            st.markdown(
                """
                <style>
                .curser-table-wrap {
                    width: 100%;
                    max-width: 100%;
                    background: #0f1117;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px;
                    padding: 10px;
                    overflow-x: auto;
                    box-sizing: border-box;
                }
                .curser-table-wrap table {
                    width: 100%;
                    min-width: 900px;
                    border-collapse: collapse;
                    background: #0f1117;
                    color: #ffffff;
                    font-family: var(--ui-font) !important;
                    font-size: 12px;
                }
                .curser-table-wrap th {
                    background: #12141a;
                    color: #ffffff;
                    text-align: left;
                    padding: 8px 10px;
                    border-bottom: 1px solid rgba(255,255,255,0.12);
                    position: sticky;
                    top: 0;
                    z-index: 1;
                }
                .curser-table-wrap td {
                    background: #0f1117;
                    color: #ffffff;
                    padding: 8px 10px;
                    border-bottom: 1px solid rgba(255,255,255,0.08);
                    vertical-align: top;
                    white-space: nowrap;
                }
                .curser-table-wrap tr:hover td {
                    background: rgba(255,70,70,0.12);
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="curser-table-wrap" dir="auto">{table_html}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Speak a match"):
                st.write("Click Speak to hear the selected word.")
                for i, r in enumerate(rows[:10]):
                    shown = r.get("display", r.get("word", ""))
                    raw = r.get("word", "")
                    r_lang = r.get("lang", "en")

                    cA, cB, cC, cD = st.columns([1.2, 0.8, 3.0, 1.0])
                    with cA:
                        st.write(f"**{shown}**")
                    with cB:
                        st.write(r_lang)
                    with cC:
                        st.write(r.get("meaning", ""))
                    with cD:
                        if st.button("Speak", key=f"speak_row_{i}_{raw}_{r_lang}"):
                            # DEBUG: confirm which word is being sent to TTS
                            st.caption(f"DEBUG speaking: '{raw}' ({r_lang})")
                            audio_bytes, audio_fmt = tts_bytes(
                                raw,
                                r_lang,
                                prefer_elevenlabs=prefer_elevenlabs,
                            )
                            if audio_bytes:
                                st.audio(audio_bytes, format=audio_fmt)
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

        export_top = rows if bool(st.session_state.get("sfw_mode", True)) else rows_all
        export_obj = {
            "text": result.get("text"),
            "best_span": result.get("best_span"),
            "g2p_lang": result.get("g2p_lang"),
            "ipa": result.get("ipa"),
            "ipa_norm": result.get("ipa_norm"),
            "top": export_top,
            "why": result.get("why"),
            "sfw_mode": bool(st.session_state.get("sfw_mode", True)),
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
# History (always renders)
# -----------------------
st.subheader("History (latest first)")
if not st.session_state["history"]:
    st.write("No saved results yet.")
else:
    sfw_on = bool(st.session_state.get("sfw_mode", True))

    for item in st.session_state["history"]:
        top1 = item.get("top1") or {}
        sev = top1.get("severity", 0)
        try:
            sev_int = int(sev)
        except Exception:
            sev_int = 0

        if sfw_on and sev_int >= 3:
            shown = "[hidden]"
            lang = ""
        else:
            shown = top1.get("display", top1.get("word", ""))
            lang = top1.get("lang", "")

        title = f"{item['ts']} | {shown} [{lang}] | {item.get('text','')}"
        with st.expander(title):
            st.write("Text:", item.get("text"))
            st.write("Best span:", item.get("best_span"))
            st.write("G2P lang:", item.get("g2p_lang"))
            st.write("IPA:", item.get("ipa"))
            if sfw_on and sev_int >= 3:
                st.write("Top1: hidden by SFW mode")
            else:
                st.write("Top1:", top1)

    hist_json = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2)
    st.download_button(
        "Download full history JSON",
        data=hist_json,
        file_name="curserio_history.json",
        mime="application/json",
    )

# -----------------------
# Rerun scheduling for local Listen mode only
# -----------------------
if do_rerun and st.session_state.get("live_mode", False) and mode == "Listen (live-ish)":
    time.sleep(rerun_sleep_s)
    st.rerun()