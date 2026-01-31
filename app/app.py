# app.py
import json
import tempfile
import time
from datetime import datetime

import streamlit as st
import numpy as np
import soundfile as sf
import sounddevice as sd
from langdetect import detect

from asr import transcribe_audio_file
from core import (
    tokenize,
    choose_best_span,
    windowed_panphon_distance_tokens,
    similarity_from_distance,
    normalize_ipa,
)
from g2p import text_to_ipa


def infer_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


@st.cache_data
def load_db():
    with open("db.json", "r", encoding="utf-8") as f:
        db = json.load(f)
    for e in db:
        if "ipa_norm" not in e:
            e["ipa_norm"] = normalize_ipa(e.get("ipa", ""))
    return db


DB = load_db()
st.title("Curserio, prototype")

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

# We will set this to True when Listen mode wants another loop
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


# -----------------------
# Recording helpers
# -----------------------
def write_temp_wav(x: np.ndarray, fs: int = 16000) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, x, fs)
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
        "candidate_word": best_e.get("word"),
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
mode = st.selectbox("Mode", ["Listen (live-ish)", "Record (one-shot)", "Upload audio"], index=0)

# Mic selector for listen + record
device_index = None
if mode in ("Listen (live-ish)", "Record (one-shot)"):
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
            st.markdown(f"**Top match:** `{top1['word']}`")
            st.markdown(f"**Lang:** `{top1['lang']}`")
            st.markdown(f"**Severity:** `{top1.get('severity','')}`")
        with c3:
            st.markdown(f"**Similarity:** `{top1['similarity']:.3f}`")
            st.markdown(f"**Distance:** `{top1['distance']:.3f}`")

        st.subheader("Top matches")
        rows = result["top"]

        for r in rows:
            r["distance"] = float(r.get("distance", 0.0))
            r["similarity"] = float(r.get("similarity", 0.0))
            sev = r.get("severity", "")
            r["severity"] = int(sev) if str(sev).strip().isdigit() else 0

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "distance": st.column_config.NumberColumn("distance", format="%.3f"),
                "similarity": st.column_config.NumberColumn("similarity", format="%.3f"),
                "severity": st.column_config.NumberColumn("severity", format="%d"),
            },
        )

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
        title = f"{item['ts']} | {top1.get('word','')} [{top1.get('lang','')}] | {item.get('text','')}"
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