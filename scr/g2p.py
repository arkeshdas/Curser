# g2p.py
import subprocess
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
import re

LANG_MAP = {
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

# removes: (en-us) flags, any (...) metadata, zero-width junk, stray newlines
JUNK_RE = re.compile(r"\([^)]*\)|[\u200b-\u200f\u2060\uFEFF]")

def _run_espeak(args: list[str]) -> str:
    p = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )
    # if espeak prints to stderr, include it in error
    if p.returncode != 0:
        raise RuntimeError(f"espeak failed: {p.stderr.strip()}")
    return (p.stdout or "").strip()

def _scrub(s: str) -> str:
    s = JUNK_RE.sub("", s)
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # extra safety: remove embedded lang tags like en-us even if not parenthesized
    s = re.sub(r"\b[a-z]{2,3}-[a-z]{2,4}\b", "", s, flags=re.IGNORECASE).strip()
    return s

# initialize once
_KO_ROMANIZER = Transliter(academic)

def text_to_ipa(text: str, lang: str = "en") -> str:
    voice = LANG_MAP.get(lang, "en-us")

    # Primary: real IPA output
    out = _run_espeak(["espeak", "-v", voice, "--ipa=3", "-q", text])
    out = _scrub(out)
    if out:
        return out

    # Korean fallback: romanize then IPA under English voice
    if lang == "ko":
        romanized = _KO_ROMANIZER.translit(text).strip()
        if romanized:
            out = _run_espeak(["espeak", "-v", "en-us", "--ipa=3", "-q", romanized])
            return _scrub(out)

    return ""

