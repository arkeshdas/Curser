# g2p.py
import subprocess

LANG_MAP = {
    "en": "en-us",
    "es": "es",
    "fr": "fr",
    "de": "de",
}

def text_to_ipa(text: str, lang: str = "en") -> str:
    voice = LANG_MAP.get(lang, "en-us")

    # --ipa=3 gives IPA output, -q = quiet
    p = subprocess.run(
        ["espeak", "-v", voice, "--ipa=3", "-q", text],
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout.strip()