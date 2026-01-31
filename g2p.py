# g2p.py
import subprocess
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

JUNK_RE = re.compile(
    r"""
    \([^)]*\)            |  # parenthesized metadata
    [\u200b-\u200f]      |  # zero-width junk
    [\u2060]             |  # word joiner
    """,
    re.VERBOSE,
)

def text_to_ipa(text: str, lang: str = "en") -> str:
    voice = LANG_MAP.get(lang, "en-us")

    p = subprocess.run(
        [
            "espeak",
            "-v", voice,
            "--pho",        # raw phonemes only
            "-q",
            text,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    ipa = p.stdout.strip()

    # scrub junk aggressively
    ipa = JUNK_RE.sub("", ipa)
    ipa = ipa.replace("\n", " ").strip()

    return ipa