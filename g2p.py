# g2p.py
from phonemizer import phonemize

LANG_MAP = {
    "en": "en-us",
    "es": "es",
    "fr": "fr-fr",
}

def text_to_ipa(text: str, lang: str = "en") -> str:
    espeak_lang = LANG_MAP.get(lang, "en-us")
    return phonemize(text, language=espeak_lang, backend="espeak", with_stress=True)