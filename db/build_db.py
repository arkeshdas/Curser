# build_db.py
from pathlib import Path
import sys
import json

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.g2p import text_to_ipa
from src.core import normalize_ipa

HERE = Path(__file__).resolve().parent
IN_PATH = HERE / "db_seed.json"
OUT_PATH = HERE / "db.json"

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    out = []
    for e in entries:
        word = e["word"]              # canonical G2P input
        display = e.get("display")    # UI-only, optional safety
        lang = e["lang"]

        ipa = text_to_ipa(word, lang=lang)
        ipa_norm = normalize_ipa(ipa)

        e2 = dict(e)
        e2["ipa"] = ipa
        e2["ipa_norm"] = ipa_norm

        # Optional sanity check
        if not ipa:
            print(f"Warning: no IPA for {word} ({lang})")

        out.append(e2)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} entries to {OUT_PATH}")

if __name__ == "__main__":
    main()