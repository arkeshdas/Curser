# build_db.py
import json
from g2p import text_to_ipa
from core import normalize_ipa

IN_PATH = "db_seed.json"
OUT_PATH = "db.json"

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    out = []
    for e in entries:
        word = e["word"]
        lang = e["lang"]

        ipa = text_to_ipa(word, lang=lang)
        ipa_norm = normalize_ipa(ipa)

        e2 = dict(e)
        e2["ipa"] = ipa
        e2["ipa_norm"] = ipa_norm
        out.append(e2)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} entries to {OUT_PATH}")

if __name__ == "__main__":
    main()