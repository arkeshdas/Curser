# build_db.py
import json
from g2p import text_to_ipa

IN_PATH = "db_seed.json"
OUT_PATH = "db.json"

def main():
    with open(IN_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for e in entries:
        e["ipa"] = text_to_ipa(e["word"], lang=e["lang"])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries to {OUT_PATH}")

if __name__ == "__main__":
    main()