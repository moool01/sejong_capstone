# strip_meta.py
import json, sys

inp = "preprocessing/dataset_sejong/afterprepro2/final_dataset.json"
out = "preprocessing/dataset_sejong/afterprepro2/final_dataset2json"
DROP = {"lang", "uid", "created_at"}  # 필요하면 여기서 조절

with open(inp, encoding="utf-8") as f:
    data = json.load(f)

clean = [{k: v for k, v in item.items() if k not in DROP} for item in data]

with open(out, "w", encoding="utf-8") as f:
    json.dump(clean, f, ensure_ascii=False, indent=2)

print(f"Saved -> {out} (items={len(clean)})")
