from pathlib import Path
import json
import torch

CACHE_DIR = Path("cache_windows_eval") # path to the data folder
MANIFEST = CACHE_DIR / "manifest.jsonl" # by defualt it should be there but if u move adjust this 

with MANIFEST.open("w", encoding="utf-8") as mf:
    for p in sorted(CACHE_DIR.glob("*.pt")):
        d = torch.load(p, map_location="cpu")
        n = int(d["x"].shape[0])

        rel = p.as_posix()
        mf.write(json.dumps({"pt_path": rel, "n": n}) + "\n")

print("Manifest written with relative paths:", MANIFEST)