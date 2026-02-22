import json
from pathlib import Path

json_path = Path("eeg_seizure_only.json")

with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

total_bytes = 0
missing = 0

for rec in records:
    edf_path = Path(rec["edf_path"])
    if edf_path.exists():
        total_bytes += edf_path.stat().st_size
    else:
        missing += 1

total_gb = total_bytes / (1024 ** 3)

print(f"EDF files counted : {len(records) - missing}")
print(f"Missing EDF files : {missing}")
print(f"Total EDF size   : {total_gb:.2f} GB")
