import json
from pathlib import Path
import pandas as pd

SEIZURE_SUBSTR = "sz"   # matches cpsz, spsz, fnsz, gnsz, etc.
BCKG_LABEL = "bckg"

def ratios_from_csv(csv_path: str | Path):
    df = pd.read_csv(
        csv_path,
        comment="#",  # skip metadata lines
        usecols=["start_time", "stop_time", "label"],
        dtype={"start_time": float, "stop_time": float, "label": str},
    )

    labels = df["label"].str.strip().str.lower()

    is_bckg = labels == BCKG_LABEL
    is_sz = labels.str.contains(SEIZURE_SUBSTR, na=False)

    # event counts
    bckg_events = int(is_bckg.sum())
    sz_events = int(is_sz.sum())

    # durations
    dur = (df["stop_time"] - df["start_time"]).clip(lower=0)
    bckg_secs = float(dur[is_bckg].sum())
    sz_secs = float(dur[is_sz].sum())

    return bckg_events, sz_events, bckg_secs, sz_secs


# ---- main ----
json_path = Path("eeg_seizure_only.json")

with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

total_bckg_events = 0
total_sz_events = 0
total_bckg_secs = 0.0
total_sz_secs = 0.0

missing_csv = 0

for rec in records:
    csv_path = Path(rec["csv_path"])
    if not csv_path.exists():
        missing_csv += 1
        continue

    b_e, s_e, b_s, s_s = ratios_from_csv(csv_path)
    total_bckg_events += b_e
    total_sz_events += s_e
    total_bckg_secs += b_s
    total_sz_secs += s_s

print(f"Missing CSV files: {missing_csv}")
print("---- Event counts ----")
print(f"Background events: {total_bckg_events}")
print(f"Seizure events   : {total_sz_events}")
if total_sz_events > 0:
    print(f"Event ratio (BCKG:SZ) = {total_bckg_events/total_sz_events:.2f}:1")
else:
    print("Event ratio (BCKG:SZ) = undefined (no seizure events found)")

print("\n---- Durations ----")
print(f"Background seconds: {total_bckg_secs:.2f}")
print(f"Seizure seconds   : {total_sz_secs:.2f}")
if total_sz_secs > 0:
    print(f"Duration ratio (BCKG:SZ) = {total_bckg_secs/total_sz_secs:.2f}:1")
else:
    print("Duration ratio (BCKG:SZ) = undefined (no seizure duration found)")
