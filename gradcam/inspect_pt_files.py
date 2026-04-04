# inspect_pt_files.py  —  MindScope binary dataset
# Scans all 195 .pt files in the binary eval dataset on Drive,
# prints window counts and seizure peak-frequency breakdown.

import os
import json
import shutil
import tempfile
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataloader import STFTSpec, SpecZNorm
transform = nn.Sequential(
    STFTSpec(fs=250, n_fft=256, hop_length=64, fmin=1.0, fmax=40.0),
    SpecZNorm(),
)
transform.eval()

EVAL_DIR = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\cache_windows_eval"
MANIFEST = os.path.join(EVAL_DIR, "manifest.jsonl")

# Some Drive files are cloud-only stubs and throw Errno 22 on copy.
# Returns None if the file cannot be read (not downloaded / corrupt).
def load_pt(drive_path):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(drive_path, tmp_path)
        data = torch.load(tmp_path, map_location="cpu", weights_only=False)
        return data
    except OSError as e:
        return None   # cloud-only stub or unreadable — caller handles None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

def remap(pt_path_from_manifest):
    """Remap D:\\EEG_DATA\\... paths to the local Drive folder."""
    return os.path.join(EVAL_DIR, Path(pt_path_from_manifest).name)

def bin_to_hz(bin_idx):
    return round(bin_idx * (250 / 256) + 1.95)

def band_name(hz):
    if hz <= 4:   return "Delta (1-4 Hz)"
    if hz <= 8:   return "Theta (4-8 Hz)"
    if hz <= 13:  return "Alpha (8-13 Hz)"
    if hz <= 30:  return "Beta (13-30 Hz)"
    return "Gamma (30+ Hz)"

with open(MANIFEST) as f:
    entries = [json.loads(l) for l in f if l.strip()]

print(f"Found {len(entries)} entries in manifest")
print(f"Reading from: {EVAL_DIR}\n")

total_seizure    = 0
total_background = 0
total_missing    = 0
total_unreadable = 0

for entry in entries:
    pt_path = remap(entry["pt_path"])

    if not os.path.exists(pt_path):
        total_missing += 1
        continue

    data = load_pt(pt_path)
    if data is None:
        total_unreadable += 1
        print(f"  SKIPPED (cloud-only / unreadable): {Path(pt_path).name}")
        continue

    x_all = data["x"]   # [N, 41, 2500]
    y_all = data["y"]   # [N]

    n_total   = len(y_all)
    n_seizure = int(y_all.sum().item())
    n_bg      = n_total - n_seizure
    fname     = Path(pt_path).name

    total_seizure    += n_seizure
    total_background += n_bg

    print(f"File: {fname}")
    print(f"  Windows: {n_total} total  |  {n_seizure} seizure  |  {n_bg} background")

    if n_seizure > 0:
        seizure_peaks = []
        for i in range(n_total):
            if y_all[i].item() != 1:
                continue
            with torch.no_grad():
                spec = transform(x_all[i])
            mean_power = spec.numpy().mean(axis=(0, 2))
            seizure_peaks.append(bin_to_hz(int(mean_power.argmax())))

        avg_peak = round(np.mean(seizure_peaks))
        counts   = {}
        for hz in seizure_peaks:
            b = band_name(hz)
            counts[b] = counts.get(b, 0) + 1

        print(f"  Seizure peak frequencies — avg {avg_peak} Hz  "
              f"range {min(seizure_peaks)}–{max(seizure_peaks)} Hz")
        print(f"    By band: " +
              "  |  ".join(f"{b}: {c}" for b, c in sorted(counts.items())))
    print()

print(f"TOTAL  |  Seizure: {total_seizure}  |  Background: {total_background}  "
      f"|  Total: {total_seizure + total_background}")
if total_missing:
    print(f"Missing (not on Drive)  : {total_missing} files")
if total_unreadable:
    print(f"Skipped (cloud-only)    : {total_unreadable} files — "
          f"open Google Drive and make them available offline to fix this")