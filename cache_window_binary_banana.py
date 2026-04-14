from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import mne


# =========================================================
# CONFIGURATION
# =========================================================

# The "Double Banana" Bipolar Montage (18 pairs)
BIPOLAR_MONTAGE = [
    ("fp1", "f7"), ("f7", "t3"), ("t3", "t5"), ("t5", "o1"),  # Left temporal chain
    ("fp2", "f8"), ("f8", "t4"), ("t4", "t6"), ("t6", "o2"),  # Right temporal chain
    ("fp1", "f3"), ("f3", "c3"), ("c3", "p3"), ("p3", "o1"),  # Left parasagittal chain
    ("fp2", "f4"), ("f4", "c4"), ("c4", "p4"), ("p4", "o2"),  # Right parasagittal chain
    ("fz", "cz"), ("cz", "pz")                                # Midline chain
]

@dataclass
class CacheConfig:
    json_path: str
    out_dir: str

    fs: int = 250
    window_sec: float = 10.0
    stride_sec: float = 5.0

    max_records: Optional[int] = None
    max_windows_per_record: Optional[int] = None

    l_freq: Optional[float] = 0.5
    h_freq: Optional[float] = 40.0   # FIXED: was 25.0, must match MelSpec f_max

    # Labels
    background_labels: Tuple[str, ...] = ("bckg", "background")
    seizure_label_name: str = "seizure"
    non_seizure_label_name: str = "non_seizure"

    # Quality & Scaling Parameters (UPDATED FOR BIPOLAR)
    # EDF data is in Volts. 1e-6 Volts = 1 Microvolt. 
    flat_std_thresh: float = 1e-6 
    max_flat_ratio: float = 0.3
    max_zero_ratio: float = 0.3

    # Robust normalization (Median + IQR, outlier-resistant)
    per_channel_normalize: bool = True
    clip_percentile: float = 2.0   # Clip at 2nd and 98th percentile before normalization


# =========================================================
# LABELS
# =========================================================

def to_binary_label(raw_label: str, background_labels: Tuple[str, ...]) -> str:
    raw_label = str(raw_label).strip().lower()
    if raw_label in set(background_labels):
        return "non_seizure"
    return "seizure"

def build_binary_label_vocab(
    seizure_label_name: str = "seizure",
    non_seizure_label_name: str = "non_seizure",
):
    labels = [non_seizure_label_name, seizure_label_name]
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    return label_to_id, id_to_label


# =========================================================
# CSV LABEL INTERVALS
# =========================================================

def read_label_intervals_from_csv(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"start_time", "stop_time", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must contain columns: start_time, stop_time, label")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")

    intervals = []
    for _, row in df.iterrows():
        s = row["start_time"]
        e = row["stop_time"]
        lab = row["label"]

        if pd.isna(s) or pd.isna(e) or pd.isna(lab) or e <= s:
            continue

        intervals.append({
            "start_time": float(s),
            "stop_time": float(e),
            "label": lab,
        })

    return intervals

def assign_raw_label_by_overlap(
    ws: float,
    we: float,
    intervals: List[Dict[str, Any]],
    default: str = "bckg",
    # UPDATED: Require 40% of the window to be a seizure to count it
    seizure_priority_threshold: float = 0.4, 
) -> str:
    overlap_by_label: Dict[str, float] = {}
    window_duration = we - ws

    for item in intervals:
        s = item["start_time"]
        e = item["stop_time"]
        lab = item["label"].lower()

        overlap = max(0.0, min(we, e) - max(ws, s))
        if overlap > 0:
            overlap_by_label[lab] = overlap_by_label.get(lab, 0.0) + overlap
            
    if not overlap_by_label:
        return default
        
    # Calculate total time that is completely unlabeled, assign it to default to prevent tiny seizures from dominating
    labeled_duration = sum(overlap_by_label.values())
    unlabeled_duration = max(0.0, window_duration - labeled_duration)
    if unlabeled_duration > 0:
        overlap_by_label[default] = overlap_by_label.get(default, 0.0) + unlabeled_duration

    # Look for seizures that hit the threshold
    for lab, overlap in overlap_by_label.items():
        if ("sz" in lab or "seiz" in lab) and (overlap / window_duration) >= seizure_priority_threshold:
            return lab 

    # Otherwise, pick the dominant label (which will likely be background if a seizure was < 40%)
    return max(overlap_by_label.items(), key=lambda x: x[1])[0]


# =========================================================
# CHANNEL / MONTAGE
# =========================================================

def normalize_channel_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("eeg ", "").replace("-ref", "").replace("-le", "").replace(" ", "")
    name = name.replace("t7", "t3").replace("t8", "t4").replace("p7", "t5").replace("p8", "t6")
    return re.sub(r"[^a-z0-9]", "", name)

def load_edf_signals(
    edf_path: str | Path,
    fs: int,
    l_freq: Optional[float],
    h_freq: Optional[float],
) -> Tuple[np.ndarray, List[str]]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick("eeg")

    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")

    data = raw.get_data().astype(np.float32)
    ch_names = list(raw.ch_names)
    return data, ch_names

def apply_bipolar_montage(
    data: np.ndarray,
    original_channels: List[str],
    bipolar_pairs: List[Tuple[str, str]]
):
    norm_original = [normalize_channel_name(ch) for ch in original_channels]
    idx_map = {ch: i for i, ch in enumerate(norm_original)}

    T = data.shape[1]
    bipolar_data = np.zeros((len(bipolar_pairs), T), dtype=np.float32)
    
    final_channels = []
    missing_channels = []

    for i, (ch1, ch2) in enumerate(bipolar_pairs):
        pair_name = f"{ch1}-{ch2}"
        final_channels.append(pair_name)

        if ch1 in idx_map and ch2 in idx_map:
            idx1, idx2 = idx_map[ch1], idx_map[ch2]
            bipolar_data[i] = data[idx1] - data[idx2]
        else:
            missing_channels.append(pair_name)
            bipolar_data[i] = np.zeros((T,), dtype=np.float32)

    meta = {
        "final_montage_channels": final_channels,
        "missing_channels": missing_channels,
        "original_channels": original_channels,
        "ignored_channels": []
    }
    return bipolar_data, meta


# =========================================================
# PREPROCESS / QUALITY
# =========================================================

def normalize_per_channel_robust(x: np.ndarray, clip_percentile: float = 2.0) -> np.ndarray:
    """
    Robust per-channel normalization using Median and IQR.
    
    Why this matters for TUSZ:
    Standard z-score uses mean/std which are DESTROYED by muscle artifacts.
    One electrode pop can inflate std by 10x, squashing real seizure signals.
    
    Median and IQR are rank-based statistics — a single outlier cannot move them.
    This ensures that a noisy eval recording normalizes to the SAME scale as a
    clean train recording, eliminating the #1 cause of the 75% ceiling.
    
    Steps:
    1. Clip extreme outliers at 2nd/98th percentiles (kills electrode pops)
    2. Center with median (robust to asymmetric noise distributions)
    3. Scale with IQR (Q75-Q25, robust to outlier-inflated variance)
    """
    # Step 1: Clip extreme values per channel
    low  = np.percentile(x, clip_percentile, axis=1, keepdims=True)
    high = np.percentile(x, 100 - clip_percentile, axis=1, keepdims=True)
    x_clipped = np.clip(x, low, high)
    
    # Step 2: Center using median (robust to skew)
    median = np.median(x_clipped, axis=1, keepdims=True)
    
    # Step 3: Scale using IQR (Q75 - Q25) — robust to outlier-inflated variance
    q25 = np.percentile(x_clipped, 25, axis=1, keepdims=True)
    q75 = np.percentile(x_clipped, 75, axis=1, keepdims=True)
    iqr = q75 - q25 + 1e-8  # avoid division by zero on flat channels
    
    return ((x_clipped - median) / iqr).astype(np.float32)

def is_bad_window(
    xw: np.ndarray,
    flat_std_thresh: float,
    max_flat_ratio: float,
    max_zero_ratio: float,
) -> bool:
    if not np.isfinite(xw).all():
        return True

    ch_std = xw.std(axis=1)
    flat_mask = ch_std < flat_std_thresh
    zero_mask = np.all(np.abs(xw) < 1e-8, axis=1)

    if flat_mask.mean() > max_flat_ratio:
        return True
    if zero_mask.mean() > max_zero_ratio:
        return True

    return False


# =========================================================
# CACHE ONE RECORD
# =========================================================

def cache_one_record_windows(
    rec: Dict[str, Any],
    out_dir: Path,
    cfg: CacheConfig,
    label_to_id: Dict[str, int],
) -> Optional[Tuple[Path, int, Dict[str, Any]]]:
    edf_path = Path(rec["edf_path"])
    csv_path = Path(rec["csv_path"])
    stem = rec.get("stem") or edf_path.stem

    if not edf_path.exists() or not csv_path.exists():
        return None

    intervals = read_label_intervals_from_csv(csv_path)

    raw_data, original_channels = load_edf_signals(
        edf_path=edf_path, fs=cfg.fs, l_freq=cfg.l_freq, h_freq=cfg.h_freq
    )

    # 1. Apply Bipolar Subtraction
    x_full, montage_meta = apply_bipolar_montage(
        data=raw_data, original_channels=original_channels, bipolar_pairs=BIPOLAR_MONTAGE
    )

    # 2. Convert Volts to Microvolts
    x_full = x_full * 1e6

    # 3. Apply Robust Normalization (Median + IQR)
    # This is the #1 change for breaking the 75% ceiling.
    # It replaces the old µV/100 global scaling which was vulnerable to
    # the massive amplitude shift between Train and Eval recordings.
    if cfg.per_channel_normalize:
        x_full = normalize_per_channel_robust(x_full, clip_percentile=cfg.clip_percentile)

    T_full = x_full.shape[1]
    win_T = int(cfg.fs * cfg.window_sec)
    stride_T = int(cfg.fs * cfg.stride_sec)

    starts = list(range(0, max(0, T_full - win_T) + 1, stride_T))
    if cfg.max_windows_per_record is not None:
        starts = starts[:cfg.max_windows_per_record]

    xs, ys = [], []
    raw_label_counts, final_label_counts = Counter(), Counter()
    bad_windows, skipped_unknown = 0, 0

    for st in starts:
        en = st + win_T
        if en > T_full: continue

        xw = x_full[:, st:en]
        
        # After robust normalization, data is on a standardized scale (median=0, IQR=1).
        # A flat/dead channel will have std ≈ 0 regardless of normalization.
        # Use a small fixed threshold appropriate for normalized data.
        adjusted_flat_thresh = 0.01 if cfg.per_channel_normalize else (cfg.flat_std_thresh * 1e6)
        
        if is_bad_window(
            xw, flat_std_thresh=adjusted_flat_thresh, max_flat_ratio=cfg.max_flat_ratio, max_zero_ratio=cfg.max_zero_ratio
        ):
            bad_windows += 1
            continue

        raw_label = assign_raw_label_by_overlap(st/cfg.fs, en/cfg.fs, intervals)
        final_label = to_binary_label(raw_label, cfg.background_labels)

        raw_label_counts[raw_label] += 1
        final_label_counts[final_label] += 1

        if final_label not in label_to_id:
            skipped_unknown += 1
            continue

        xs.append(torch.from_numpy(xw))
        ys.append(label_to_id[final_label])

    if len(xs) == 0:
        print(f"[SKIP] no valid windows: {stem}")
        return None

    X = torch.stack(xs, dim=0)
    Y = torch.tensor(ys, dtype=torch.long)
    out_path = out_dir / f"{stem}.pt"

    torch.save({
        "x": X, "y": Y,
        "meta": {
            "stem": stem, "fs": cfg.fs, "window_sec": cfg.window_sec,
            "label_mode": "binary", "bad_windows_skipped": bad_windows,
            **montage_meta,
        },
    }, out_path)

    report = {
        "stem": stem, "n_windows": len(xs), "bad_windows_skipped": bad_windows,
        "final_label_counts": dict(final_label_counts),
        "missing_channels": montage_meta["missing_channels"],
    }
    return out_path, len(xs), report


# =========================================================
# BUILD CACHE
# =========================================================

def build_cache_from_json(cfg: CacheConfig) -> Path:
    json_path = Path(cfg.json_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if cfg.max_records is not None:
        records = records[:cfg.max_records]

    label_to_id, id_to_label = build_binary_label_vocab(cfg.seizure_label_name, cfg.non_seizure_label_name)

    manifest_path = out_dir / "manifest.jsonl"
    n_cached_records, total_windows = 0, 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, rec in enumerate(records, start=1):
            result = cache_one_record_windows(rec, out_dir, cfg, label_to_id)
            if result:
                out_pt, n, report = result
                n_cached_records += 1
                total_windows += n
                mf.write(json.dumps({"pt_path": str(out_pt), "n": n}, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(records)}] cached: {out_pt.name} ({n} windows)")

    print(f"\nDone. Cached records: {n_cached_records} | Total windows: {total_windows}")
    return manifest_path


if __name__ == "__main__":
    # =============================================
    # RE-CACHE INSTRUCTIONS:
    # Run this TWICE — once for train, once for eval.
    # Uncomment the config you want to build.
    # =============================================

    # --- TRAIN ---
    cfg = CacheConfig(
        json_path=r"assets\eeg_seizure_only.json",
        out_dir=r"cache_windows_binary_10_sec",
        fs=256,
        window_sec=10,
        stride_sec=5,
        l_freq=0.5,
        h_freq=40.0,
    )

    # --- EVAL ---
    # cfg = CacheConfig(
    #     json_path=r"assets\eeg_seizure_only_eval.json",
    #     out_dir=r"cache_windows_binary_10_sec_eval",
    #     fs=256,
    #     window_sec=10,
    #     stride_sec=5,
    #     l_freq=0.5,
    #     h_freq=40.0,
    # )
    build_cache_from_json(cfg)