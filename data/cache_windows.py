from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import mne


# -----------------------
# Config
# -----------------------
SEIZURE_SUBSTR = "sz"   # matches cpsz, spsz, fnsz, gnsz, etc.


def read_seizure_intervals_from_csv(csv_path: str | Path) -> List[Tuple[float, float]]:
    """
    Reads TUH-style CSV annotation file and returns seizure intervals as:
        [(start_sec, stop_sec), ...]

    Expected columns:
        channel,start_time,stop_time,label,confidence
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(
        csv_path,
        comment="#",
        dtype={"start_time": float, "stop_time": float, "label": str},
        usecols=["start_time", "stop_time", "label"],
    )

    labels = df["label"].astype(str).str.strip().str.lower()
    is_sz = labels.str.contains(SEIZURE_SUBSTR, na=False)

    intervals: List[Tuple[float, float]] = []
    for s, e in zip(df.loc[is_sz, "start_time"], df.loc[is_sz, "stop_time"]):
        if pd.isna(s) or pd.isna(e):
            continue
        s = float(s)
        e = float(e)
        if e > s:
            intervals.append((s, e))

    # Merge overlapping intervals
    intervals.sort()
    merged: List[List[float]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(float(a), float(b)) for a, b in merged]


def window_overlaps_any_interval(
    ws: float,
    we: float,
    intervals: List[Tuple[float, float]],
) -> bool:
    """
    Returns True if window [ws, we] overlaps any seizure interval.
    """
    for s, e in intervals:
        if ws < e and we > s:
            return True
    return False


def load_full_edf_all_channels(
    edf_path: str | Path,
    fs: int,
    c_max: int,
) -> Tuple[np.ndarray, int]:
    """
    Load EEG channels from EDF, resample if needed, normalize per channel,
    and pad/truncate to c_max channels.

    Returns:
        x: np.ndarray of shape [c_max, T], dtype float32
        T: number of time samples
    """
    edf_path = str(edf_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick("eeg")

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")

    data = raw.get_data().astype(np.float32)  # [C_i, T]

    # Normalize per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    data = (data - mean) / std

    c_i, T = data.shape
    x = np.zeros((c_max, T), dtype=np.float32)
    c = min(c_i, c_max)
    x[:c] = data[:c]

    return x, T


def cache_one_record_windows(
    rec: Dict[str, Any],
    out_dir: str | Path,
    fs: int = 250,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    c_max: int = 41,
    max_windows: Optional[int] = None,
) -> Optional[Tuple[Path, int]]:
    """
    Create cached windows for one record and save:
        out_dir/<stem>.pt

    Saved structure:
        {
            "x": [N, C, T],
            "y": [N],
            "meta": {...}
        }

    Returns:
        (out_path, n_windows) or None if skipped
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_path = Path(rec["edf_path"])
    csv_path = Path(rec["csv_path"])
    stem = rec.get("stem") or edf_path.stem

    if not edf_path.exists():
        print(f"[SKIP] missing EDF: {edf_path}")
        return None

    if not csv_path.exists():
        print(f"[SKIP] missing CSV: {csv_path}")
        return None

    seizure_intervals = read_seizure_intervals_from_csv(csv_path)

    # Load full signal once
    x_full, T_full = load_full_edf_all_channels(
        edf_path=edf_path,
        fs=fs,
        c_max=c_max,
    )

    win_T = int(fs * window_sec)
    stride_T = int(fs * stride_sec)

    if win_T <= 0 or stride_T <= 0:
        raise ValueError("window_sec and stride_sec must be > 0")

    xs: List[torch.Tensor] = []
    ys: List[int] = []

    max_start = max(0, T_full - win_T)
    starts = list(range(0, max_start + 1, stride_T))

    if max_windows is not None:
        starts = starts[:max_windows]

    for st in starts:
        en = st + win_T
        ws = st / fs
        we = en / fs

        xw = x_full[:, st:en]  # [C, win_T]
        if xw.shape[1] != win_T:
            continue

        y = 1 if window_overlaps_any_interval(ws, we, seizure_intervals) else 0

        xs.append(torch.from_numpy(xw))
        ys.append(y)

    if len(xs) == 0:
        print(f"[SKIP] no windows: {stem}")
        return None

    X = torch.stack(xs, dim=0)             # [N, C, T]
    Y = torch.tensor(ys, dtype=torch.long) # [N]

    out_path = out_dir / f"{stem}.pt"

    torch.save(
        {
            "x": X,
            "y": Y,
            "meta": {
                "stem": stem,
                "edf_path": str(edf_path),
                "csv_path": str(csv_path),
                "subject": rec.get("subject"),
                "session": rec.get("session"),
                "montage": rec.get("montage"),
                "fs": fs,
                "window_sec": window_sec,
                "stride_sec": stride_sec,
                "c_max": c_max,
            },
        },
        out_path,
    )

    n_windows = len(xs)
    return out_path, n_windows


def build_cache_from_json(
    json_path: str | Path,
    out_dir: str | Path,
    fs: int = 250,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    c_max: int = 41,
    max_records: Optional[int] = None,
    max_windows_per_record: Optional[int] = None,
) -> Path:
    """
    Build cached .pt files for many records and write a JSONL manifest.

    Each manifest line looks like:
        {"pt_path": "...", "n": 123}
    """
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if max_records is not None:
        records = records[:max_records]

    manifest_path = out_dir / "manifest.jsonl"
    n_ok = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, rec in enumerate(records, start=1):
            result = cache_one_record_windows(
                rec=rec,
                out_dir=out_dir,
                fs=fs,
                window_sec=window_sec,
                stride_sec=stride_sec,
                c_max=c_max,
                max_windows=max_windows_per_record,
            )

            if result is None:
                continue

            out_pt, n = result
            n_ok += 1

            manifest_entry = {
                "pt_path": str(out_pt),
                "n": n,
            }
            mf.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

            print(f"[{i}/{len(records)}] cached: {out_pt.name} ({n} windows)")

    print(f"Done. Cached {n_ok} recordings.")
    print(f"Manifest: {manifest_path}")

    return manifest_path


if __name__ == "__main__":
    # -----------------------
    # Edit these paths
    # -----------------------
    seizure_only_json = "assets/eeg_seizure_only_eval.json"
    cache_dir = "cache_windows_eval"

    # -----------------------
    # Cache settings
    # -----------------------
    manifest = build_cache_from_json(
        json_path=seizure_only_json,
        out_dir=cache_dir,
        fs=250,
        window_sec=10.0,
        stride_sec=5.0,
        c_max=41,
        max_records=None,             # e.g. 50 for quick test
        max_windows_per_record=None,  # e.g. 100 for quick test
    )