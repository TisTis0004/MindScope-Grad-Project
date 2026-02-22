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
SEIZURE_SUBSTR = "sz"    # matches cpsz, spsz, fnsz, gnsz, etc.
BCKG_LABEL = "bckg"


def read_seizure_intervals_from_csv(csv_path: str | Path) -> List[Tuple[float, float]]:
    """
    Reads TUH csv (with comment '#' header lines), returns seizure intervals (start, stop) in seconds.
    CSV format (as your screenshot):
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

    intervals = []
    for s, e in zip(df.loc[is_sz, "start_time"], df.loc[is_sz, "stop_time"]):
        if pd.isna(s) or pd.isna(e):
            continue
        s = float(s); e = float(e)
        if e > s:
            intervals.append((s, e))

    # merge overlaps (optional but good)
    intervals.sort()
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [(float(a), float(b)) for a, b in merged]


def window_overlaps_any_interval(ws: float, we: float, intervals: List[Tuple[float, float]]) -> bool:
    for s, e in intervals:
        if ws < e and we > s:  # overlap
            return True
    return False


def load_full_edf_all_channels(
    edf_path: str | Path,
    fs: int,
    C_max: int,
) -> Tuple[np.ndarray, int]:
    """
    Loads full EEG data, resamples to fs, returns:
      data: np.ndarray [C_max, T] float32 normalized per-channel
      T: number of samples
    """
    edf_path = str(edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick("eeg")

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")

    data = raw.get_data().astype(np.float32)  # [C_i, T]
    # normalize per channel
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)

    C_i, T = data.shape
    x = np.zeros((C_max, T), dtype=np.float32)
    c = min(C_i, C_max)
    x[:c] = data[:c]
    return x, T


def cache_one_record_windows(
    rec: Dict[str, Any],
    out_dir: str | Path,
    fs: int = 250,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    C_max: int = 41,
    max_windows: Optional[int] = None,
) -> Optional[Path]:
    """
    Creates window tensors for one record and saves:
      out_dir/<stem>.pt  containing {"x": [N,C,T], "y": [N], "meta": {...}}
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

    # load full signal once (CPU heavy, one-time)
    x_full, T_full = load_full_edf_all_channels(edf_path, fs=fs, C_max=C_max)

    win_T = int(fs * window_sec)
    stride_T = int(fs * stride_sec)
    if win_T <= 0 or stride_T <= 0:
        raise ValueError("window_sec and stride_sec must be > 0")

    # compute windows
    xs = []
    ys = []

    # number of windows
    max_start = max(0, T_full - win_T)
    starts = list(range(0, max_start + 1, stride_T))
    if max_windows is not None:
        starts = starts[:max_windows]

    for st in starts:
        en = st + win_T
        ws = st / fs
        we = en / fs

        xw = x_full[:, st:en]  # [C_max, win_T]
        if xw.shape[1] != win_T:
            continue

        y = 1 if window_overlaps_any_interval(ws, we, seizure_intervals) else 0
        xs.append(torch.from_numpy(xw))
        ys.append(y)

    if len(xs) == 0:
        print(f"[SKIP] no windows: {stem}")
        return None

    X = torch.stack(xs, dim=0)                       # [N, C, T]
    Y = torch.tensor(ys, dtype=torch.long)           # [N]

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
            },
        },
        out_path,
    )

    return out_path


def build_cache_from_json(
    json_path: str | Path,
    out_dir: str | Path,
    fs: int = 250,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    C_max: int = 41,
    max_records: Optional[int] = None,
    max_windows_per_record: Optional[int] = None,
) -> Path:
    """
    Saves many per-record .pt files + a manifest.jsonl listing them.
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
        for i, rec in enumerate(records, 1):
            out_pt = cache_one_record_windows(
                rec,
                out_dir=out_dir,
                fs=fs,
                window_sec=window_sec,
                stride_sec=stride_sec,
                C_max=C_max,
                max_windows=max_windows_per_record,
            )
            if out_pt is None:
                continue

            n_ok += 1
            mf.write(json.dumps({"pt_path": str(out_pt)}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(records)}] cached: {out_pt.name}")

    print(f"Done. Cached {n_ok} recordings.")
    print(f"Manifest: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    # ---- EDIT THESE PATHS ----
    seizure_only_json = "assets/eeg_seizure_only_eval.json"   # your seizure-only JSON
    cache_dir = "cache_windows_eval"                   # output folder

    # ---- CACHE SETTINGS ----
    manifest = build_cache_from_json(
        json_path=seizure_only_json,
        out_dir=cache_dir,
        fs=250,
        window_sec=10.0,
        stride_sec=5.0,
        C_max=41,
        max_records=None,          # set 50 for quick test
        max_windows_per_record=None, # set 100 for quick test
    )
