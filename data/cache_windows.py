from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import mne
from scipy.signal import butter, filtfilt


def read_label_intervals_from_csv(csv_path: str | Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []

    df = pd.read_csv(
        csv_path,
        comment="#",
        usecols=["start_time", "stop_time", "label"],
        dtype={"start_time": float, "stop_time": float, "label": str},
    )

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    intervals = []
    for s, e, lab in zip(df["start_time"], df["stop_time"], df["label"]):
        if pd.isna(s) or pd.isna(e) or pd.isna(lab):
            continue
        s, e = float(s), float(e)
        if e > s:
            intervals.append((s, e, lab))

    return intervals


def get_window_label(ws, we, intervals, default="bckg"):
    overlap_by_label = {}

    for s, e, lab in intervals:
        overlap = max(0.0, min(we, e) - max(ws, s))
        if overlap > 0:
            overlap_by_label[lab] = overlap_by_label.get(lab, 0.0) + overlap

    if not overlap_by_label:
        return default

    return max(overlap_by_label.items(), key=lambda x: x[1])[0]


def load_label_vocab(label_map_path: str | Path):
    print(type(label_map_path), label_map_path)
    label_map_path = Path(label_map_path)

    with label_map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label_to_id = {str(k): int(v) for k, v in data["label_to_id"].items()}
    id_to_label = {int(k): str(v) for k, v in data["id_to_label"].items()}

    return label_to_id, id_to_label


def build_label_vocab(records):
    labels = set()

    for rec in records:
        csv_path = Path(rec["csv_path"])
        if not csv_path.exists():
            continue

        df = pd.read_csv(
            csv_path,
            comment="#",
            usecols=["label"],
            dtype={"label": str},
        )

        labs = (
            df["label"].astype(str).str.strip().str.lower().dropna().unique().tolist()
        )
        labels.update(labs)

    labels = sorted(labels)

    if "bckg" in labels:
        labels.remove("bckg")
        labels = ["bckg"] + labels

    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    return label_to_id, id_to_label


def load_full_edf_all_channels(
    edf_path: str | Path,
    fs: int,
    c_max: int,
) -> Tuple[np.ndarray, int]:
    edf_path = str(edf_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick("eeg")

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")

    data = raw.get_data().astype(np.float32)

    # --- ADDED: 5th-order Butterworth bandpass filter (0.5 - 40 Hz) ---
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(5, [low, high], btype="band")
    data = filtfilt(b, a, data, axis=1).astype(np.float32)
    # ------------------------------------------------------------------

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
    label_to_id: Dict[str, int],
    fs: int = 250,
    window_sec: float = 1.0,  # changed from 10.0
    stride_sec: float = 1.0,  # changed from 5.0
    c_max: int = 21,  # changed from 41
    max_windows: Optional[int] = None,
) -> Optional[Tuple[Path, int]]:
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

    label_intervals = read_label_intervals_from_csv(csv_path)

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

        xw = x_full[:, st:en]
        if xw.shape[1] != win_T:
            continue

        label = get_window_label(ws, we, label_intervals, default="bckg")

        if label not in label_to_id:
            print(f"[SKIP] unknown label '{label}' in {csv_path}")
            continue

        y = label_to_id[label]

        xs.append(torch.from_numpy(xw))
        ys.append(y)

    if len(xs) == 0:
        print(f"[SKIP] no windows: {stem}")
        return None

    X = torch.stack(xs, dim=0)
    Y = torch.tensor(ys, dtype=torch.long)

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
                "label_to_id": label_to_id,
                "id_to_label": {v: k for k, v in label_to_id.items()},
            },
        },
        out_path,
    )

    return out_path, len(xs)


def build_cache_from_json(
    json_path: str | Path,
    out_dir: str | Path,
    fs: int = 250,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    c_max: int = 41,
    max_records: Optional[int] = None,
    max_windows_per_record: Optional[int] = None,
    label_map_path: Optional[str | Path] = None,
) -> Path:
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if max_records is not None:
        records = records[:max_records]

    if label_map_path is not None:

        label_to_id, id_to_label = load_label_vocab(label_map_path)
        print(f"Loaded label map from: {label_map_path}")

    else:
        label_to_id, id_to_label = build_label_vocab(records)
        print("Built label map from current records")

    print("Global label mapping:")
    for lab, idx in label_to_id.items():
        print(f"  {lab} -> {idx}")

    label_map_path = out_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"label_to_id": label_to_id, "id_to_label": id_to_label},
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest_path = out_dir / "manifest.jsonl"
    n_ok = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, rec in enumerate(records, start=1):
            result = cache_one_record_windows(
                rec=rec,
                out_dir=out_dir,
                label_to_id=label_to_id,
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

            mf.write(
                json.dumps({"pt_path": str(out_pt), "n": n}, ensure_ascii=False) + "\n"
            )
            print(f"[{i}/{len(records)}] cached: {out_pt.name} ({n} windows)")

    print(f"Done. Cached {n_ok} recordings.")
    print(f"Manifest: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    manifest = build_cache_from_json(
        json_path=r"eeg_seizure_only_eval.json",
        out_dir=r"cache_windows_eval_8_classes",
        label_map_path=r"label_map.json",
        fs=250,
        window_sec=1.0,
        stride_sec=1.0,
        c_max=21,  # to match the 2025 paper approach
        max_records=None,
        max_windows_per_record=None,
    )
    print(manifest)
