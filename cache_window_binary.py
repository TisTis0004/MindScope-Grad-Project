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
# CONFIG
# =========================================================

CANONICAL_21 = [
    "fp1", "fp2",
    "f7", "f3", "fz", "f4", "f8",
    "t3", "c3", "cz", "c4", "t4",
    "t5", "p3", "pz", "p4", "t6",
    "o1", "o2",
    "a1", "a2",
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
    h_freq: Optional[float] = 40.0

    # binary labels
    background_labels: Tuple[str, ...] = ("bckg", "background")
    seizure_label_name: str = "seizure"
    non_seizure_label_name: str = "non_seizure"

    # montage
    target_montage_channels: Tuple[str, ...] = tuple(CANONICAL_21)
    montage_mode: str = "pad_missing"      # "pad_missing" | "subset" | "strict"
    missing_channel_fill: str = "zero"     # "zero" | "mean"

    # quality
    flat_std_thresh: float = 1e-3
    max_flat_ratio: float = 0.3
    max_zero_ratio: float = 0.3

    # normalization
    per_channel_normalize: bool = True


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
        raise ValueError(
            f"CSV {csv_path} must contain columns: start_time, stop_time, label"
        )

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["stop_time"] = pd.to_numeric(df["stop_time"], errors="coerce")

    intervals = []
    for _, row in df.iterrows():
        s = row["start_time"]
        e = row["stop_time"]
        lab = row["label"]

        if pd.isna(s) or pd.isna(e) or pd.isna(lab):
            continue

        s = float(s)
        e = float(e)
        if e <= s:
            continue

        item = {
            "start_time": s,
            "stop_time": e,
            "label": lab,
        }

        if "channel" in df.columns:
            ch = str(row["channel"]).strip()
            item["channel"] = ch

        intervals.append(item)

    return intervals


def assign_raw_label_by_overlap(
    ws: float,
    we: float,
    intervals: List[Dict[str, Any]],
    default: str = "bckg",
) -> str:
    overlap_by_label: Dict[str, float] = {}

    for item in intervals:
        s = item["start_time"]
        e = item["stop_time"]
        lab = item["label"]

        overlap = max(0.0, min(we, e) - max(ws, s))
        if overlap > 0:
            overlap_by_label[lab] = overlap_by_label.get(lab, 0.0) + overlap

    if not overlap_by_label:
        return default

    return max(overlap_by_label.items(), key=lambda x: x[1])[0]


# =========================================================
# CHANNEL / MONTAGE
# =========================================================

def normalize_channel_name(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("eeg ", "")
    name = name.replace("-ref", "")
    name = name.replace("-le", "")
    name = name.replace(" ", "")
    name = name.replace("t7", "t3")
    name = name.replace("t8", "t4")
    name = name.replace("p7", "t5")
    name = name.replace("p8", "t6")
    name = re.sub(r"[^a-z0-9]", "", name)
    return name


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


def map_to_canonical_montage(
    data: np.ndarray,
    original_channels: List[str],
    target_channels: List[str],
    montage_mode: str = "pad_missing",
    missing_channel_fill: str = "zero",
):
    norm_original = [normalize_channel_name(ch) for ch in original_channels]
    norm_target = [normalize_channel_name(ch) for ch in target_channels]

    idx_map = {ch: i for i, ch in enumerate(norm_original)}

    matched_indices = []
    missing_channels = []
    final_channels = []

    for raw_tgt, norm_tgt in zip(target_channels, norm_target):
        if norm_tgt in idx_map:
            matched_indices.append(idx_map[norm_tgt])
            final_channels.append(raw_tgt)
        else:
            matched_indices.append(None)
            final_channels.append(raw_tgt)
            missing_channels.append(raw_tgt)

    ignored_channels = [
        ch for ch, nch in zip(original_channels, norm_original)
        if nch not in set(norm_target)
    ]

    if montage_mode == "strict" and missing_channels:
        raise ValueError(f"Missing target channels: {missing_channels}")

    if montage_mode == "subset":
        present = [(ch, idx) for ch, idx in zip(final_channels, matched_indices) if idx is not None]
        if not present:
            raise ValueError("No target montage channels found in EDF")
        used_channels, used_idx = zip(*present)
        x = data[list(used_idx)]
        meta = {
            "original_channels": original_channels,
            "final_montage_channels": list(used_channels),
            "missing_channels": missing_channels,
            "ignored_channels": ignored_channels,
        }
        return x, meta

    # pad_missing or strict with full presence
    T = data.shape[1]
    x = np.zeros((len(target_channels), T), dtype=np.float32)

    if missing_channel_fill == "mean":
        fill_signal = data.mean(axis=0).astype(np.float32)
    else:
        fill_signal = np.zeros((T,), dtype=np.float32)

    for i, src_idx in enumerate(matched_indices):
        if src_idx is None:
            x[i] = fill_signal
        else:
            x[i] = data[src_idx]

    meta = {
        "original_channels": original_channels,
        "final_montage_channels": list(target_channels),
        "missing_channels": missing_channels,
        "ignored_channels": ignored_channels,
    }
    return x, meta


# =========================================================
# PREPROCESS / QUALITY
# =========================================================

def normalize_per_channel(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def is_bad_window(
    xw: np.ndarray,
    flat_std_thresh: float = 1e-3,
    max_flat_ratio: float = 0.3,
    max_zero_ratio: float = 0.3,
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

    if not edf_path.exists():
        print(f"[SKIP] missing EDF: {edf_path}")
        return None

    if not csv_path.exists():
        print(f"[SKIP] missing CSV: {csv_path}")
        return None

    intervals = read_label_intervals_from_csv(csv_path)

    raw_data, original_channels = load_edf_signals(
        edf_path=edf_path,
        fs=cfg.fs,
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
    )

    x_full, montage_meta = map_to_canonical_montage(
        data=raw_data,
        original_channels=original_channels,
        target_channels=list(cfg.target_montage_channels),
        montage_mode=cfg.montage_mode,
        missing_channel_fill=cfg.missing_channel_fill,
    )

    if cfg.per_channel_normalize:
        x_full = normalize_per_channel(x_full)

    T_full = x_full.shape[1]

    win_T = int(cfg.fs * cfg.window_sec)
    stride_T = int(cfg.fs * cfg.stride_sec)

    if win_T <= 0 or stride_T <= 0:
        raise ValueError("window_sec and stride_sec must be > 0")

    starts = list(range(0, max(0, T_full - win_T) + 1, stride_T))
    if cfg.max_windows_per_record is not None:
        starts = starts[:cfg.max_windows_per_record]

    xs: List[torch.Tensor] = []
    ys: List[int] = []

    raw_label_counts = Counter()
    final_label_counts = Counter()
    bad_windows = 0
    skipped_unknown = 0

    for st in starts:
        en = st + win_T
        if en > T_full:
            continue

        ws = st / cfg.fs
        we = en / cfg.fs

        xw = x_full[:, st:en]
        if xw.shape[1] != win_T:
            continue

        if is_bad_window(
            xw,
            flat_std_thresh=cfg.flat_std_thresh,
            max_flat_ratio=cfg.max_flat_ratio,
            max_zero_ratio=cfg.max_zero_ratio,
        ):
            bad_windows += 1
            continue

        raw_label = assign_raw_label_by_overlap(ws, we, intervals, default="bckg")
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

    torch.save(
        {
            "x": X,   # [N, C, T]
            "y": Y,   # [N]
            "meta": {
                "stem": stem,
                "edf_path": str(edf_path),
                "csv_path": str(csv_path),
                "subject": rec.get("subject"),
                "session": rec.get("session"),
                "fs": cfg.fs,
                "window_sec": cfg.window_sec,
                "stride_sec": cfg.stride_sec,
                "label_mode": "binary",
                "label_to_id": label_to_id,
                "id_to_label": {v: k for k, v in label_to_id.items()},
                "background_labels": list(cfg.background_labels),
                "target_montage_channels": list(cfg.target_montage_channels),
                "montage_mode": cfg.montage_mode,
                "missing_channel_fill": cfg.missing_channel_fill,
                "bandpass": [cfg.l_freq, cfg.h_freq],
                "bad_windows_skipped": bad_windows,
                "unknown_labels_skipped": skipped_unknown,
                "raw_label_counts": dict(raw_label_counts),
                "final_label_counts": dict(final_label_counts),
                **montage_meta,
            },
        },
        out_path,
    )

    report = {
        "stem": stem,
        "pt_path": str(out_path),
        "n_windows": len(xs),
        "bad_windows_skipped": bad_windows,
        "unknown_labels_skipped": skipped_unknown,
        "raw_label_counts": dict(raw_label_counts),
        "final_label_counts": dict(final_label_counts),
        "missing_channels": montage_meta["missing_channels"],
        "ignored_channels": montage_meta["ignored_channels"],
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

    label_to_id, id_to_label = build_binary_label_vocab(
        seizure_label_name=cfg.seizure_label_name,
        non_seizure_label_name=cfg.non_seizure_label_name,
    )

    # save label map
    label_map_path = out_dir / "label_map.json"
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"label_to_id": label_to_id, "id_to_label": id_to_label},
            f,
            indent=2,
            ensure_ascii=False,
        )

    manifest_path = out_dir / "manifest.jsonl"
    quality_report_path = out_dir / "quality_report.json"

    n_cached_records = 0
    total_windows = 0
    all_reports = []
    total_final_label_counts = Counter()

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, rec in enumerate(records, start=1):
            result = cache_one_record_windows(
                rec=rec,
                out_dir=out_dir,
                cfg=cfg,
                label_to_id=label_to_id,
            )

            if result is None:
                continue

            out_pt, n, report = result
            n_cached_records += 1
            total_windows += n
            all_reports.append(report)
            total_final_label_counts.update(report["final_label_counts"])

            mf.write(
                json.dumps({"pt_path": str(out_pt), "n": n}, ensure_ascii=False) + "\n"
            )

            print(f"[{i}/{len(records)}] cached: {out_pt.name} ({n} windows)")

    quality_report = {
        "n_input_records": len(records),
        "n_cached_records": n_cached_records,
        "total_windows": total_windows,
        "total_final_label_counts": dict(total_final_label_counts),
        "config": asdict(cfg),
        "records": all_reports,
    }

    with quality_report_path.open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Cached records: {n_cached_records}")
    print(f"Total windows: {total_windows}")
    print(f"Manifest: {manifest_path}")
    print(f"Label map: {label_map_path}")
    print(f"Quality report: {quality_report_path}")

    return manifest_path


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    cfg = CacheConfig(
        json_path=r"assets\eeg_seizure_only.json",
        out_dir=r"cache_windows_binary",

        fs=250,
        window_sec=5.0,
        stride_sec=1.0,

        max_records=None,
        max_windows_per_record=None,

        l_freq=0.5,
        h_freq=40.0,

        montage_mode="pad_missing",     # "pad_missing" is safest
        missing_channel_fill="zero",    # "zero" or "mean"
    )

    build_cache_from_json(cfg)