"""
MindScope XAI: DeepLIFT with Frequency Band Decomposition
==========================================================
Method  : DeepLIFT (Captum) applied independently to each clinical EEG
          frequency band (delta, theta, alpha, beta, gamma).

Why this method:
    Integrated Gradients and GradientSHAP both produce a single attribution
    score per electrode per time sample, blending all frequency content into
    one number. This script separates those contributions by frequency band,
    giving a three-axis explanation: channel x time x band. This is the
    approach proposed in XAI4EEG (Raab et al., 2023) and is not possible
    with IG or GradientSHAP alone.

    Baseline choice: instead of a zero (silent) baseline, this script uses
    the channel-wise mean of a random sample of background windows. This asks
    "compared to a typical non-seizure EEG, what pushed the model toward
    seizure?" which is the clinically meaningful question and reduces the
    Central-channel dominance artifact produced by zero baselines.

Reference:
    Raab, D., Theissler, A., Spiliopoulou, M. (2023).
    XAI4EEG: Spectral and spatio-temporal explanation of deep learning-based
    seizure detection in EEG time series.
    Neural Computing and Applications, 35, 10051-10068.
    https://doi.org/10.1007/s00521-022-07809-x

Data format (same as existing IG script):
    Each cached .pt file contains:
        data["x"]  shape [N, 17, 2500]   float32 EEG windows
        data["y"]  shape [N]              int labels (0=background, 1=seizure)

Model:
    Both EEGNet and EEGTCNet use EEGNetv4 from braindecode.
    Input : [batch, 17, 2500]
    Output: 2-class softmax (0=background, 1=seizure)

Output structure per model (example for EEGNet):
    xai_deeplift_EEGNet/
        channel_by_band.png
        region_by_band.png
        temporal_by_band.png
        heatmaps_by_band.png
        dominant_band_per_channel.png
        window_summary.csv
        summary.json
        EEGNet_delta_seizure_heatmap.npy  (one .npy per band per class)

Install requirements (everything else already in your environment):
    pip install captum scipy
"""

import os
import sys
import csv
import json
import random
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from captum.attr import DeepLift

sys.path.append(str(Path(__file__).resolve().parent))
from braindecode.models import EEGNetv4


DRIVE_DATA_DIR      = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\cache_windows_binary_10_sec_eval"
MANIFEST_PATH       = os.path.join(DRIVE_DATA_DIR, "manifest.jsonl")
EEGTCNET_CHECKPOINT = "weights/eegtcn_afterSampling.pt"
EEGNET_CHECKPOINT   = "weights/eegnet_afterLoader.pt"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

MAX_PER_CLASS = None

# Number of background windows averaged to form the DeepLIFT baseline.
# 200 is enough for a stable mean; raise to 500 if you want more precision
# at the cost of a slightly longer baseline-computation step.
BASELINE_N_SAMPLES = 200

RANDOM_SEED   = 42
N_CHANS       = 17
N_OUTPUTS     = 2
N_TIMES       = 2500
SAMPLING_RATE = 250

CHANNEL_NAMES = [
    "f7", "f3", "fz", "f4", "f8",
    "t3", "c3", "cz", "c4", "t4",
    "t5", "p3", "pz", "p4", "t6",
    "o1", "o2",
]

BRAIN_REGIONS = {
    "Frontal":   [0, 1, 2, 3, 4],
    "Temporal":  [5, 9, 10, 14],
    "Central":   [6, 7, 8],
    "Parietal":  [11, 12, 13],
    "Occipital": [15, 16],
}

TIME_SEGMENTS = {
    "Early 0-3s":    (0,    750),
    "Mid 3-6s":      (750,  1500),
    "Mid-Late 6-9s": (1500, 2250),
    "Late 9-10s":    (2250, 2500),
}

BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

BAND_COLORS = {
    "delta": "#4472C4",
    "theta": "#ED7D31",
    "alpha": "#70AD47",
    "beta":  "#C00000",
    "gamma": "#7030A0",
}


def load_model(checkpoint_path: str) -> nn.Module:
    model = EEGNetv4(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
    ).to(DEVICE)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')} "
              f"best metric {checkpoint.get('best_metric', '?')}")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint.to(DEVICE)

    model.eval()
    return model


def safe_load_pt(pt_path: str):
    """
    Copy the .pt file to a local temp path before loading.
    Prevents Errno 22 on long Google Drive shortcut paths.
    Returns None if the file cannot be loaded.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(pt_path, tmp_path)
        data = torch.load(tmp_path, map_location="cpu", weights_only=False)
        return data
    except Exception as e:
        print(f"  [SKIP] {Path(pt_path).name}: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def resolve_pt_path(manifest_pt_path: str) -> str:
    filename = Path(manifest_pt_path).name
    return os.path.join(DRIVE_DATA_DIR, filename)


def bandpass_filter(signal_np: np.ndarray, low_hz: float, high_hz: float,
                    fs: int = 250, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter applied to a [C, T] numpy array.
    Uses sosfiltfilt (forward-backward pass) to prevent phase distortion.
    """
    nyq  = fs / 2.0
    low  = max(low_hz  / nyq, 1e-4)
    high = min(high_hz / nyq, 1.0 - 1e-4)
    sos  = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal_np, axis=-1).astype(np.float32)


def compute_mean_background_baseline(selected: list,
                                     n_samples: int = BASELINE_N_SAMPLES
                                     ) -> torch.Tensor:
    """
    Compute a channel-wise mean baseline from a random sample of background
    windows. The result is a single [1, C, T] float32 tensor.

    Using the mean background signal as the DeepLIFT reference point means
    attributions answer "what in this window is different from a typical
    non-seizure EEG?" rather than "what is different from silence?". This
    produces more spatially diverse and clinically interpretable attributions
    and reduces the Central-channel dominance caused by zero baselines.

    Args:
        selected  : combined list of (pt_path, window_idx, label) tuples,
                    already built by sample_balanced.
        n_samples : how many background windows to average (default 200).

    Returns:
        torch.Tensor of shape [1, C, T]
    """
    rng = random.Random(RANDOM_SEED + 1)   # different seed from sampling

    bg_windows = [(p, i) for p, i, lbl in selected if lbl == 0]
    sampled    = rng.sample(bg_windows, min(n_samples, len(bg_windows)))

    # Sort by pt_path to minimise file loads (same trick as the main loop).
    sampled.sort(key=lambda x: x[0])

    accum      = np.zeros((N_CHANS, N_TIMES), dtype=np.float64)
    n_loaded   = 0
    loaded_pt  = None
    loaded_data = None

    for pt_path, w_idx in sampled:
        if pt_path != loaded_pt:
            loaded_data = safe_load_pt(pt_path)
            loaded_pt   = pt_path
        if loaded_data is None:
            continue
        accum    += loaded_data["x"][w_idx].numpy().astype(np.float64)
        n_loaded += 1

    if n_loaded == 0:
        print("  [WARNING] Could not load any background windows for baseline. "
              "Falling back to zero baseline.")
        return torch.zeros(1, N_CHANS, N_TIMES)

    mean_bg = (accum / n_loaded).astype(np.float32)
    baseline = torch.from_numpy(mean_bg).unsqueeze(0)   # [1, C, T]
    print(f"  Baseline: mean of {n_loaded} background windows  "
          f"(global mean={baseline.mean():.4f}  std={baseline.std():.4f})")
    return baseline


def compute_deeplift(model: nn.Module, x: torch.Tensor,
                     target_class: int,
                     baseline: torch.Tensor) -> np.ndarray:
    """
    Run DeepLIFT for one band-filtered window against the provided baseline.

    DeepLIFT (Shrikumar et al., 2017) assigns attribution scores by comparing
    each neuron's activation to a reference activation from a baseline input.
    Unlike IG, it does not integrate over interpolated steps; instead it
    propagates contribution scores layer-by-layer in a single forward-backward
    pass. This makes it fast enough to run independently per frequency band on
    every window in the evaluation set.

    Args:
        model        : EEGNetv4 in eval mode on DEVICE
        x            : [1, C, T] float32 tensor, already band-filtered
        target_class : 1 for seizure attribution, 0 for background attribution
        baseline     : [1, C, T] float32 tensor — mean background EEG signal.
                       Must be band-filtered to the same band as x before
                       this call (see run_deeplift_all_bands).

    Returns:
        numpy array [C, T] of raw DeepLIFT attributions
    """
    inp        = x.to(DEVICE).float()
    baseline_d = baseline.to(DEVICE).float()
    dl         = DeepLift(model)
    attrs      = dl.attribute(inp, baselines=baseline_d, target=target_class)
    return attrs.detach().cpu().squeeze(0).numpy()


def run_deeplift_all_bands(model: nn.Module, x_np: np.ndarray,
                           target_class: int,
                           baseline_np: np.ndarray) -> dict:
    """
    Bandpass-filter one window into all 5 bands, then run DeepLIFT on each.

    The baseline is also bandpass-filtered to the same band before being
    passed to DeepLIFT. This is important: if the input is delta-filtered
    but the baseline is broadband, the attribution contrast is meaningless.

    Args:
        x_np        : [C, T] float32 numpy array (raw window)
        target_class: 1 or 0
        baseline_np : [C, T] float32 numpy array (mean background window,
                      raw / broadband — will be filtered per band here)

    Returns:
        dict band_name -> numpy array [C, T]
    """
    result = {}
    for band_name, (lo, hi) in BANDS.items():
        filtered_x        = bandpass_filter(x_np,        lo, hi)
        filtered_baseline = bandpass_filter(baseline_np, lo, hi)
        inp               = torch.from_numpy(filtered_x).unsqueeze(0)
        base_t            = torch.from_numpy(filtered_baseline).unsqueeze(0)
        result[band_name] = compute_deeplift(model, inp, target_class, base_t)
    return result


def peak_brain_region(attr: np.ndarray) -> str:
    scores = {
        region: float(np.maximum(attr[idxs], 0.0).mean())
        for region, idxs in BRAIN_REGIONS.items()
    }
    return max(scores, key=scores.get)


def peak_time_segment(attr: np.ndarray) -> str:
    time_relevance = np.maximum(attr.mean(axis=0), 0.0)
    scores = {
        name: float(time_relevance[s:e].mean())
        for name, (s, e) in TIME_SEGMENTS.items()
    }
    return max(scores, key=scores.get)


def make_accumulators() -> dict:
    return {
        band: {
            "channel_sum":   np.zeros(N_CHANS, dtype=np.float64),
            "heatmap_sum":   np.zeros((N_CHANS, N_TIMES), dtype=np.float64),
            "region_counts": defaultdict(int),
            "time_counts":   defaultdict(int),
            "count":         0,
        }
        for band in BANDS
    }


def accumulate(acc: dict, band_attrs: dict, class_key: str):
    for band, attr in band_attrs.items():
        pos = np.maximum(attr, 0.0)
        acc[class_key][band]["channel_sum"] += pos.mean(axis=1)
        acc[class_key][band]["heatmap_sum"] += np.abs(attr)
        acc[class_key][band]["count"]       += 1
        acc[class_key][band]["region_counts"][peak_brain_region(attr)] += 1
        acc[class_key][band]["time_counts"][peak_time_segment(attr)]   += 1


def finalize(acc: dict, class_key: str) -> dict:
    result = {}
    for band in BANDS:
        cnt = max(acc[class_key][band]["count"], 1)

        region_pct = {
            r: acc[class_key][band]["region_counts"].get(r, 0) / cnt * 100
            for r in BRAIN_REGIONS
        }

        time_raw = {
            s: acc[class_key][band]["time_counts"].get(s, 0)
            for s in TIME_SEGMENTS
        }
        total_t  = max(sum(time_raw.values()), 1)
        time_pct = {s: v / total_t * 100 for s, v in time_raw.items()}

        result[band] = {
            "channel_mean": (acc[class_key][band]["channel_sum"] / cnt).tolist(),
            "region_pct":   region_pct,
            "temporal_pct": time_pct,
            "heatmap_mean": acc[class_key][band]["heatmap_sum"] / cnt,
            "n_windows":    cnt,
        }
    return result


def scan_manifest(entries: list) -> tuple:
    """
    Pass 1: read every .pt file in the manifest and build two index lists:
        seizure_index    list of (pt_path, window_idx) tuples for label==1
        background_index list of (pt_path, window_idx) tuples for label==0
    """
    seizure_index    = []
    background_index = []

    for entry in entries:
        pt_path = resolve_pt_path(entry["pt_path"])
        if not os.path.exists(pt_path):
            print(f"  [MISSING] {Path(pt_path).name}")
            continue

        data = safe_load_pt(pt_path)
        if data is None:
            continue

        y_all = data["y"]
        for w_idx in range(len(y_all)):
            label = int(y_all[w_idx].item())
            if label == 1:
                seizure_index.append((pt_path, w_idx))
            else:
                background_index.append((pt_path, w_idx))

    return seizure_index, background_index


def sample_balanced(seizure_index: list, background_index: list) -> list:
    """
    If MAX_PER_CLASS is set, randomly sample that many windows from each class.
    Otherwise use all windows.
    Returns a combined list of (pt_path, window_idx, label) tuples sorted by
    pt_path so that all windows from the same file are processed together,
    avoiding redundant file loads.
    """
    rng = random.Random(RANDOM_SEED)

    if MAX_PER_CLASS is not None:
        sz = rng.sample(seizure_index,    min(MAX_PER_CLASS, len(seizure_index)))
        bg = rng.sample(background_index, min(MAX_PER_CLASS, len(background_index)))
    else:
        sz = list(seizure_index)
        bg = list(background_index)

    combined = [(p, i, 1) for p, i in sz] + [(p, i, 0) for p, i in bg]
    combined.sort(key=lambda x: x[0])
    return combined


def run_xai(model: nn.Module, model_name: str):
    out_dir = f"xai_deeplift_{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running DeepLIFT band analysis: {model_name}")
    print(f"Output directory: {out_dir}")

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    entries = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Manifest entries: {len(entries)}")

    print("Scanning all .pt files to build class index (pass 1 of 2)")
    seizure_index, background_index = scan_manifest(entries)
    print(f"  Found {len(seizure_index)} seizure windows and "
          f"{len(background_index)} background windows across all files")

    selected = sample_balanced(seizure_index, background_index)
    n_sei_selected = sum(1 for _, _, lbl in selected if lbl == 1)
    n_bg_selected  = sum(1 for _, _, lbl in selected if lbl == 0)
    print(f"  Selected {n_sei_selected} seizure + {n_bg_selected} background "
          f"= {len(selected)} total windows for attribution (pass 2 of 2)")

    # ------------------------------------------------------------------
    # Compute mean background baseline ONCE before the attribution loop.
    # This tensor is band-filtered per band inside run_deeplift_all_bands,
    # so we store it as broadband numpy here.
    # ------------------------------------------------------------------
    print("Computing mean background baseline...")
    baseline_tensor = compute_mean_background_baseline(selected)
    baseline_np     = baseline_tensor.squeeze(0).numpy()   # [C, T]  broadband

    csv_path = os.path.join(out_dir, "window_summary.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "file", "window_idx", "true_label", "pred_label", "prob_seizure",
        "delta_peak_region", "theta_peak_region", "alpha_peak_region",
        "beta_peak_region",  "gamma_peak_region",
        "delta_peak_time",   "theta_peak_time",   "alpha_peak_time",
        "beta_peak_time",    "gamma_peak_time",
    ])

    acc = {
        "seizure":    make_accumulators(),
        "background": make_accumulators(),
    }

    total        = 0
    correct      = 0
    n_seizure    = 0
    n_background = 0

    # Group by pt_path so each file is loaded exactly once.
    for pt_path, group in groupby(selected, key=lambda x: x[0]):
        windows = list(group)

        data = safe_load_pt(pt_path)
        if data is None:
            continue
        print(f"  Loaded {Path(pt_path).name}  ({len(windows)} windows)")

        for _, w_idx, true_label in windows:
            x_tensor  = data["x"][w_idx].unsqueeze(0).float()
            x_np      = data["x"][w_idx].numpy().astype(np.float32)
            class_key = "seizure" if true_label == 1 else "background"

            with torch.no_grad():
                logits       = model(x_tensor.to(DEVICE))
                probs        = torch.softmax(logits, dim=1)
                prob_seizure = probs[0, 1].item()
            pred = 1 if prob_seizure >= 0.5 else 0

            total   += 1
            correct += int(pred == true_label)
            if true_label == 1:
                n_seizure    += 1
            else:
                n_background += 1

            # Pass baseline_np so each band gets a matched filtered baseline.
            band_attrs = run_deeplift_all_bands(
                model, x_np, target_class=true_label, baseline_np=baseline_np
            )
            accumulate(acc, band_attrs, class_key)

            writer.writerow([
                Path(pt_path).name,
                w_idx,
                true_label,
                pred,
                round(prob_seizure, 4),
                peak_brain_region(band_attrs["delta"]),
                peak_brain_region(band_attrs["theta"]),
                peak_brain_region(band_attrs["alpha"]),
                peak_brain_region(band_attrs["beta"]),
                peak_brain_region(band_attrs["gamma"]),
                peak_time_segment(band_attrs["delta"]),
                peak_time_segment(band_attrs["theta"]),
                peak_time_segment(band_attrs["alpha"]),
                peak_time_segment(band_attrs["beta"]),
                peak_time_segment(band_attrs["gamma"]),
            ])

    csv_file.close()

    print(f"Processed {total} windows ({n_seizure} seizure, {n_background} background)")
    print(f"Accuracy  {correct}/{total} = {100 * correct / max(total, 1):.1f}%")

    sz_results = finalize(acc, "seizure")
    bg_results = finalize(acc, "background")

    plot_channel_by_band(sz_results, bg_results, model_name, out_dir, n_seizure, n_background)
    plot_region_by_band(sz_results,  bg_results, model_name, out_dir, n_seizure, n_background)
    plot_temporal_by_band(sz_results, bg_results, model_name, out_dir, n_seizure, n_background)
    plot_heatmaps_by_band(sz_results, bg_results, model_name, out_dir)
    plot_dominant_band(sz_results, model_name, out_dir)
    save_summary(sz_results, bg_results, model_name, out_dir, total, correct, n_seizure, n_background)
    save_heatmap_arrays(sz_results, bg_results, model_name, out_dir)

    print(f"All outputs saved to {out_dir}/")


def plot_channel_by_band(sz, bg, model_name, out_dir, n_sei, n_bg):
    bands  = list(BANDS.keys())
    y_pos  = np.arange(N_CHANS)
    fig, axes = plt.subplots(1, 5, figsize=(22, 8), sharey=True)
    fig.suptitle(
        f"{model_name}   DeepLIFT Channel Attribution by Frequency Band\n"
        "(baseline: mean background EEG)",
        fontsize=12
    )
    for col, band in enumerate(bands):
        ax      = axes[col]
        sz_vals = np.array(sz[band]["channel_mean"])
        bg_vals = np.array(bg[band]["channel_mean"])
        ax.barh(y_pos + 0.2, sz_vals, height=0.4, color="#C00000",
                label=f"Seizure (n={n_sei})")
        ax.barh(y_pos - 0.2, bg_vals, height=0.4, color="#2E75B6",
                label=f"Background (n={n_bg})")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)
        ax.set_title(f"{band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)", fontsize=9)
        ax.set_xlabel("Mean positive attribution", fontsize=7)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "channel_by_band.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_region_by_band(sz, bg, model_name, out_dir, n_sei, n_bg):
    bands   = list(BANDS.keys())
    regions = list(BRAIN_REGIONS.keys())
    x_pos   = np.arange(len(regions))
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle(
        f"{model_name}   DeepLIFT Brain Region Attribution by Frequency Band\n"
        "Y-axis: percentage of windows where this region had peak attribution  "
        "(baseline: mean background EEG)",
        fontsize=11
    )
    for col, band in enumerate(bands):
        ax      = axes[col]
        sz_vals = [sz[band]["region_pct"].get(r, 0) for r in regions]
        bg_vals = [bg[band]["region_pct"].get(r, 0) for r in regions]
        ax.bar(x_pos - 0.2, sz_vals, width=0.4, color="#C00000",
               label=f"Seizure (n={n_sei})")
        ax.bar(x_pos + 0.2, bg_vals, width=0.4, color="#2E75B6",
               label=f"Background (n={n_bg})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([r[:4] for r in regions], fontsize=7, rotation=30)
        ax.set_title(f"{band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)", fontsize=9)
        ax.set_ylabel("% of windows", fontsize=7)
        if col == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "region_by_band.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_temporal_by_band(sz, bg, model_name, out_dir, n_sei, n_bg):
    bands      = list(BANDS.keys())
    segments   = list(TIME_SEGMENTS.keys())
    seg_labels = ["0-3s", "3-6s", "6-9s", "9-10s"]
    x_pos      = np.arange(len(segments))
    fig, axes  = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle(
        f"{model_name}   DeepLIFT Temporal Attribution by Frequency Band\n"
        "Y-axis: percentage of total attribution falling in each time segment  "
        "(baseline: mean background EEG)",
        fontsize=11
    )
    for col, band in enumerate(bands):
        ax      = axes[col]
        sz_vals = [sz[band]["temporal_pct"].get(s, 0) for s in segments]
        bg_vals = [bg[band]["temporal_pct"].get(s, 0) for s in segments]
        ax.plot(x_pos, sz_vals, "o-",  color="#C00000",
                label=f"Seizure (n={n_sei})",   linewidth=1.5)
        ax.plot(x_pos, bg_vals, "s--", color="#2E75B6",
                label=f"Background (n={n_bg})", linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(seg_labels, fontsize=7)
        ax.set_title(f"{band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)", fontsize=9)
        ax.set_ylabel("% of total attribution", fontsize=7)
        if col == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "temporal_by_band.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_heatmaps_by_band(sz, bg, model_name, out_dir):
    """
    10-panel grid: 5 bands (columns) x 2 classes (rows).
    Each panel shows mean absolute DeepLIFT attribution as a heatmap.
    Channels on the Y-axis, time in seconds on the X-axis.
    """
    bands     = list(BANDS.keys())
    fig, axes = plt.subplots(2, 5, figsize=(26, 9))
    fig.suptitle(
        f"{model_name}   DeepLIFT Mean Attribution Heatmaps by Frequency Band\n"
        "Y-axis: EEG channels   X-axis: time (0 to 10 seconds)  "
        "(baseline: mean background EEG)",
        fontsize=11
    )
    for col, band in enumerate(bands):
        for row, (results, label) in enumerate([(sz, "Seizure"), (bg, "Background")]):
            ax   = axes[row][col]
            hmap = results[band]["heatmap_mean"]
            vmax = hmap.max() if hmap.max() > 0 else 1.0
            im   = ax.imshow(
                hmap, aspect="auto", cmap="Reds",
                origin="upper", vmin=0, vmax=vmax,
                extent=[0, 10, N_CHANS, 0],
            )
            ax.set_yticks(np.arange(N_CHANS) + 0.5)
            ax.set_yticklabels(CHANNEL_NAMES, fontsize=5)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_title(
                f"{label}   {band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)",
                fontsize=8
            )
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "heatmaps_by_band.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_dominant_band(sz, model_name, out_dir):
    """
    For each channel, identifies which frequency band has the highest mean
    positive attribution in seizure windows and colors the bar accordingly.
    """
    bands  = list(BANDS.keys())
    y_pos  = np.arange(N_CHANS)

    dominant_band  = []
    dominant_value = []
    for ch in range(N_CHANS):
        scores    = {band: sz[band]["channel_mean"][ch] for band in bands}
        best_band = max(scores, key=scores.get)
        dominant_band.append(best_band)
        dominant_value.append(scores[best_band])

    colors = [BAND_COLORS[b] for b in dominant_band]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(y_pos, dominant_value, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=9)
    ax.set_xlabel("Peak band attribution score", fontsize=9)
    ax.set_title(
        f"{model_name}   Dominant Frequency Band per Channel (Seizure windows)\n"
        "(baseline: mean background EEG)",
        fontsize=10
    )
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1,
                       color=BAND_COLORS[b],
                       label=f"{b}  ({BANDS[b][0]}-{BANDS[b][1]} Hz)")
        for b in bands
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "dominant_band_per_channel.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def save_summary(sz, bg, model_name, out_dir, total, correct, n_sei, n_bg):
    summary = {
        "model":         model_name,
        "total_windows": total,
        "accuracy":      round(100 * correct / max(total, 1), 2),
        "n_seizure":     n_sei,
        "n_background":  n_bg,
        "baseline":      f"mean of {BASELINE_N_SAMPLES} background windows",
        "bands":         {},
    }
    for band in BANDS:
        sz_top_ch = CHANNEL_NAMES[int(np.argmax(sz[band]["channel_mean"]))]
        bg_top_ch = CHANNEL_NAMES[int(np.argmax(bg[band]["channel_mean"]))]
        sz_top_rg = max(sz[band]["region_pct"], key=sz[band]["region_pct"].get)
        bg_top_rg = max(bg[band]["region_pct"], key=bg[band]["region_pct"].get)
        sz_top_tm = max(sz[band]["temporal_pct"], key=sz[band]["temporal_pct"].get)
        bg_top_tm = max(bg[band]["temporal_pct"], key=bg[band]["temporal_pct"].get)

        summary["bands"][band] = {
            "seizure": {
                "top_channel":      sz_top_ch,
                "top_region":       sz_top_rg,
                "top_region_pct":   round(sz[band]["region_pct"][sz_top_rg], 2),
                "top_time_segment": sz_top_tm,
                "channel_mean":     [round(v, 8) for v in sz[band]["channel_mean"]],
                "region_pct":       {k: round(v, 2) for k, v in sz[band]["region_pct"].items()},
                "temporal_pct":     {k: round(v, 2) for k, v in sz[band]["temporal_pct"].items()},
            },
            "background": {
                "top_channel":      bg_top_ch,
                "top_region":       bg_top_rg,
                "top_region_pct":   round(bg[band]["region_pct"][bg_top_rg], 2),
                "top_time_segment": bg_top_tm,
                "channel_mean":     [round(v, 8) for v in bg[band]["channel_mean"]],
                "region_pct":       {k: round(v, 2) for k, v in bg[band]["region_pct"].items()},
                "temporal_pct":     {k: round(v, 2) for k, v in bg[band]["temporal_pct"].items()},
            },
        }

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {path}")

    print(f"\nSummary for {model_name}")
    print(f"  Accuracy {summary['accuracy']}%  ({n_sei} seizure, {n_bg} background)")
    for band in BANDS:
        b = summary["bands"][band]
        print(f"  {band} ({BANDS[band][0]}-{BANDS[band][1]} Hz)")
        print(f"    Seizure    top channel {b['seizure']['top_channel']}"
              f"   top region {b['seizure']['top_region']}"
              f" ({b['seizure']['top_region_pct']:.1f}%)"
              f"   top time {b['seizure']['top_time_segment']}")
        print(f"    Background top channel {b['background']['top_channel']}"
              f"   top region {b['background']['top_region']}"
              f" ({b['background']['top_region_pct']:.1f}%)"
              f"   top time {b['background']['top_time_segment']}")


def save_heatmap_arrays(sz, bg, model_name, out_dir):
    for band in BANDS:
        np.save(
            os.path.join(out_dir, f"{model_name}_{band}_seizure_heatmap.npy"),
            sz[band]["heatmap_mean"],
        )
        np.save(
            os.path.join(out_dir, f"{model_name}_{band}_background_heatmap.npy"),
            bg[band]["heatmap_mean"],
        )


if __name__ == "__main__":
    print(f"Device         : {DEVICE}")
    print(f"Max per class  : {MAX_PER_CLASS if MAX_PER_CLASS else 'all'}")
    print(f"Baseline samples: {BASELINE_N_SAMPLES} background windows")

    print("\nLoading EEGTCNet")
    model_tcnet = load_model(EEGTCNET_CHECKPOINT)
    run_xai(model_tcnet, "EEGTCNet")
    del model_tcnet
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nLoading EEGNet")
    model_eegnet = load_model(EEGNET_CHECKPOINT)
    run_xai(model_eegnet, "EEGNet")
    del model_eegnet
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\nDone.")
    print("Results in:")
    print("  xai_deeplift_EEGTCNet/")
    print("  xai_deeplift_EEGNet/")