"""
MindScope XAI on the Ensemble Model: Integrated Gradients (per-band)
Runs Integrated Gradients (Captum, 100 steps) on the heterogeneous ensemble:
    Branch A : 1D EEGNet  operating on raw bipolar EEG
    Branch B : 2D CNN-LSTM operating on mel-spectrograms (via EEGToSpectrogram)
Both branches see the SAME raw EEG window; spectrogram conversion is internal
to the wrapper so Captum can compute gradients end-to-end w.r.t. raw EEG.

Input: 18-channel Double Banana bipolar montage, 10 s @ 256 Hz = 2560 samples.

Ensemble combination:
    p_ensemble = w * p_2d + (1 - w) * p_1d

Attribution target:
    Always class 1 (SEIZURE), regardless of true label or prediction.
    For seizure windows: attribution shows "what looks like seizure here".
    For background windows: attribution shows "what would have been seizure-
    like here, if anything". The contrast between the two reveals the
    discriminative features the model uses for seizure detection.

Per-band decomposition (XAI4EEG; Raab et al., 2023):
    For each window and each clinical band (delta/theta/alpha/beta/gamma):
      - Bandpass-filter the input EEG into that band.
      - Bandpass-filter the baseline into the same band.
      - Run IG on the filtered input against the filtered baseline.

DeepLIFT was removed because it disagreed with IG at Pearson r ~0.3 due to
known limitations with nn.LSTM, InstanceNorm2d, and softmax attention used
in the CNN-LSTM branch. IG is theoretically exact on those layers.

Output:
    xai_ig_Ensemble/

References:
    Sundararajan et al. (2017). Integrated Gradients. ICML.
    Raab, Theissler, Spiliopoulou (2023). XAI4EEG. Neural Computing and Applications.
    Ravindran & Contreras-Vidal (2023). Empirical comparison of XAI for EEG.
      Scientific Reports.
"""

import os
import sys
import csv
import json
import random
import shutil
import tempfile
import time
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

from captum.attr import IntegratedGradients

from braindecode.models import EEGNet
sys.path.append(str(Path(__file__).resolve().parent))
from helper.T import EEGToSpectrogram
from helper.models import Spectrogram_CNN_LSTM


DRIVE_DATA_DIR         = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\V2 of 10 sec\cache_windows_binary_10_sec_eval"
MANIFEST_PATH          = os.path.join(DRIVE_DATA_DIR, "manifest.jsonl")
SPECTROGRAM_CHECKPOINT = r"weights\cnn_lstm_melspectrogram_dropout_new4changes.pt"
EEGNET_CHECKPOINT      = r"weights\eegnet_1d_best.pt"

ENSEMBLE_WEIGHT_2D = 0.5

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PER_CLASS = None
RANDOM_SEED   = 42

ATTRIBUTION_TARGET_CLASS = 1

N_CHANS       = 18
N_OUTPUTS     = 2
N_TIMES       = 2560
SAMPLING_RATE = 256

BASELINE_N_SAMPLES = 200
IG_N_STEPS         = 100
INPUT_CLAMP        = 20.0

EEGNET_CFG = dict(
    n_chans=N_CHANS,
    n_outputs=N_OUTPUTS,
    n_times=N_TIMES,
    final_conv_length="auto",
    pool_mode="mean",
    F1=16, D=2, F2=32,
    kernel_length=128,
    drop_prob=0.5,
)

CHANNEL_NAMES = [
    "fp1-f7", "f7-t3",  "t3-t5",  "t5-o1",
    "fp2-f8", "f8-t4",  "t4-t6",  "t6-o2",
    "fp1-f3", "f3-c3",  "c3-p3",  "p3-o1",
    "fp2-f4", "f4-c4",  "c4-p4",  "p4-o2",
    "fz-cz",  "cz-pz",
]

BRAIN_REGIONS = {
    "Frontal":   [0, 4, 8, 12],
    "Temporal":  [1, 2, 5, 6],
    "Central":   [9, 13, 16],
    "Parietal":  [10, 14, 17],
    "Occipital": [3, 7, 11, 15],
}

TIME_SEGMENTS = {
    "Early 0-3s":    (0,    768),
    "Mid 3-6s":      (768,  1536),
    "Mid-Late 6-9s": (1536, 2304),
    "Late 9-10s":    (2304, 2560),
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


class EnsembleWrapper(nn.Module):
    def __init__(self, eegnet, cnn_lstm, spec_transform, weight_2d=0.5):
        super().__init__()
        self.eegnet         = eegnet
        self.cnn_lstm       = cnn_lstm
        self.spec_transform = spec_transform
        self.weight_2d      = float(weight_2d)
        self.weight_1d      = 1.0 - float(weight_2d)

    def forward(self, x):
        x_clamped = torch.clamp(x, min=-INPUT_CLAMP, max=INPUT_CLAMP)

        logits_1d = self.eegnet(x_clamped)
        probs_1d  = torch.softmax(logits_1d, dim=1)

        spec      = self.spec_transform(x_clamped)
        logits_2d = self.cnn_lstm(spec)
        probs_2d  = torch.softmax(logits_2d, dim=1)

        return self.weight_2d * probs_2d + self.weight_1d * probs_1d


def _clean_swa_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[new_k] = v
    return cleaned


def load_ensemble(weight_2d=ENSEMBLE_WEIGHT_2D):
    print(f"[device] {DEVICE} | weight_2d={weight_2d} | max_per_class={MAX_PER_CLASS}", flush=True)
    print(f"[config] attribution target = class {ATTRIBUTION_TARGET_CLASS} (seizure)", flush=True)

    print("[load] EEGNet-1D...", flush=True)
    eegnet = EEGNet(**EEGNET_CFG).to(DEVICE)
    ckpt_1d = torch.load(EEGNET_CHECKPOINT, map_location=DEVICE, weights_only=False)
    sd_1d   = ckpt_1d.get("model_state_dict", ckpt_1d) if isinstance(ckpt_1d, dict) else ckpt_1d
    if isinstance(ckpt_1d, dict) and ckpt_1d.get("swa", False):
        sd_1d = _clean_swa_state_dict(sd_1d)
    eegnet.load_state_dict(sd_1d)
    eegnet.eval()
    print("[load] EEGNet-1D OK", flush=True)

    print("[load] CNN-LSTM-2D...", flush=True)
    cnn_lstm = Spectrogram_CNN_LSTM().to(DEVICE)
    ckpt_2d  = torch.load(SPECTROGRAM_CHECKPOINT, map_location=DEVICE, weights_only=False)
    sd_2d    = ckpt_2d.get("model_state_dict", ckpt_2d) if isinstance(ckpt_2d, dict) else ckpt_2d
    cnn_lstm.load_state_dict(sd_2d)
    cnn_lstm.eval()
    print("[load] CNN-LSTM-2D OK", flush=True)

    print("[load] spectrogram transform...", flush=True)
    spec = EEGToSpectrogram().to(DEVICE)
    spec.eval()

    print("[load] building ensemble wrapper...", flush=True)
    wrapper = EnsembleWrapper(eegnet, cnn_lstm, spec, weight_2d=weight_2d).to(DEVICE)
    wrapper.eval()
    print("[load] Ensemble ready", flush=True)
    return wrapper


def safe_load_pt(pt_path):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(pt_path, tmp_path)
        return torch.load(tmp_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def resolve_pt_path(manifest_pt_path):
    return os.path.join(DRIVE_DATA_DIR, Path(manifest_pt_path).name)


def scan_manifest(entries):
    seizure_index    = []
    background_index = []
    first_shape_checked = False
    n_total = len(entries)
    t0 = time.time()

    for idx, entry in enumerate(entries):
        pt_path = resolve_pt_path(entry["pt_path"])
        if not os.path.exists(pt_path):
            continue
        data = safe_load_pt(pt_path)
        if data is None:
            continue

        if not first_shape_checked:
            x_shape = tuple(data["x"].shape)
            if x_shape[1] != N_CHANS or x_shape[2] != N_TIMES:
                raise RuntimeError(
                    f"Shape mismatch. Cached data is {x_shape}, "
                    f"expected [N, {N_CHANS}, {N_TIMES}]."
                )
            first_shape_checked = True

        y_all = data["y"]
        for w_idx in range(len(y_all)):
            label = int(y_all[w_idx].item())
            if label == 1:
                seizure_index.append((pt_path, w_idx))
            else:
                background_index.append((pt_path, w_idx))

        if (idx + 1) % 25 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            print(f"[scan] {idx+1}/{n_total} files read ({elapsed:.0f}s elapsed)", flush=True)

    return seizure_index, background_index


def sample_balanced(seizure_index, background_index):
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


def compute_mean_background_baseline(selected, n_samples=BASELINE_N_SAMPLES):
    rng = random.Random(RANDOM_SEED + 1)
    bg_windows = [(p, i) for p, i, lbl in selected if lbl == 0]
    sampled    = rng.sample(bg_windows, min(n_samples, len(bg_windows)))
    sampled.sort(key=lambda x: x[0])

    accum       = np.zeros((N_CHANS, N_TIMES), dtype=np.float64)
    n_loaded    = 0
    loaded_pt   = None
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
        return torch.zeros(1, N_CHANS, N_TIMES)

    mean_bg  = (accum / n_loaded).astype(np.float32)
    return torch.from_numpy(mean_bg).unsqueeze(0)


def bandpass_filter(signal_np, low_hz, high_hz, fs=SAMPLING_RATE, order=4):
    nyq  = fs / 2.0
    low  = max(low_hz  / nyq, 1e-4)
    high = min(high_hz / nyq, 1.0 - 1e-4)
    sos  = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal_np, axis=-1).astype(np.float32)


def run_attribution(attributor, x_tensor, baseline_tensor, target_class):
    x_tensor        = x_tensor.to(DEVICE).float()
    baseline_tensor = baseline_tensor.to(DEVICE).float()
    attrs = attributor.attribute(
        x_tensor, baselines=baseline_tensor,
        target=target_class, n_steps=IG_N_STEPS,
    )
    return attrs.detach().cpu().squeeze(0).numpy()


def run_attribution_all_bands(attributor, x_np, target_class, baseline_np):
    result = {}
    for band_name, (lo, hi) in BANDS.items():
        filt_x        = bandpass_filter(x_np,        lo, hi)
        filt_baseline = bandpass_filter(baseline_np, lo, hi)
        x_t = torch.from_numpy(filt_x).unsqueeze(0)
        b_t = torch.from_numpy(filt_baseline).unsqueeze(0)
        result[band_name] = run_attribution(attributor, x_t, b_t, target_class)
    return result


def peak_brain_region(attr):
    scores = {
        region: float(np.maximum(attr[idxs], 0.0).mean())
        for region, idxs in BRAIN_REGIONS.items()
    }
    return max(scores, key=scores.get)


def peak_time_segment(attr):
    time_relevance = np.maximum(attr.mean(axis=0), 0.0)
    scores = {
        name: float(time_relevance[s:e].mean())
        for name, (s, e) in TIME_SEGMENTS.items()
    }
    return max(scores, key=scores.get)


def make_accumulators():
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


def accumulate(acc, band_attrs, class_key):
    for band, attr in band_attrs.items():
        pos = np.maximum(attr, 0.0)
        acc[class_key][band]["channel_sum"] += pos.mean(axis=1)
        acc[class_key][band]["heatmap_sum"] += np.abs(attr)
        acc[class_key][band]["count"]       += 1
        acc[class_key][band]["region_counts"][peak_brain_region(attr)] += 1
        acc[class_key][band]["time_counts"][peak_time_segment(attr)]   += 1


def finalize(acc, class_key):
    result = {}
    for band in BANDS:
        cnt = max(acc[class_key][band]["count"], 1)
        region_pct = {
            r: acc[class_key][band]["region_counts"].get(r, 0) / cnt * 100
            for r in BRAIN_REGIONS
        }
        time_raw = {s: acc[class_key][band]["time_counts"].get(s, 0) for s in TIME_SEGMENTS}
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


def plot_channel_by_band(sz, bg, out_dir, n_sei, n_bg):
    bands = list(BANDS.keys())
    y_pos = np.arange(N_CHANS)
    fig, axes = plt.subplots(1, 5, figsize=(22, 8), sharey=True)
    fig.suptitle(
        "Ensemble   Integrated Gradients Channel Attribution by Frequency Band\n"
        "(target: seizure class 1; baseline: mean background EEG)",
        fontsize=12,
    )
    for col, band in enumerate(bands):
        ax = axes[col]
        sz_vals = np.array(sz[band]["channel_mean"])
        bg_vals = np.array(bg[band]["channel_mean"])
        ax.barh(y_pos + 0.2, sz_vals, height=0.4, color="#C00000",
                label=f"Seizure (n={n_sei})")
        ax.barh(y_pos - 0.2, bg_vals, height=0.4, color="#2E75B6",
                label=f"Background (n={n_bg})")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=6)
        ax.set_title(f"{band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)", fontsize=9)
        ax.set_xlabel("Mean positive attribution", fontsize=7)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "channel_by_band.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difference_chart(sz, bg, out_dir, n_sei, n_bg):
    bands = list(BANDS.keys())
    y_pos = np.arange(N_CHANS)
    fig, axes = plt.subplots(1, 5, figsize=(22, 8), sharey=True)
    fig.suptitle(
        "Ensemble   IG Difference Chart: Seizure - Background\n"
        "Red = more seizure attention | Blue = more background attention",
        fontsize=12,
    )
    for col, band in enumerate(bands):
        ax = axes[col]
        sz_vals = np.array(sz[band]["channel_mean"])
        bg_vals = np.array(bg[band]["channel_mean"])
        diff = sz_vals - bg_vals
        colors = ["#C00000" if d > 0 else "#2E75B6" for d in diff]
        ax.barh(y_pos, diff, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=6)
        ax.set_title(f"{band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)", fontsize=9)
        ax.set_xlabel("Sz - Bg attribution", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "difference_chart.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_region_by_band(sz, bg, out_dir, n_sei, n_bg):
    bands   = list(BANDS.keys())
    regions = list(BRAIN_REGIONS.keys())
    x_pos   = np.arange(len(regions))
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle(
        "Ensemble   Integrated Gradients Brain Region Attribution by Frequency Band\n"
        "(target: seizure class 1; baseline: mean background EEG)",
        fontsize=11,
    )
    for col, band in enumerate(bands):
        ax = axes[col]
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
    fig.savefig(os.path.join(out_dir, "region_by_band.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_by_band(sz, bg, out_dir, n_sei, n_bg):
    bands      = list(BANDS.keys())
    segments   = list(TIME_SEGMENTS.keys())
    seg_labels = ["0-3s", "3-6s", "6-9s", "9-10s"]
    x_pos      = np.arange(len(segments))
    fig, axes  = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle(
        "Ensemble   Integrated Gradients Temporal Attribution by Frequency Band\n"
        "(target: seizure class 1; baseline: mean background EEG)",
        fontsize=11,
    )
    for col, band in enumerate(bands):
        ax = axes[col]
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
    fig.savefig(os.path.join(out_dir, "temporal_by_band.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps_by_band(sz, bg, out_dir):
    bands     = list(BANDS.keys())
    fig, axes = plt.subplots(2, 5, figsize=(26, 9))
    fig.suptitle(
        "Ensemble   Integrated Gradients Mean Attribution Heatmaps by Frequency Band\n"
        "(target: seizure class 1; baseline: mean background EEG)",
        fontsize=11,
    )
    for col, band in enumerate(bands):
        for row, (results, label) in enumerate([(sz, "Seizure"), (bg, "Background")]):
            ax   = axes[row][col]
            hmap = results[band]["heatmap_mean"]
            vmax = hmap.max() if hmap.max() > 0 else 1.0
            im = ax.imshow(
                hmap, aspect="auto", cmap="Reds",
                origin="upper", vmin=0, vmax=vmax,
                extent=[0, 10, N_CHANS, 0],
            )
            ax.set_yticks(np.arange(N_CHANS) + 0.5)
            ax.set_yticklabels(CHANNEL_NAMES, fontsize=5)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_title(f"{label}   {band}  ({BANDS[band][0]}-{BANDS[band][1]} Hz)", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "heatmaps_by_band.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dominant_band(sz, out_dir):
    bands = list(BANDS.keys())
    y_pos = np.arange(N_CHANS)

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
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=8)
    ax.set_xlabel("Peak band attribution score", fontsize=9)
    ax.set_title(
        "Ensemble   IG Dominant Frequency Band per Channel (Seizure)\n"
        "(target: seizure class 1; baseline: mean background EEG)",
        fontsize=10,
    )
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=BAND_COLORS[b],
                      label=f"{b}  ({BANDS[b][0]}-{BANDS[b][1]} Hz)")
        for b in bands
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "dominant_band_per_channel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(sz, bg, out_dir, total, correct, n_sei, n_bg, weight_2d):
    summary = {
        "model":              "Ensemble (CNN-LSTM-2D + EEGNet-1D)",
        "montage":            "Double Banana (18 bipolar pairs)",
        "xai_method":         "Integrated Gradients",
        "attribution_target": f"class {ATTRIBUTION_TARGET_CLASS} (seizure) for all windows",
        "ig_n_steps":         IG_N_STEPS,
        "ensemble_weight_2d": round(weight_2d, 4),
        "ensemble_weight_1d": round(1.0 - weight_2d, 4),
        "total_windows":      total,
        "ensemble_accuracy":  round(100 * correct / max(total, 1), 2),
        "n_seizure":          n_sei,
        "n_background":       n_bg,
        "baseline":           f"mean of {BASELINE_N_SAMPLES} background windows",
        "bands":              {},
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

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def save_heatmap_arrays(sz, bg, out_dir):
    for band in BANDS:
        np.save(os.path.join(out_dir, f"Ensemble_{band}_seizure_heatmap.npy"),
                sz[band]["heatmap_mean"])
        np.save(os.path.join(out_dir, f"Ensemble_{band}_background_heatmap.npy"),
                bg[band]["heatmap_mean"])


def run_xai_on_ensemble(wrapper):
    out_dir = "xai_ig_Ensemble"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    print("[manifest] reading manifest.jsonl...", flush=True)
    entries = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"[manifest] {len(entries)} entries", flush=True)

    print(f"[scan] reading pt files from Drive (this can take 1-3 min)...", flush=True)
    seizure_index, background_index = scan_manifest(entries)
    print(f"[scan] found {len(seizure_index)} seizure + {len(background_index)} background windows", flush=True)

    selected = sample_balanced(seizure_index, background_index)
    n_sei_sel = sum(1 for _, _, lbl in selected if lbl == 1)
    n_bg_sel  = sum(1 for _, _, lbl in selected if lbl == 0)
    print(f"[scan] selected {n_sei_sel} seizure + {n_bg_sel} background = {len(selected)} total", flush=True)

    print("[baseline] computing mean-background baseline...", flush=True)
    baseline_tensor = compute_mean_background_baseline(selected)
    baseline_np     = baseline_tensor.squeeze(0).numpy()
    print("[baseline] done", flush=True)

    csv_path = os.path.join(out_dir, "window_summary.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "file", "window_idx", "true_label", "pred_label",
        "prob_seizure_ensemble", "attribution_target",
        "delta_peak_region", "theta_peak_region", "alpha_peak_region",
        "beta_peak_region",  "gamma_peak_region",
        "delta_peak_time",   "theta_peak_time",   "alpha_peak_time",
        "beta_peak_time",    "gamma_peak_time",
    ])

    acc = {
        "seizure":    make_accumulators(),
        "background": make_accumulators(),
    }
    attributor = IntegratedGradients(wrapper)

    total        = 0
    correct      = 0
    n_seizure    = 0
    n_background = 0
    n_total_target = len(selected)
    print(f"[xai] starting attribution on {n_total_target} windows...", flush=True)
    t_start = time.time()

    for pt_path, group in groupby(selected, key=lambda x: x[0]):
        windows = list(group)
        data    = safe_load_pt(pt_path)
        if data is None:
            continue

        for _, w_idx, true_label in windows:
            x_tensor = data["x"][w_idx].unsqueeze(0).float()
            x_np     = data["x"][w_idx].numpy().astype(np.float32)

            with torch.no_grad():
                probs        = wrapper(x_tensor.to(DEVICE))
                prob_seizure = probs[0, 1].item()
            pred      = 1 if prob_seizure >= 0.5 else 0
            class_key = "seizure" if true_label == 1 else "background"

            total   += 1
            correct += int(pred == true_label)
            if true_label == 1:
                n_seizure    += 1
            else:
                n_background += 1

            band_attrs = run_attribution_all_bands(
                attributor, x_np,
                target_class=ATTRIBUTION_TARGET_CLASS,
                baseline_np=baseline_np,
            )
            accumulate(acc, band_attrs, class_key)

            writer.writerow([
                Path(pt_path).name, w_idx, true_label, pred,
                round(prob_seizure, 4), ATTRIBUTION_TARGET_CLASS,
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

            if total % 5 == 0 or total == n_total_target:
                elapsed = time.time() - t_start
                rate    = total / max(elapsed, 1e-3)
                eta     = (n_total_target - total) / max(rate, 1e-3)
                print(f"[xai] {total}/{n_total_target} windows | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining", flush=True)

    csv_file.close()
    print(f"[xai] attribution complete. {total} windows, {correct} correct ({100*correct/max(total,1):.1f}%)", flush=True)

    print("[plot] generating outputs...", flush=True)
    sz_results = finalize(acc, "seizure")
    bg_results = finalize(acc, "background")

    plot_channel_by_band(sz_results, bg_results, out_dir, n_seizure, n_background)
    plot_difference_chart(sz_results, bg_results, out_dir, n_seizure, n_background)
    plot_region_by_band(sz_results,  bg_results, out_dir, n_seizure, n_background)
    plot_temporal_by_band(sz_results, bg_results, out_dir, n_seizure, n_background)
    plot_heatmaps_by_band(sz_results, bg_results, out_dir)
    plot_dominant_band(sz_results, out_dir)
    save_summary(sz_results, bg_results, out_dir,
                 total, correct, n_seizure, n_background, wrapper.weight_2d)
    save_heatmap_arrays(sz_results, bg_results, out_dir)

    print("[done] all outputs saved to xai_ig_Ensemble/", flush=True)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    wrapper = load_ensemble(weight_2d=ENSEMBLE_WEIGHT_2D)
    run_xai_on_ensemble(wrapper)