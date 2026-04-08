"""
Integrated Gradients + GradientSHAP XAI on two binary seizure detectors:
    1. EEGTCNet  (weights/eegtcn_afterSampling.pt)
    2. EEGNet    (weights/eegnet_afterLoader.pt)

Model input : [batch, 17, 2500]  17 EEG channels, 10 s at 250 Hz
Model output: 2-class softmax    0 = non_seizure, 1 = seizure

--- Integrated Gradients ---
Asks: if I slowly turn up the EEG signal from silence (zeros) to the real
recording, how much does each time sample on each channel push the model
toward predicting seizure?
Output shape per window: [17, 2500] — full input resolution, no upsampling.

--- GradientSHAP ---
Extends IG by sampling multiple noisy versions of the zero baseline instead
of using a single fixed baseline. This gives a stochastic estimate of the
SHAP values that is less sensitive to the exact choice of baseline.
Baseline here: zeros + small Gaussian noise (n_samples draws per window).
Output shape per window: [17, 2500] — same resolution as IG.

Why running both methods matters:
    IG and GradientSHAP share the same theoretical foundation (path-integral
    attribution) but differ in how they handle baseline uncertainty.
    Agreement between the two = robust finding.
    Disagreement = the attribution is sensitive to baseline choice, which is
    itself a finding worth reporting.

17 channels are the PURE_BRAIN_CHANNELS (fp1, fp2, a1, a2 removed as noise):
    Frontal:   f7, f3, fz, f4, f8
    Temporal:  t3, t4, t5, t6
    Central:   c3, cz, c4
    Parietal:  p3, pz, p4
    Occipital: o1, o2

Output structure per model (example for EEGTCNet):
    xai_results_EEGTCNet/
        ig/
            summary_ig.csv
            channel_attribution.png
            time_attribution.png
            brain_region_attribution.png
            attribution_heatmap.png
        gradshap/
            summary_gradshap.csv
            channel_attribution.png
            time_attribution.png
            brain_region_attribution.png
            attribution_heatmap.png
        comparison_ig_vs_gradshap.png   <-- IG vs GradSHAP side-by-side
"""

import os
import sys
import csv
import json
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from captum.attr import GradientShap

sys.path.append(str(Path(__file__).resolve().parent))
from braindecode.models import EEGNetv4


DRIVE_DATA_DIR = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\cache_windows_binary_10_sec_eval"
MANIFEST_PATH  = os.path.join(DRIVE_DATA_DIR, "manifest.jsonl")

EEGTCNET_CHECKPOINT = "weights/eegtcn_afterSampling.pt"
EEGNET_CHECKPOINT   = "weights/eegnet_afterLoader.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WINDOWS = None

IG_STEPS = 100

GS_SAMPLES = 50

# Noise standard deviation added to zero baseline for GradientSHAP.
# Small enough not to alter signal character, large enough for stochasticity.
GS_NOISE_STD = 0.01

N_CHANS   = 17
N_OUTPUTS = 2
N_TIMES   = 2500   # 10 seconds x 250 Hz


CHANNEL_NAMES = [
    "f7", "f3", "fz", "f4", "f8",    # indices 0-4   frontal
    "t3", "c3", "cz", "c4", "t4",    # indices 5-9   temporal/central
    "t5", "p3", "pz", "p4", "t6",    # indices 10-14 temporal/parietal
    "o1", "o2",                        # indices 15-16 occipital
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


def load_eegnet(checkpoint_path: str, device: str) -> nn.Module:
    model = EEGNetv4(
        n_chans=N_CHANS,
        n_outputs=N_OUTPUTS,
        n_times=N_TIMES,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded epoch {checkpoint.get('epoch', '?')} | "
              f"best metric = {checkpoint.get('best_metric', '?')}")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint.to(device)

    model.eval()
    return model


def safe_load_pt(pt_path: str):
    """
    Copy .pt file to a local temp path before loading.
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


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,         # [1, 17, 2500]
    baseline: torch.Tensor,  # [1, 17, 2500] zeros
    target_class: int,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single window.

    Steps:
    1. Build n_steps interpolations from baseline (silence) to x (real EEG).
    2. Forward pass all interpolations in one batch.
    3. Backward pass — gradients of the target class score w.r.t. each input.
    4. Average gradients across n_steps (approximates the path integral).
    5. Multiply by (x - baseline) to scale each sample's contribution.

    Returns: numpy array [17, 2500]
    """
    x        = x.to(DEVICE).float()
    baseline = baseline.to(DEVICE).float()

    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1, 1)

    x_sq = x.squeeze(0)        # [17, 2500]
    b_sq = baseline.squeeze(0) # [17, 2500]

    interpolated = (b_sq + alphas * (x_sq - b_sq)).requires_grad_(True)

    logits      = model(interpolated)
    class_score = logits[:, target_class].sum()

    model.zero_grad()
    class_score.backward()

    grads     = interpolated.grad.detach()  # [n_steps, 17, 2500]
    avg_grads = grads.mean(dim=0)           # [17, 2500]

    ig = (x_sq - b_sq) * avg_grads
    return ig.detach().cpu().numpy()        # [17, 2500]


def gradient_shap(
    model: nn.Module,
    x: torch.Tensor,        # [1, 17, 2500]
    target_class: int,
    n_samples: int = 20,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Compute GradientSHAP for a single window using captum.

    GradientSHAP extends IG by using a distribution of baselines instead
    of a single fixed one. Here the baseline distribution is:
        zeros + Gaussian noise with std = noise_std

    captum's GradientShap handles the sampling and averaging internally.
    For each of n_samples draws it runs IG between a noisy-zero baseline
    and the input, then averages across all draws.

    n_samples baselines are stacked as [n_samples, 17, 2500] and passed
    to captum alongside the single input [1, 17, 2500].

    Returns: numpy array [17, 2500]
    """
    x = x.to(DEVICE).float()

    # Build n_samples noisy-zero baselines: [n_samples, 17, 2500]
    baselines = torch.randn(n_samples, N_CHANS, N_TIMES, device=DEVICE) * noise_std

    gs   = GradientShap(model)
    attr = gs.attribute(
        x,
        baselines=baselines,
        target=target_class,
        n_samples=n_samples,
        stdevs=noise_std,
    )
    # attr shape from captum: [1, 17, 2500]
    return attr.squeeze(0).detach().cpu().numpy()  # [17, 2500]



def peak_time_segment(attr_map: np.ndarray) -> str:
    """attr_map: [17, 2500] — returns name of time segment with highest mean positive attribution."""
    time_relevance = np.maximum(attr_map.mean(axis=0), 0)
    scores = {
        name: float(time_relevance[s:e].mean())
        for name, (s, e) in TIME_SEGMENTS.items()
    }
    return max(scores, key=scores.get)


def peak_channel(attr_map: np.ndarray) -> int:
    """Returns channel index with highest total positive attribution."""
    return int(np.argmax(np.maximum(attr_map, 0).sum(axis=1)))


def peak_brain_region(attr_map: np.ndarray) -> str:
    """Returns brain region with highest mean positive attribution."""
    scores = {
        region: float(np.maximum(attr_map[idxs, :], 0).mean())
        for region, idxs in BRAIN_REGIONS.items()
    }
    return max(scores, key=scores.get)


def run_xai(model: nn.Module, model_name: str):
    """
    Runs IG and GradientSHAP on every eval window for one model.
    Results go into:
        xai_results_{model_name}/ig/
        xai_results_{model_name}/gradshap/
        xai_results_{model_name}/comparison_ig_vs_gradshap.png
    """
    base_dir = f"xai_results_{model_name}"
    ig_dir   = os.path.join(base_dir, "ig")
    gs_dir   = os.path.join(base_dir, "gradshap")
    os.makedirs(ig_dir, exist_ok=True)
    os.makedirs(gs_dir, exist_ok=True)

    
    print(f"  Running IG + GradientSHAP on: {model_name}")
    print(f"  IG output   : {ig_dir}")
    print(f"  SHAP output : {gs_dir}")
    

    baseline = torch.zeros(1, N_CHANS, N_TIMES, device=DEVICE)

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    entries = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Manifest entries: {len(entries)} files\n")

    
    ig_csv_path = os.path.join(ig_dir, "summary_ig.csv")
    gs_csv_path = os.path.join(gs_dir, "summary_gradshap.csv")

    ig_csv_file = open(ig_csv_path, "w", newline="", encoding="utf-8")
    gs_csv_file = open(gs_csv_path, "w", newline="", encoding="utf-8")

    csv_header = [
        "file", "window_idx",
        "true_label", "pred_label", "prob_seizure",
        "peak_time_segment", "peak_channel", "peak_brain_region",
        "correct",
    ]
    ig_writer = csv.writer(ig_csv_file)
    gs_writer = csv.writer(gs_csv_file)
    ig_writer.writerow(csv_header)
    gs_writer.writerow(csv_header)

    
    def make_accumulators():
        return {
            "ig_accum":      {"seizure": [], "background": []},
            "time_counts":   {"seizure": defaultdict(int), "background": defaultdict(int)},
            "region_counts": {"seizure": defaultdict(int), "background": defaultdict(int)},
            "chan_scores":   {"seizure": np.zeros(N_CHANS), "background": np.zeros(N_CHANS)},
            "n_windows":     {"seizure": 0, "background": 0},
        }

    ig_acc = make_accumulators()
    gs_acc = make_accumulators()

    
    comparison = {
        "ig_sei": np.zeros((N_CHANS, N_TIMES)),
        "ig_bkg": np.zeros((N_CHANS, N_TIMES)),
        "gs_sei": np.zeros((N_CHANS, N_TIMES)),
        "gs_bkg": np.zeros((N_CHANS, N_TIMES)),
        "n_sei":  0,
        "n_bkg":  0,
    }

    total   = 0
    correct = 0

    for entry in entries:
        if MAX_WINDOWS is not None and total >= MAX_WINDOWS:
            break

        pt_path = resolve_pt_path(entry["pt_path"])
        if not os.path.exists(pt_path):
            print(f"  [MISSING] {Path(pt_path).name}")
            continue

        data = safe_load_pt(pt_path)
        if data is None:
            continue

        x_all = data["x"]   # [N, 17, 2500]
        y_all = data["y"]   # [N]
        N     = len(y_all)
        print(f"  {Path(pt_path).name} | {N} windows")

        for w_idx in range(N):
            if MAX_WINDOWS is not None and total >= MAX_WINDOWS:
                break

            true_label = int(y_all[w_idx].item())
            x          = x_all[w_idx].unsqueeze(0).float()  # [1, 17, 2500]
            class_key  = "seizure" if true_label == 1 else "background"

            with torch.no_grad():
                logits       = model(x.to(DEVICE))
                probs        = torch.softmax(logits, dim=1)
                prob_seizure = probs[0, 1].item()
            pred = 1 if prob_seizure >= 0.5 else 0

            is_correct = int(pred == true_label)
            total   += 1
            correct += is_correct

            # --- Integrated Gradients ---
            ig_map = integrated_gradients(
                model, x, baseline,
                target_class=true_label,
                n_steps=IG_STEPS,
            )
            _accumulate(ig_acc, ig_map, class_key)
            _write_csv_row(ig_writer, pt_path, w_idx, true_label, pred,
                           prob_seizure, ig_map, is_correct)

            # --- GradientSHAP ---
            gs_map = gradient_shap(
                model, x,
                target_class=true_label,
                n_samples=GS_SAMPLES,
                noise_std=GS_NOISE_STD,
            )
            _accumulate(gs_acc, gs_map, class_key)
            _write_csv_row(gs_writer, pt_path, w_idx, true_label, pred,
                           prob_seizure, gs_map, is_correct)

            # Update running sums for comparison figure
            if class_key == "seizure":
                comparison["ig_sei"] += np.maximum(ig_map, 0)
                comparison["gs_sei"] += np.maximum(gs_map, 0)
                comparison["n_sei"]  += 1
            else:
                comparison["ig_bkg"] += np.maximum(ig_map, 0)
                comparison["gs_bkg"] += np.maximum(gs_map, 0)
                comparison["n_bkg"]  += 1

    ig_csv_file.close()
    gs_csv_file.close()

    n_sei = ig_acc["n_windows"]["seizure"]
    n_bkg = ig_acc["n_windows"]["background"]

    print(f"\nProcessed : {total} windows ({n_sei} seizure, {n_bkg} background)")
    print(f"Accuracy  : {correct}/{total} = {100*correct/max(total,1):.1f}%")
    print(f"IG CSV    : {ig_csv_path}")
    print(f"SHAP CSV  : {gs_csv_path}")

    # Normalize channel scores by window count
    for acc in (ig_acc, gs_acc):
        for ck in ("seizure", "background"):
            n = acc["n_windows"][ck]
            if n > 0:
                acc["chan_scores"][ck] /= n

    # Normalize comparison maps from running sums to means
    if comparison["n_sei"] > 0:
        comparison["ig_sei"] /= comparison["n_sei"]
        comparison["gs_sei"] /= comparison["n_sei"]
    if comparison["n_bkg"] > 0:
        comparison["ig_bkg"] /= comparison["n_bkg"]
        comparison["gs_bkg"] /= comparison["n_bkg"]

    # IG figures
    _make_channel_figure(ig_acc["chan_scores"],  n_sei, n_bkg, ig_dir, model_name, "IG")
    _make_time_figure(ig_acc["time_counts"],     n_sei, n_bkg, total, correct, ig_dir, model_name, "IG")
    _make_region_figure(ig_acc["region_counts"], n_sei, n_bkg, ig_dir, model_name, "IG")
    _make_heatmap_figure(ig_acc["ig_accum"],     ig_dir, model_name, "IG")

    # GradSHAP figures
    _make_channel_figure(gs_acc["chan_scores"],  n_sei, n_bkg, gs_dir, model_name, "GradientSHAP")
    _make_time_figure(gs_acc["time_counts"],     n_sei, n_bkg, total, correct, gs_dir, model_name, "GradientSHAP")
    _make_region_figure(gs_acc["region_counts"], n_sei, n_bkg, gs_dir, model_name, "GradientSHAP")
    _make_heatmap_figure(gs_acc["ig_accum"],     gs_dir, model_name, "GradientSHAP")

    # Comparison figure (the key figure for the paper)
    _make_comparison_figure(comparison, base_dir, model_name)


def _accumulate(acc: dict, attr_map: np.ndarray, class_key: str):
    """Update running accumulators for channel scores, time, region, heatmap."""
    acc["chan_scores"][class_key] += np.maximum(attr_map, 0).mean(axis=1)
    acc["n_windows"][class_key]  += 1

    if len(acc["ig_accum"][class_key]) < 100:
        acc["ig_accum"][class_key].append(attr_map.copy())

    acc["time_counts"][class_key][peak_time_segment(attr_map)]  += 1
    acc["region_counts"][class_key][peak_brain_region(attr_map)] += 1


def _write_csv_row(writer, pt_path, w_idx, true_label, pred,
                   prob_seizure, attr_map, is_correct):
    writer.writerow([
        Path(pt_path).name, w_idx,
        true_label, pred, round(prob_seizure, 4),
        peak_time_segment(attr_map),
        CHANNEL_NAMES[peak_channel(attr_map)],
        peak_brain_region(attr_map),
        is_correct,
    ])


def _make_channel_figure(channel_scores, n_sei, n_bkg, output_dir, model_name, method_label):
    sei_scores = channel_scores["seizure"]
    bkg_scores = channel_scores["background"]
    x = np.arange(N_CHANS)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"{model_name} | {method_label} — Per-Channel Attribution\n"
        "(Mean positive attribution — higher = model focused here more)",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    ax.bar(x - w/2, sei_scores, width=w, color="#e05252", label=f"Seizure (n={n_sei})")
    ax.bar(x + w/2, bkg_scores, width=w, color="#5285e0", label=f"Background (n={n_bkg})")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean positive attribution")
    ax.set_title("Seizure vs Background — Channel Attribution")
    ax.legend()

    ax = axes[1]
    diff   = sei_scores - bkg_scores
    colors = ["#e05252" if d > 0 else "#5285e0" for d in diff]
    ax.bar(x, diff, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Seizure minus Background attribution")
    ax.set_title("Difference: Red = more attention for seizure")

    plt.tight_layout()
    out = os.path.join(output_dir, "channel_attribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _make_time_figure(time_counts, n_sei, n_bkg, total, correct, output_dir, model_name, method_label):
    seg_names = list(TIME_SEGMENTS.keys())
    sei_pct   = [time_counts["seizure"].get(s, 0) / max(n_sei, 1) * 100 for s in seg_names]
    bkg_pct   = [time_counts["background"].get(s, 0) / max(n_bkg, 1) * 100 for s in seg_names]
    x = np.arange(len(seg_names))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"{model_name} | {method_label} — Temporal Attribution\n"
        f"{total} windows | Accuracy {100*correct/max(total,1):.1f}%",
        fontsize=12, fontweight="bold"
    )

    for ax, pct, label, color in [
        (axes[0], sei_pct, f"Seizure (n={n_sei})",    "#e05252"),
        (axes[1], bkg_pct, f"Background (n={n_bkg})", "#5285e0"),
    ]:
        bars = ax.bar(x, pct, color=color, width=w*2)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(seg_names, fontsize=8, rotation=10)
        ax.set_ylabel("% of windows (peak time segment)")
        ax.set_ylim(0, 100)
        for bar, v in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width()/2, v+1,
                    f"{v:.1f}%", ha="center", fontsize=8)

    ax = axes[2]
    ax.bar(x - w/2, sei_pct, width=w, color="#e05252", label="Seizure")
    ax.bar(x + w/2, bkg_pct, width=w, color="#5285e0", label="Background")
    ax.set_title("Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_names, fontsize=8, rotation=10)
    ax.set_ylabel("% of windows")
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "time_attribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _make_region_figure(region_counts, n_sei, n_bkg, output_dir, model_name, method_label):
    region_names = list(BRAIN_REGIONS.keys())
    sei_pct = [region_counts["seizure"].get(r, 0) / max(n_sei, 1) * 100 for r in region_names]
    bkg_pct = [region_counts["background"].get(r, 0) / max(n_bkg, 1) * 100 for r in region_names]
    x = np.arange(len(region_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"{model_name} | {method_label} — Brain Region Attribution\n"
        "(% of windows where this region had peak positive attribution)",
        fontsize=12, fontweight="bold"
    )

    ax.bar(x - w/2, sei_pct, width=w, color="#e05252", label=f"Seizure (n={n_sei})")
    ax.bar(x + w/2, bkg_pct, width=w, color="#5285e0", label=f"Background (n={n_bkg})")
    ax.set_xticks(x)
    ax.set_xticklabels(region_names, fontsize=11)
    ax.set_ylabel("% of windows (peak brain region)")
    ax.set_ylim(0, 100)
    ax.legend()

    for i, (sv, bv) in enumerate(zip(sei_pct, bkg_pct)):
        ax.text(i - w/2, sv+1, f"{sv:.1f}%", ha="center", fontsize=9, color="#e05252")
        ax.text(i + w/2, bv+1, f"{bv:.1f}%", ha="center", fontsize=9, color="#5285e0")

    plt.tight_layout()
    out = os.path.join(output_dir, "brain_region_attribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _make_heatmap_figure(ig_accumulator, output_dir, model_name, method_label):
    """
    2D heatmap of mean positive attribution: channels (y) vs time (x).
    One row for seizure windows, one for background.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(
        f"{model_name} | {method_label} — Mean Attribution Heatmap\n"
        "Y-axis: EEG channels | X-axis: time samples (0 to 2500 = 10 seconds)",
        fontsize=12, fontweight="bold"
    )

    for ax, class_key, label in [
        (axes[0], "seizure",    "Seizure windows"),
        (axes[1], "background", "Background windows"),
    ]:
        maps = ig_accumulator[class_key]
        if not maps:
            ax.set_title(f"{label} — no data")
            continue

        mean_map = np.mean(np.stack(maps, axis=0), axis=0)  # [17, 2500]
        mean_pos = np.maximum(mean_map, 0)

        im = ax.imshow(mean_pos, aspect="auto", origin="upper", cmap="Reds")
        ax.set_yticks(range(N_CHANS))
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=8)
        ax.set_xlabel("Time sample (250 samples = 1 second)")
        ax.set_title(f"{label} (n={len(maps)} windows averaged)")

        for s in range(0, N_TIMES, 250):
            ax.axvline(s, color="white", linewidth=0.4, alpha=0.5)

        plt.colorbar(im, ax=ax, label="Mean positive attribution")

    plt.tight_layout()
    out = os.path.join(output_dir, "attribution_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _make_comparison_figure(comparison: dict, base_dir: str, model_name: str):
    """
    Side-by-side IG vs GradientSHAP heatmaps.
    Layout: 2 rows (seizure / background) x 2 columns (IG / GradSHAP).
    This is the key paper figure — shows whether both methods agree on
    which channels and time segments drive the seizure prediction.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(
        f"{model_name} — Integrated Gradients vs GradientSHAP\n"
        "Mean positive attribution | Y: channels | X: time (10 s)\n"
        "Agreement = robust finding. Divergence = baseline-sensitive attribution.",
        fontsize=12, fontweight="bold"
    )

    panels = [
        (axes[0, 0], comparison["ig_sei"],  "IG — Seizure",          comparison["n_sei"]),
        (axes[0, 1], comparison["gs_sei"],  "GradientSHAP — Seizure",comparison["n_sei"]),
        (axes[1, 0], comparison["ig_bkg"],  "IG — Background",        comparison["n_bkg"]),
        (axes[1, 1], comparison["gs_bkg"],  "GradientSHAP — Background", comparison["n_bkg"]),
    ]

    for ax, data, title, n in panels:
        if n == 0:
            ax.set_title(f"{title} — no data")
            continue

        im = ax.imshow(data, aspect="auto", origin="upper", cmap="Reds")
        ax.set_yticks(range(N_CHANS))
        ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)
        ax.set_xlabel("Time sample")
        ax.set_title(f"{title} (n={n})", fontsize=10)

        for s in range(0, N_TIMES, 250):
            ax.axvline(s, color="white", linewidth=0.4, alpha=0.5)

        plt.colorbar(im, ax=ax, label="Mean positive attribution")

    plt.tight_layout()
    out = os.path.join(base_dir, "comparison_ig_vs_gradshap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print(f"Device      : {DEVICE}")
    print(f"IG steps    : {IG_STEPS}")
    print(f"SHAP samples: {GS_SAMPLES}  noise_std={GS_NOISE_STD}")
    print(f"Max windows : {MAX_WINDOWS if MAX_WINDOWS else 'all'}\n")

    

    print("Loading EEGTCNet checkpoint (EEGNet architecture, post-sampling weights)...")
    model_tcnet = load_eegnet(EEGTCNET_CHECKPOINT, DEVICE)
    run_xai(model_tcnet, "EEGTCNet")
    del model_tcnet
    torch.cuda.empty_cache()

    print("\nLoading EEGNet checkpoint (EEGNet architecture, post-loader weights)...")
    model_eegnet = load_eegnet(EEGNET_CHECKPOINT, DEVICE)
    run_xai(model_eegnet, "EEGNet")
    del model_eegnet
    torch.cuda.empty_cache()

    print("\nDone.")
    print("Results in:")
    print("  xai_results_EEGTCNet/ig/")
    print("  xai_results_EEGTCNet/gradshap/")
    print("  xai_results_EEGTCNet/comparison_ig_vs_gradshap.png")
    print("  xai_results_EEGNet/ig/")
    print("  xai_results_EEGNet/gradshap/")
    print("  xai_results_EEGNet/comparison_ig_vs_gradshap.png")
   