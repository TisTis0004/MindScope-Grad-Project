"""
xai_integrated_gradients.py
============================
Integrated Gradients XAI on the MindScope Cascade Stage 1 binary CNN.

What Integrated Gradients does (plain English):
  - It asks: "if I slowly turn up the EEG signal from silence (zeros)
    to the real signal, how much does each time sample contribute to
    pushing the prediction toward seizure?"
  - Every one of the 250 time steps gets a relevance score per channel.
  - Positive score  -> pushed toward seizure
  - Negative score  -> pushed toward background
  - Output shape per window: [21 channels, 250 time steps]

Why this works better than Grad-CAM here:
  - Works at full input resolution (250 steps), no upsampling needed
  - Does not depend on architecture depth — works on 1-layer and
    3-layer CNNs equally well
  - Consistent method that can be reused on Stage 2 with zero changes

Outputs:
  - xai_results/summary_ig.csv   — one row per window
  - xai_results/summary_ig.png   — summary figure

Run from project root:
  python xai_integrated_gradients.py
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

sys.path.append(str(Path(__file__).resolve().parent))
from models.models import BinarySeizureCNN



MODEL_PATH   = "Binary_Balanced_Run_1.pt"
MANIFEST     = r"G:\.shortcut-targets-by-id\1NILfnYdShyoztLuAa5RVrCdsfnBDubyl\cache_windows_eval_8_classes\manifest.jsonl"
DATA_DIR     = r"G:\.shortcut-targets-by-id\1NILfnYdShyoztLuAa5RVrCdsfnBDubyl\cache_windows_eval_8_classes"
OUTPUT_DIR   = "xai_results_balanced"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# How many windows to process. 
MAX_WINDOWS  = None 

# Number of steps for the integral approximation.
# 50 is the standard. Higher = more accurate but slower.
IG_STEPS     = 50

# FREQUENCY BANDS — for the summary analysis
# EEG is bandpass filtered 0.5-40 Hz, sampled at 250 Hz
# Each time step = 4ms, so 250 steps = 1 second

# Time segments (in samples out of 250) for analysis
SEGMENTS = {
    "Early (0-83ms)":  (0,   83),
    "Mid (83-166ms)":  (83,  166),
    "Late (166-250ms)":(166, 250),
}


def load_model(model_path: str, device: str) -> nn.Module:
    """Load BinarySeizureCNN from checkpoint."""
    model = BinarySeizureCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint.to(device)

    model.eval()
    return model


def safe_load_pt(pt_path: str):
    """
    Copy .pt file to local temp before loading.
    Prevents Errno 22 on long Google Drive shortcut paths.
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


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,          # shape [1, 21, 250]
    baseline: torch.Tensor,   # shape [1, 21, 250] — zeros (silence)
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single window.

    How it works step by step:
      1. Create n_steps interpolations from baseline (silence) to x (real EEG)
      2. Run a forward+backward pass at each interpolation point
      3. Collect the gradient of the output with respect to the input at each step
      4. Average those gradients (this approximates the integral)
      5. Multiply by (x - baseline) — this scales each sample's contribution

    The result tells you: for each of the 21 channels and 250 time steps,
    how much did that sample contribute to the final seizure probability?

    Returns: numpy array of shape [21, 250]
    """
    x        = x.to(DEVICE)
    baseline = baseline.to(DEVICE)

    # Build interpolated inputs: alpha goes from 0 to 1 in n_steps
    # Shape: [n_steps, 1, 21, 250]
    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1, 1, 1)
    interpolated = baseline + alphas * (x - baseline)  # [n_steps, 1, 21, 250]
    interpolated = interpolated.squeeze(1)              # [n_steps, 21, 250]
    interpolated.requires_grad_(True)

    # Forward pass on all interpolations at once
    logits = model(interpolated)          # [n_steps, 1]
    output = torch.sigmoid(logits).sum()  # scalar — sum so backward works

    # Backward pass — get gradients w.r.t. interpolated inputs
    model.zero_grad()
    output.backward()

    grads = interpolated.grad.detach()    # [n_steps, 21, 250]

    # Average gradients across interpolation steps (trapezoidal approximation)
    avg_grads = grads.mean(dim=0)         # [21, 250]

    # Scale by (input - baseline) — this is the IG formula
    integrated_grads = ((x.squeeze(0) - baseline.squeeze(0)) * avg_grads)
    return integrated_grads.detach().cpu().numpy()   # [21, 250]

# SEGMENT ANALYSIS — which time segment had the most positive relevance?

def peak_segment(ig_map: np.ndarray) -> str:
    """
    ig_map: [21, 250]
    Average over channels, take only positive relevance,
    find which time segment has the highest mean.
    Returns segment name.
    """
    time_relevance = ig_map.mean(axis=0)          # [250] — mean over channels
    positive       = np.maximum(time_relevance, 0) # only positive contributions

    scores = {}
    for name, (start, end) in SEGMENTS.items():
        region = positive[start:end]
        scores[name] = float(region.mean()) if len(region) > 0 else 0.0

    return max(scores, key=scores.get)


def peak_channel(ig_map: np.ndarray) -> int:
    """
    ig_map: [21, 250]
    Returns the channel index (0-20) with the highest total positive relevance.
    This tells you which EEG electrode contributed most.
    """
    channel_relevance = np.maximum(ig_map, 0).sum(axis=1)  # [21]
    return int(np.argmax(channel_relevance))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH}")
    print()

    
    print("Loading model...")
    model = load_model(MODEL_PATH, DEVICE)
    print("Model loaded.\n")

    # Baseline: silence — a tensor of zeros, same shape as input [1, 21, 250]
    baseline = torch.zeros(1, 21, 250, device=DEVICE)

    print(f"Reading manifest: {MANIFEST}")
    entries = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Manifest entries: {len(entries)} files\n")

    csv_path = os.path.join(OUTPUT_DIR, "summary_ig.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "file", "window_idx",
        "true_label",         # 0=background, 1=seizure
        "pred_label",         # model prediction
        "prob_seizure",       # sigmoid probability
        "ig_peak_segment",    # which time segment had most positive IG
        "ig_peak_channel",    # which EEG channel had most positive IG
        "correct",
    ])

    seg_counts = {
        "seizure":    defaultdict(int),
        "background": defaultdict(int),
    }
    total   = 0
    correct = 0

    for entry in entries:
        # Remap manifest path to local DATA_DIR (same pattern as gradcam_all.py)
        pt_filename = Path(entry["pt_path"]).name
        pt_path     = os.path.join(DATA_DIR, pt_filename)

        if not os.path.exists(pt_path):
            print(f"  [MISSING] {pt_filename}")
            continue

        data = safe_load_pt(pt_path)
        if data is None:
            continue

        x_all = data["x"]   # [N, 21, 250]
        y_all = data["y"]   # [N]
        N     = len(y_all)

        print(f"  {pt_filename} | {N} windows")

        for w_idx in range(N):
            if MAX_WINDOWS is not None and total >= MAX_WINDOWS:
                break

            raw_label = int(y_all[w_idx].item())
            # Binary mapping: 0=background, anything else=seizure
            true_label = 1 if raw_label > 0 else 0

            # Input tensor: [1, 21, 250]
            x = x_all[w_idx].unsqueeze(0).float()

            # Forward pass for prediction
            with torch.no_grad():
                logit = model(x.to(DEVICE))
                prob  = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0

            # Integrated Gradients backward pass
            ig_map = integrated_gradients(model, x, baseline, n_steps=IG_STEPS)
            # ig_map shape: [21, 250]

            # Analysis
            seg  = peak_segment(ig_map)
            chan = peak_channel(ig_map)

            # Record
            class_key = "seizure" if true_label == 1 else "background"
            seg_counts[class_key][seg] += 1

            is_correct = int(pred == true_label)
            total   += 1
            correct += is_correct

            writer.writerow([
                pt_filename, w_idx,
                true_label, pred, round(prob, 4),
                seg, chan, is_correct,
            ])

        if MAX_WINDOWS is not None and total >= MAX_WINDOWS:
            print(f"\nReached MAX_WINDOWS={MAX_WINDOWS}, stopping early.")
            break

    csv_file.close()
    print(f"\nProcessed {total} windows")
    print(f"Accuracy : {correct}/{total} = {100*correct/max(total,1):.1f}%")
    print(f"CSV saved -> {csv_path}")

    _make_figure(seg_counts, total, correct)


def _make_figure(seg_counts, total, correct):
    """Three-panel summary figure."""
    seg_names = list(SEGMENTS.keys())
    sei_total = max(sum(seg_counts["seizure"].values()), 1)
    bkg_total = max(sum(seg_counts["background"].values()), 1)

    sei_pct = [seg_counts["seizure"].get(s, 0) / sei_total * 100 for s in seg_names]
    bkg_pct = [seg_counts["background"].get(s, 0) / bkg_total * 100 for s in seg_names]

    x = np.arange(len(seg_names))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"MindScope Stage 1 — Integrated Gradients\n"
        f"{total} windows | Accuracy {100*correct/max(total,1):.1f}%",
        fontsize=13, fontweight="bold"
    )

    # Panel 1 — Seizure
    ax = axes[0]
    bars = ax.bar(x, sei_pct, color="#e05252", width=w * 2)
    ax.set_title("IG Attribution — Seizure Windows")
    ax.set_xticks(x); ax.set_xticklabels(seg_names, fontsize=8)
    ax.set_ylabel("% of windows (peak segment)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, sei_pct):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)

    # Panel 2 — Background
    ax = axes[1]
    bars = ax.bar(x, bkg_pct, color="#5285e0", width=w * 2)
    ax.set_title("IG Attribution — Background Windows")
    ax.set_xticks(x); ax.set_xticklabels(seg_names, fontsize=8)
    ax.set_ylabel("% of windows (peak segment)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, bkg_pct):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)

    # Panel 3 — Comparison
    ax = axes[2]
    ax.bar(x - w/2, sei_pct, width=w, color="#e05252", label="Seizure")
    ax.bar(x + w/2, bkg_pct, width=w, color="#5285e0", label="Background")
    ax.set_title("Seizure vs Background Comparison")
    ax.set_xticks(x); ax.set_xticklabels(seg_names, fontsize=8)
    ax.set_ylabel("% of windows (peak segment)")
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "summary_ig.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    main()