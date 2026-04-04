# gradcam_eeg.py  —  Grad-CAM on 6 random windows (binary)
# Picks NUM_EACH random seizure + background windows from the
# binary eval dataset on Drive, runs Grad-CAM, saves PNGs.

import os
import json
import random
import shutil
import tempfile
import warnings

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
from dataloader import STFTSpec, SpecZNorm
MODEL_PATH    = "model_weights.pth"
EVAL_DIR      = r"G:\.shortcut-targets-by-id\1IS7vV1RQpfSoVy_vC4cp3EmiZ-sVdd6t\data V1\binary data\cache_windows_eval"
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.jsonl")
OUTPUT_FOLDER = "gradcam_results"
NUM_EACH      = 3   # seizure windows + background windows to analyse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# WINDOWS TEMP-COPY WORKAROUND
# torch.load fails on long Drive shortcut paths (Errno 22).
# Copy each .pt to a short local temp, load it, delete it.
# We always load to CPU first, then move tensors to DEVICE.
def load_pt(drive_path):
    """Copy to local temp before loading — fixes Errno 22 on Drive shortcut paths.
    Returns None if the file is a cloud-only stub or otherwise unreadable."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        shutil.copy2(drive_path, tmp_path)
        data = torch.load(tmp_path, map_location="cpu", weights_only=False)
        return data
    except OSError:
        return None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

def remap(pt_path_from_manifest):
    """Remap the original D:\\EEG_DATA\\... path to the Drive folder."""
    return os.path.join(EVAL_DIR, Path(pt_path_from_manifest).name)

# PREPROCESSING PIPELINE  matches the training exactly
# The transform runs on CPU — it outputs a tensor we then
# move to GPU for the model forward/backward pass.
transform = nn.Sequential(
    STFTSpec(fs=250, n_fft=256, hop_length=64, fmin=1.0, fmax=40.0),
    SpecZNorm(),
)
transform.eval()

def build_model():
    m = models.resnet18()
    m.conv1   = nn.Conv2d(41, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, 1)
    return m

class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self._features  = None
        self._gradients = None
        target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, "_features", out.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_gradients", go[0].detach()))

    def run(self, inp):
        # inp arrives on DEVICE already
        inp   = inp.float().requires_grad_(True)
        logit = self.model(inp)
        prob  = torch.sigmoid(logit).item()
        self.model.zero_grad()
        logit.backward()

        # Move feature maps and gradients to CPU for numpy ops
        weights = self._gradients.cpu().mean(dim=[2, 3], keepdim=True)
        feats   = self._features.cpu()
        cam     = torch.relu((weights * feats).sum(dim=1)).squeeze().numpy()

        if cam.ndim < 2 or cam.shape[0] == 0:
            cam = np.zeros((39, 40))
        else:
            cam = zoom(cam, (39 / cam.shape[0], 40 / cam.shape[1]), order=1)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, prob

BANDS = [
    ("Delta\n1-4 Hz",   0,  3),
    ("Theta\n4-8 Hz",   3,  7),
    ("Alpha\n8-13 Hz",  7, 12),
    ("Beta\n13-30 Hz", 12, 29),
    ("Gamma\n30+Hz",   29, 38),
]

def freq_bin_to_hz(bin_idx):
    return round(bin_idx * (250 / 256) + 1.95)

def plot_result(spec, cam, prob, true_label, window_id, save_path):
    predicted = "SEIZURE"    if prob > 0.5 else "BACKGROUND"
    true_str  = "SEIZURE"    if true_label == 1 else "BACKGROUND"
    correct   = (prob > 0.5) == (true_label == 1)
    title_col = "#ff6666"    if predicted == "SEIZURE" else "#6699ff"

    ytick_pos    = [(lo + hi) // 2 for _, lo, hi in BANDS]
    ytick_labels = [name for name, _, _ in BANDS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#0f0f1a")

    result_str = "✓ Correct" if correct else "✗ Wrong"
    fig.suptitle(
        f"{window_id}   |   True: {true_str}   |   Predicted: {predicted}"
        f"   |   P(seizure) = {prob:.1%}   |   {result_str}",
        color=title_col, fontsize=11, fontweight="bold"
    )

    def style_ax(ax, title):
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Time (0 → 10 seconds)", color="white", fontsize=9)
        ax.set_ylabel("Frequency band", color="white", fontsize=9)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, color="white", fontsize=8)
        ax.tick_params(colors="white")
        for _, _, hi in BANDS:
            ax.axhline(hi, color="white", linewidth=0.5, linestyle="--", alpha=0.3)

    ax1 = axes[0]
    ax1.imshow(spec.mean(axis=0), aspect="auto", origin="lower",
               cmap="viridis", interpolation="nearest")
    style_ax(ax1, "EEG Spectrogram\n(mean log-power across 41 channels)")

    ax2 = axes[1]
    im = ax2.imshow(cam, aspect="auto", origin="lower",
                    cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    style_ax(ax2, "Grad-CAM Attention Heatmap\n"
             "(bright = model focused here | dark = model ignored)")
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Attention intensity\n0 = ignored  →  1 = maximum focus",
                   color="white", fontsize=8)
    cbar.ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")

    peak_bin = int(cam.mean(axis=1).argmax())
    peak_hz  = freq_bin_to_hz(peak_bin)
    ax2.axhline(peak_bin, color="lime", linewidth=1.5, linestyle="--", alpha=0.8)
    ax2.text(1, peak_bin + 0.5, f"Peak ≈ {peak_hz} Hz", color="lime", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"    Saved → {save_path}")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading model...")
    model = build_model().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.\n")

    gradcam = GradCAM(model, model.layer4[-1].conv2)

    print("Scanning manifest...")
    seizure_wins    = []
    background_wins = []

    with open(MANIFEST_PATH) as f:
        for line in f:
            entry   = json.loads(line)
            pt_path = remap(entry["pt_path"])

            if not os.path.exists(pt_path):
                continue

            data = load_pt(pt_path)
            if data is None:
                print(f"  SKIPPED (cloud-only): {Path(pt_path).name}")
                continue
            labels = data["y"]
            for win_idx, lbl in enumerate(labels.tolist()):
                win = (pt_path, win_idx, int(lbl))
                if lbl == 1:
                    seizure_wins.append(win)
                else:
                    background_wins.append(win)

    print(f"  Seizure windows    : {len(seizure_wins)}")
    print(f"  Background windows : {len(background_wins)}")

    random.seed(42)
    selected = (
        random.sample(seizure_wins,    min(NUM_EACH, len(seizure_wins))) +
        random.sample(background_wins, min(NUM_EACH, len(background_wins)))
    )
    print(f"  Selected: {NUM_EACH} seizure + {NUM_EACH} background\n")

    summary = []

    for i, (pt_path, win_idx, true_label) in enumerate(selected):
        label_str = "SEI" if true_label == 1 else "BKG"
        window_id = f"{label_str}_{i+1:02d}"
        fname     = Path(pt_path).name

        print(f"[{i+1}/{len(selected)}] {window_id}  —  {fname}  window #{win_idx}")

        data  = load_pt(pt_path)
        if data is None:
            print(f"    SKIPPED (cloud-only): {Path(pt_path).name}")
            continue
        x_raw = data["x"][win_idx]

        with torch.no_grad():
            spec = transform(x_raw)   # CPU

        # Move spectrogram to GPU for the model pass
        inp = spec.unsqueeze(0).to(DEVICE)
        cam, prob = gradcam.run(inp)

        predicted  = "SEIZURE" if prob > 0.5 else "BACKGROUND"
        correct    = (prob > 0.5) == (true_label == 1)
        peak_bin   = int(cam.mean(axis=1).argmax())
        peak_hz    = freq_bin_to_hz(peak_bin)

        if peak_hz <= 4:    band = "Delta"
        elif peak_hz <= 8:  band = "Theta"
        elif peak_hz <= 13: band = "Alpha"
        elif peak_hz <= 30: band = "Beta"
        else:               band = "Gamma"

        true_disp = "SEIZURE" if true_label == 1 else "BACKGROUND"
        print(f"    True: {true_disp:<12} Predicted: {predicted:<12} "
              f"P(seizure): {prob:.3f}   Peak: ~{peak_hz} Hz ({band})   "
              f"{'✓ Correct' if correct else '✗ Wrong'}")

        out_path = os.path.join(OUTPUT_FOLDER, f"{window_id}.png")
        plot_result(spec.cpu().numpy(), cam, prob, true_label, window_id, out_path)

        summary.append((window_id, true_label, predicted, prob, correct, peak_hz, band))
        print()

    print("\n")
    print(f"{'Window':<10} {'True':<12} {'Predicted':<12} {'P(sei)':<10} {'Result':<10} {'Peak Hz'}")
    print("\n")
    print("\n")
    for wid, tl, pred, prob, ok, hz, band in summary:
        true_str = "SEIZURE" if tl == 1 else "BACKGROUND"
        print(f"{wid:<10} {true_str:<12} {pred:<12} {prob:<10.3f} "
              f"{'CORRECT' if ok else 'WRONG':<10} ~{hz} Hz → {band}")
    
    print(f"\nAll figures saved to: {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()