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


MODEL_PATH   = "Binary_Imbalanced_Finetune_Run_2.pt"
MANIFEST     = r"G:\.shortcut-targets-by-id\1NILfnYdShyoztLuAa5RVrCdsfnBDubyl\cache_windows_eval_8_classes\manifest.jsonl"
DATA_DIR     = r"G:\.shortcut-targets-by-id\1NILfnYdShyoztLuAa5RVrCdsfnBDubyl\cache_windows_eval_8_classes"
OUTPUT_DIR   = "xai_results_imbalanced"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MAX_WINDOWS  = None
IG_STEPS     = 50

SEGMENTS = {
    "Early (0-83ms)":  (0,   83),
    "Mid (83-166ms)":  (83,  166),
    "Late (166-250ms)":(166, 250),
}


def load_model(model_path: str, device: str) -> nn.Module:
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
    x: torch.Tensor,
    baseline: torch.Tensor,
    n_steps: int = 50,
) -> np.ndarray:
    x        = x.to(DEVICE)
    baseline = baseline.to(DEVICE)

    alphas = torch.linspace(0, 1, n_steps, device=DEVICE).view(-1, 1, 1, 1)
    interpolated = baseline + alphas * (x - baseline)
    interpolated = interpolated.squeeze(1)
    interpolated.requires_grad_(True)

    logits = model(interpolated)
    output = torch.sigmoid(logits).sum()

    model.zero_grad()
    output.backward()

    grads = interpolated.grad.detach()
    avg_grads = grads.mean(dim=0)

    integrated_grads = ((x.squeeze(0) - baseline.squeeze(0)) * avg_grads)
    return integrated_grads.detach().cpu().numpy()


def peak_segment(ig_map: np.ndarray) -> str:
    time_relevance = ig_map.mean(axis=0)
    positive       = np.maximum(time_relevance, 0)

    scores = {}
    for name, (start, end) in SEGMENTS.items():
        region = positive[start:end]
        scores[name] = float(region.mean()) if len(region) > 0 else 0.0

    return max(scores, key=scores.get)


def peak_channel(ig_map: np.ndarray) -> int:
    channel_relevance = np.maximum(ig_map, 0).sum(axis=1)
    return int(np.argmax(channel_relevance))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH}")
    print()

    print("Loading model...")
    model = load_model(MODEL_PATH, DEVICE)
    print("Model loaded.\n")

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
        "true_label",
        "pred_label",
        "prob_seizure",
        "ig_peak_segment",
        "ig_peak_channel",
        "correct",
    ])

    seg_counts = {
        "seizure":    defaultdict(int),
        "background": defaultdict(int),
    }
    total   = 0
    correct = 0

    for entry in entries:
        pt_filename = Path(entry["pt_path"]).name
        pt_path     = os.path.join(DATA_DIR, pt_filename)

        if not os.path.exists(pt_path):
            print(f"  [MISSING] {pt_filename}")
            continue

        data = safe_load_pt(pt_path)
        if data is None:
            continue

        x_all = data["x"]
        y_all = data["y"]
        N     = len(y_all)

        print(f"  {pt_filename} | {N} windows")

        for w_idx in range(N):
            if MAX_WINDOWS is not None and total >= MAX_WINDOWS:
                break

            raw_label  = int(y_all[w_idx].item())
            true_label = 1 if raw_label > 0 else 0

            x = x_all[w_idx].unsqueeze(0).float()

            with torch.no_grad():
                logit = model(x.to(DEVICE))
                prob  = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0

            ig_map = integrated_gradients(model, x, baseline, n_steps=IG_STEPS)

            seg  = peak_segment(ig_map)
            chan = peak_channel(ig_map)

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
    seg_names = list(SEGMENTS.keys())
    sei_total = max(sum(seg_counts["seizure"].values()), 1)
    bkg_total = max(sum(seg_counts["background"].values()), 1)

    sei_pct = [seg_counts["seizure"].get(s, 0) / sei_total * 100 for s in seg_names]
    bkg_pct = [seg_counts["background"].get(s, 0) / bkg_total * 100 for s in seg_names]

    x = np.arange(len(seg_names))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"MindScope Stage 1 - Integrated Gradients\n"
        f"{total} windows | Accuracy {100*correct/max(total,1):.1f}%",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    bars = ax.bar(x, sei_pct, color="#e05252", width=w * 2)
    ax.set_title("IG Attribution - Seizure Windows")
    ax.set_xticks(x); ax.set_xticklabels(seg_names, fontsize=8)
    ax.set_ylabel("% of windows (peak segment)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, sei_pct):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)

    ax = axes[1]
    bars = ax.bar(x, bkg_pct, color="#5285e0", width=w * 2)
    ax.set_title("IG Attribution - Background Windows")
    ax.set_xticks(x); ax.set_xticklabels(seg_names, fontsize=8)
    ax.set_ylabel("% of windows (peak segment)")
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, bkg_pct):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)

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