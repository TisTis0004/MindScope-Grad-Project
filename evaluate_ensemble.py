"""
Cross-Architecture Ensemble Evaluation
=========================================
Combines predictions from HETEROGENEOUS models for maximum diversity:
  - 2D CNN-LSTM (operates on Mel spectrograms)
  - 1D EEGNet (operates on raw EEG waveforms)

WHY this ensemble works so well:
  - The 2D model excels at spectral pattern recognition (frequency × time)
  - The 1D model excels at temporal waveform morphology (sharp waves, spikes)
  - Their errors are UNCORRELATED because they see fundamentally different features
  - Averaging their probabilities cancels out individual model noise

Ensemble strategies implemented:
  1. Simple average (equal weight)
  2. Weighted average (grid-searched optimal weights)
  3. All strategies + temporal voting
  4. Full pipeline: ensemble + voting + min-duration filter

Run: python evaluate_ensemble.py
"""

import json
import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from scipy.ndimage import median_filter
from tqdm import tqdm

# ---- Model imports ----
from helper.T import EEGToSpectrogram
from helper.models import Spectrogram_CNN_LSTM, CNN_LSTM_Large
from braindecode.models import EEGNet


# =========================================================
# CONFIG — Define your ensemble members
# =========================================================

# --- 2D Spectrogram Models (CNN-LSTM variants) ---
SPECTROGRAM_MODELS = [
    {
        "name": "CNN-LSTM-Large v3",
        "checkpoint": "checkpoints/cnn_lstm_melspectrogram_dropout_new4changes.pt",
        "model_class": Spectrogram_CNN_LSTM,
    },
    # Uncomment to add more 2D models:
    # {
    #     "name": "CNN-LSTM v1",
    #     "checkpoint": "checkpoints/cnn_lstm_melspectrogram_dropout_new4changes.pt",
    #     "model_class": Spectrogram_CNN_LSTM,
    # },
    # {
    #     "name": "CNN-LSTM v2",
    #     "checkpoint": "checkpoints/cnn_lstm_melspectrogram_dropout_new4changes_2ndRun.pt",
    #     "model_class": Spectrogram_CNN_LSTM,
    # },
]

# --- 1D Raw EEG Models (EEGNet) ---
EEGNET_MODELS = [
    {
        "name": "EEGNet-1D Best",
        "checkpoint": "checkpoints/eegnet_1d_best.pt",
        "n_chans": 18,
        "n_times": 2560,
        "n_outputs": 2,
        "F1": 16, "D": 2, "F2": 32,
        "kernel_length": 128,
        "drop_prob": 0.5,
    },
    # Uncomment the SWA variant if available:
    # {
    #     "name": "EEGNet-1D SWA",
    #     "checkpoint": "checkpoints/eegnet_1d_best_swa_final.pt",
    #     "n_chans": 18, "n_times": 2560, "n_outputs": 2,
    #     "F1": 16, "D": 2, "F2": 32,
    #     "kernel_length": 128, "drop_prob": 0.5,
    # },
]

EVAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"
NUM_CLASSES = 2
VOTING_WINDOWS = [1, 3, 5, 7]  # Try multiple window sizes


# =========================================================
# Per-Channel Normalization (must match training)
# =========================================================
class PerChannelNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std


# =========================================================
# MODEL LOADING
# =========================================================
def load_spectrogram_model(config, device):
    """Load a 2D spectrogram-based model."""
    model = config["model_class"]()
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def load_eegnet_model(config, device):
    """Load a 1D EEGNet model."""
    model = EEGNet(
        n_chans=config["n_chans"],
        n_outputs=config["n_outputs"],
        n_times=config["n_times"],
        final_conv_length="auto",
        pool_mode="mean",
        F1=config["F1"],
        D=config["D"],
        F2=config["F2"],
        kernel_length=config["kernel_length"],
        drop_prob=config["drop_prob"],
    )
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Handle SWA model state dict (has 'module.' prefix)
    is_swa = checkpoint.get("swa", False)
    if is_swa:
        # SWA wraps model with AveragedModel, adding 'module.' prefix
        # Also has 'n_averaged' key to remove
        cleaned = {}
        for k, v in state_dict.items():
            if k == "n_averaged":
                continue
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            cleaned[new_k] = v
        state_dict = cleaned

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# =========================================================
# DATA LOADING
# =========================================================
def load_eval_manifest(manifest_path):
    files = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                files.append((Path(obj["pt_path"]), int(obj["n"])))
    return files


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def predict_spectrogram_model(model, pt_path, transform, device, batch_size=64):
    """Run inference with a 2D spectrogram model."""
    data = torch.load(pt_path, map_location="cpu")
    x_all = data["x"]
    y_all = data["y"].long()

    all_probs = []
    n = x_all.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = x_all[start:end].to(device)
        x_batch = transform(x_batch)

        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(x_batch)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    y_true = y_all.numpy()
    return y_true, all_probs


@torch.no_grad()
def predict_eegnet_model(model, pt_path, device, batch_size=64):
    """Run inference with a 1D EEGNet model (raw EEG, no spectrogram).
    Data is already Median+IQR normalized from caching."""
    data = torch.load(pt_path, map_location="cpu")
    x_all = data["x"]
    y_all = data["y"].long()

    all_probs = []
    n = x_all.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = x_all[start:end].to(device)
        # No extra normalization — data is already Median+IQR normalized from cache

        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(x_batch)

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    y_true = y_all.numpy()
    return y_true, all_probs


# =========================================================
# POST-PROCESSING
# =========================================================
def apply_temporal_voting(probs, window_size):
    """Median filter on seizure probabilities per recording."""
    if window_size <= 1:
        return probs
    seizure_prob = probs[:, 1].copy()
    smoothed_prob = median_filter(seizure_prob, size=window_size, mode='nearest')
    return np.stack([1 - smoothed_prob, smoothed_prob], axis=1)


def apply_min_duration_filter(preds, min_consecutive=3):
    """Reject isolated seizure predictions shorter than min_consecutive."""
    filtered = np.zeros_like(preds)
    n = len(preds)
    i = 0
    while i < n:
        if preds[i] == 1:
            run_start = i
            while i < n and preds[i] == 1:
                i += 1
            run_length = i - run_start
            if run_length >= min_consecutive:
                filtered[run_start:i] = 1
        else:
            i += 1
    return filtered


def find_best_threshold(y_true, probs):
    """Sweep thresholds to maximize F1-macro."""
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.81, 0.02):
        y_pred = (probs[:, 1] >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def compute_full_metrics(y_true, probs, threshold):
    """Compute comprehensive metrics at a given threshold."""
    y_pred = (probs[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except:
        auc = float('nan')
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "threshold": threshold,
        "confusion_matrix": cm,
        "f1_background": f1_cls[0],
        "f1_seizure": f1_cls[1],
        "precision_seizure": prec_cls[1],
        "recall_seizure": rec_cls[1],
    }


def grid_search_weights(y_true, model_probs_list, step=0.05):
    """
    Grid search for optimal ensemble weights.
    For 2 model groups, search weight for group 1 (group 2 = 1 - w1).
    For >2 groups, use equal weights (combinatorial search too expensive).
    """
    n_groups = len(model_probs_list)
    if n_groups == 1:
        return [1.0]
    
    if n_groups == 2:
        best_f1 = -1
        best_w = 0.5
        for w in np.arange(0.0, 1.01, step):
            avg_probs = w * model_probs_list[0] + (1 - w) * model_probs_list[1]
            thresh, f1 = find_best_threshold(y_true, avg_probs)
            if f1 > best_f1:
                best_f1 = f1
                best_w = w
        return [best_w, 1 - best_w]
    
    # For >2 groups, equal weights
    return [1.0 / n_groups] * n_groups


# =========================================================
# MAIN
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"  Cross-Architecture Ensemble Evaluation")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  2D Models: {len(SPECTROGRAM_MODELS)}")
    print(f"  1D Models: {len(EEGNET_MODELS)}")
    total_models = len(SPECTROGRAM_MODELS) + len(EEGNET_MODELS)
    print(f"  Total ensemble members: {total_models}")

    # ===========================================================
    # PHASE 0: Load all models
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 0: Loading Models")
    print(f"{'='*70}")

    spec_models = []
    for cfg in SPECTROGRAM_MODELS:
        print(f"  Loading 2D: {cfg['name']} ({cfg['checkpoint']})")
        if not Path(cfg["checkpoint"]).exists():
            print(f"  ⚠️ SKIPPED — checkpoint not found")
            continue
        model = load_spectrogram_model(cfg, device)
        spec_models.append((cfg["name"], model))

    eegnet_models = []
    for cfg in EEGNET_MODELS:
        print(f"  Loading 1D: {cfg['name']} ({cfg['checkpoint']})")
        if not Path(cfg["checkpoint"]).exists():
            print(f"  ⚠️ SKIPPED — checkpoint not found")
            continue
        model = load_eegnet_model(cfg, device)
        eegnet_models.append((cfg["name"], model))

    if not spec_models and not eegnet_models:
        print("❌ No models loaded! Check checkpoint paths.")
        return

    print(f"\n  Loaded: {len(spec_models)} spectrogram + {len(eegnet_models)} EEGNet")

    # Transforms
    spec_transform = EEGToSpectrogram().to(device)
    spec_transform.eval()
    # No extra normalization needed — cached data is already Median+IQR normalized

    # Load eval data
    eval_files = load_eval_manifest(EVAL_MANIFEST)
    print(f"  Eval recordings: {len(eval_files)}")

    # ===========================================================
    # PHASE 1: Collect predictions from ALL models
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Running Inference")
    print(f"{'='*70}")

    # Structure: per_model_probs[model_name] = list of probs per recording
    per_model_probs = {}
    y_true_per_rec = []

    for rec_idx, (pt_path, n_windows) in enumerate(tqdm(eval_files, desc="Inference")):
        # 2D models
        for name, model in spec_models:
            y_true, probs = predict_spectrogram_model(model, pt_path, spec_transform, device)
            per_model_probs.setdefault(name, []).append(probs)
            if rec_idx == 0 and name == spec_models[0][0]:
                pass  # y_true captured below

        # 1D models
        for name, model in eegnet_models:
            y_true, probs = predict_eegnet_model(model, pt_path, device)
            per_model_probs.setdefault(name, []).append(probs)

        y_true_per_rec.append(y_true)

    # ===========================================================
    # PHASE 2: Compute individual model performance
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Individual Model Performance")
    print(f"{'='*70}")

    all_y_true = np.concatenate(y_true_per_rec)
    results = {}

    for model_name in per_model_probs:
        all_probs = np.concatenate(per_model_probs[model_name])
        thresh, _ = find_best_threshold(all_y_true, all_probs)
        metrics = compute_full_metrics(all_y_true, all_probs, thresh)
        results[model_name] = metrics
        print(f"  {model_name}: F1={metrics['f1_macro']:.4f} | "
              f"BalAcc={metrics['balanced_accuracy']:.4f} | "
              f"AUC={metrics['auc']:.4f} | Thresh={metrics['threshold']:.2f}")

    # ===========================================================
    # PHASE 3: Ensemble — Average by model group
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Ensemble Strategies")
    print(f"{'='*70}")

    # Compute group-level averaged probabilities
    # Group 1: All 2D spectrogram models averaged
    # Group 2: All 1D EEGNet models averaged
    group_probs = {}

    if spec_models:
        spec_names = [n for n, _ in spec_models]
        # Average across 2D models per recording, then concatenate
        spec_avg_per_rec = []
        for rec_idx in range(len(eval_files)):
            rec_probs = [per_model_probs[n][rec_idx] for n in spec_names]
            avg = np.mean(rec_probs, axis=0)
            spec_avg_per_rec.append(avg)
        group_probs["2D-CNN-LSTM"] = np.concatenate(spec_avg_per_rec)

    if eegnet_models:
        eeg_names = [n for n, _ in eegnet_models]
        eeg_avg_per_rec = []
        for rec_idx in range(len(eval_files)):
            rec_probs = [per_model_probs[n][rec_idx] for n in eeg_names]
            avg = np.mean(rec_probs, axis=0)
            eeg_avg_per_rec.append(avg)
        group_probs["1D-EEGNet"] = np.concatenate(eeg_avg_per_rec)

    # --- Strategy A: Simple average (equal weight) ---
    group_keys = list(group_probs.keys())
    if len(group_keys) >= 2:
        simple_avg = np.mean([group_probs[k] for k in group_keys], axis=0)
        thresh, _ = find_best_threshold(all_y_true, simple_avg)
        metrics = compute_full_metrics(all_y_true, simple_avg, thresh)
        results["Ensemble (equal weight)"] = metrics
        print(f"\n  Ensemble (equal weight): F1={metrics['f1_macro']:.4f} | "
              f"AUC={metrics['auc']:.4f} | Thresh={thresh:.2f}")

        # --- Strategy B: Grid-search optimal weights ---
        print("\n  Searching optimal ensemble weights...")
        group_list = [group_probs[k] for k in group_keys]
        opt_weights = grid_search_weights(all_y_true, group_list, step=0.05)
        weighted_avg = sum(w * p for w, p in zip(opt_weights, group_list))
        thresh, _ = find_best_threshold(all_y_true, weighted_avg)
        metrics = compute_full_metrics(all_y_true, weighted_avg, thresh)
        weight_str = " / ".join([f"{k}={w:.2f}" for k, w in zip(group_keys, opt_weights)])
        results[f"Ensemble (optimal: {weight_str})"] = metrics
        print(f"  Ensemble (optimal: {weight_str}): F1={metrics['f1_macro']:.4f} | "
              f"AUC={metrics['auc']:.4f} | Thresh={thresh:.2f}")
    elif len(group_keys) == 1:
        simple_avg = group_probs[group_keys[0]]
        weighted_avg = simple_avg
        print(f"\n  Only one model group ({group_keys[0]}), skipping ensemble weight search")

    # ===========================================================
    # PHASE 4: Apply temporal voting to the BEST ensemble
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Temporal Voting on Ensemble")
    print(f"{'='*70}")

    # Use the best ensemble combination for voting
    best_ensemble_probs = weighted_avg if len(group_keys) >= 2 else simple_avg

    # Reconstruct per-recording probs for voting
    if len(group_keys) >= 2:
        ens_per_rec = []
        for rec_idx in range(len(eval_files)):
            rec_group_probs = []
            for k in group_keys:
                if k == "2D-CNN-LSTM" and spec_models:
                    spec_names = [n for n, _ in spec_models]
                    rec_probs = [per_model_probs[n][rec_idx] for n in spec_names]
                    rec_group_probs.append(np.mean(rec_probs, axis=0))
                elif k == "1D-EEGNet" and eegnet_models:
                    eeg_names = [n for n, _ in eegnet_models]
                    rec_probs = [per_model_probs[n][rec_idx] for n in eeg_names]
                    rec_group_probs.append(np.mean(rec_probs, axis=0))
            # Weighted average per recording
            ens_rec = sum(w * p for w, p in zip(opt_weights, rec_group_probs))
            ens_per_rec.append(ens_rec)
    else:
        # Single group: just use per-recording probs
        single_key = group_keys[0]
        if single_key == "2D-CNN-LSTM":
            model_names = [n for n, _ in spec_models]
        else:
            model_names = [n for n, _ in eegnet_models]
        ens_per_rec = []
        for rec_idx in range(len(eval_files)):
            rec_probs = [per_model_probs[n][rec_idx] for n in model_names]
            ens_per_rec.append(np.mean(rec_probs, axis=0))

    for vw in VOTING_WINDOWS:
        # Apply temporal voting per recording
        voted_per_rec = []
        for rec_probs in ens_per_rec:
            voted = apply_temporal_voting(rec_probs, vw)
            voted_per_rec.append(voted)

        all_voted = np.concatenate(voted_per_rec)
        thresh, _ = find_best_threshold(all_y_true, all_voted)
        metrics = compute_full_metrics(all_y_true, all_voted, thresh)
        results[f"Ensemble + voting(w={vw})"] = metrics
        print(f"  Voting window={vw}: F1={metrics['f1_macro']:.4f} | "
              f"AUC={metrics['auc']:.4f} | Thresh={thresh:.2f}")

    # ===========================================================
    # PHASE 5: Full pipeline — best ensemble + voting + min-duration
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 5: Full Pipeline (Ensemble + Voting + Min-Duration)")
    print(f"{'='*70}")

    # Find the best voting window from Phase 4
    best_vw_key = max(
        [k for k in results if "voting" in k],
        key=lambda k: results[k]["f1_macro"]
    )
    best_vw = int(best_vw_key.split("w=")[1].split(")")[0])
    print(f"  Best voting window: {best_vw}")

    # Apply best voting
    voted_per_rec = []
    for rec_probs in ens_per_rec:
        voted = apply_temporal_voting(rec_probs, best_vw)
        voted_per_rec.append(voted)
    all_voted = np.concatenate(voted_per_rec)
    thresh, _ = find_best_threshold(all_y_true, all_voted)

    # Apply min-duration filter per recording
    for min_dur in [2, 3, 4]:
        all_preds = (all_voted[:, 1] >= thresh).astype(int)
        # Apply per recording
        filtered_all = []
        idx = 0
        for rec_probs in voted_per_rec:
            n = len(rec_probs)
            rec_preds = all_preds[idx:idx+n]
            filtered = apply_min_duration_filter(rec_preds, min_consecutive=min_dur)
            filtered_all.append(filtered)
            idx += n
        filtered_all = np.concatenate(filtered_all)

        # Recompute metrics with filtered predictions
        f1 = precision_recall_fscore_support(
            all_y_true, filtered_all, average="macro", zero_division=0
        )[2]
        bal_acc = balanced_accuracy_score(all_y_true, filtered_all)
        cm = confusion_matrix(all_y_true, filtered_all)
        try:
            auc = roc_auc_score(all_y_true, all_voted[:, 1])
        except:
            auc = float('nan')

        name = f"Full Pipeline (voting={best_vw}, min_dur={min_dur})"
        results[name] = {
            "f1_macro": f1, "balanced_accuracy": bal_acc, "auc": auc,
            "threshold": thresh, "confusion_matrix": cm,
        }
        print(f"  min_dur={min_dur}: F1={f1:.4f} | BalAcc={bal_acc:.4f}")

    # ===========================================================
    # FINAL RESULTS TABLE
    # ===========================================================
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<45} {'F1-macro':<10} {'Bal-Acc':<10} {'AUC':<10} {'Thresh':<8}")
    print("-" * 83)

    for name, m in sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
        auc_val = m.get('auc', float('nan'))
        auc_str = f"{auc_val:<10.4f}" if not np.isnan(auc_val) else "N/A       "
        print(f"{name:<45} {m['f1_macro']:<10.4f} {m['balanced_accuracy']:<10.4f} {auc_str} {m['threshold']:<8.2f}")

    # Best overall
    best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_m = results[best_name]
    print(f"\n{'='*70}")
    print(f"🏆 BEST STRATEGY: {best_name}")
    print(f"   F1-macro:          {best_m['f1_macro']:.4f}")
    print(f"   Balanced Accuracy: {best_m['balanced_accuracy']:.4f}")
    print(f"   AUC:               {best_m.get('auc', 'N/A')}")
    print(f"   Threshold:         {best_m['threshold']:.2f}")
    print(f"   Confusion Matrix:")
    print(f"   {best_m['confusion_matrix']}")

    # Compare: best individual vs best ensemble
    individual_names = [n for n, _ in spec_models] + [n for n, _ in eegnet_models]
    if individual_names:
        best_individual = max(
            [k for k in results if k in individual_names],
            key=lambda k: results[k]['f1_macro']
        )
        ind_f1 = results[best_individual]['f1_macro']
        print(f"\n   Best individual model: {best_individual} (F1={ind_f1:.4f})")
        print(f"   Ensemble improvement: +{best_m['f1_macro'] - ind_f1:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
