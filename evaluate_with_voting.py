"""
Unified Evaluation and Voting Script
======================================
Evaluates any single model OR an ensemble of mixed models (CNN-LSTM + EEGNet) with custom weights.
Applies temporal smoothing (voting) across consecutive windows.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from scipy.ndimage import median_filter
from tqdm import tqdm

from helper.T import EEGToSpectrogram
from helper.models import CNN_LSTM_Large, Spectrogram_CNN_LSTM
from braindecode.models import EEGNet

# =========================================================
# CONFIG — Define your models and their weights here
# =========================================================

MODELS = [
    # --- EEGNet 1D (Last Run) ---
    {
        "name": "EEGNet-1D",
        "type": "eegnet",  # 'eegnet' or 'cnn_lstm'
        "checkpoint": "checkpoints/eegnet_1d_best.pt",
        "weight": 1.0      # Weight for probability averaging
    },
    
    # --- Example: To add your CNN-LSTM, uncomment this ---
    # {
    #     "name": "CNN-LSTM",
    #     "type": "cnn_lstm",
    #     "checkpoint": "checkpoints/cnn_lstm_large_v3.pt",
    #     "weight": 1.0  
    # }
]


EVAL_MANIFEST = "cache_windows_binary_10_sec_eval/manifest.jsonl"
NUM_CLASSES = 2
VOTING_WINDOW_SIZES = [1, 3, 5, 7, 9]  # Test multiple voting windows

# =========================================================
# MODEL LOADING
# =========================================================

def build_eegnet(device):
    """Build EEGNet matched to our training script."""
    model = EEGNet(
        n_chans=18,
        n_outputs=NUM_CLASSES,
        n_times=2560,
        final_conv_length="auto",
        pool_mode="mean",
        F1=16,
        D=2,
        F2=32,
        kernel_length=128,
        drop_prob=0.5,
    )
    return model.to(device)


def load_model(config, device):
    """Load model from checkpoint based on its type."""
    if config["type"] == "eegnet":
        model = build_eegnet(device)
    else:  # cnn_lstm variants
        model = CNN_LSTM_Large() if "large" in config["checkpoint"].lower() else Spectrogram_CNN_LSTM()
        model = model.to(device)

    # Load weights
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Clean SWA prefixes if present
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k == "n_averaged": continue
        cleaned_dict[k.replace("module.", "")] = v
        
    model.load_state_dict(cleaned_dict)
    model.eval()
    return model


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
def get_model_probabilities(models_config, pt_path, transform, device, batch_size=64):
    """
    Run inference on ALL windows from a single recording using ALL models,
    then combines their probabilities according to their 'weight'.
    """
    data = torch.load(pt_path, map_location="cpu")
    x_raw = data["x"]  # [N, C, T]
    y_all = data["y"].long().numpy()  # [N]
    
    n = x_raw.shape[0]
    final_probs = np.zeros((n, NUM_CLASSES))
    weight_sum = sum(cfg.get("weight", 1.0) for cfg in models_config)
    
    for cfg in models_config:
        model = cfg["model_instance"]
        m_type = cfg["type"]
        weight = cfg.get("weight", 1.0)
        
        all_m_probs = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_batch = x_raw[start:end].to(device)
            
            # Handle input formatting based on model architecture
            if m_type == "cnn_lstm":
                x_batch = transform(x_batch)
            elif m_type == "eegnet":
                # Ensure safety clamp for EEGNet as in training
                x_batch = torch.clamp(x_batch, min=-20.0, max=20.0)
            
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(x_batch)
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_m_probs.append(probs)
            
        all_m_probs = np.concatenate(all_m_probs, axis=0) # [N, 2]
        final_probs += all_m_probs * (weight / weight_sum)
        
    preds = np.argmax(final_probs, axis=1)
    return y_all, preds, final_probs


def apply_temporal_voting(probs, window_size):
    """Median filter the seizure probability, then re-threshold."""
    if window_size <= 1:
        return probs
    
    seizure_prob = probs[:, 1].copy()
    smoothed_prob = median_filter(seizure_prob, size=window_size, mode='nearest')
    return np.stack([1 - smoothed_prob, smoothed_prob], axis=1)


def find_best_threshold(y_true, probs):
    """Sweep thresholds to find the one that maximizes F1-macro."""
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.20, 0.81, 0.02):
        y_pred_t = (probs[:, 1] >= thresh).astype(int)
        _, _, f1_t, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average="macro", zero_division=0
        )
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_with_voting(models_config, eval_files, transform, device, voting_window=1):
    all_y_true = []
    all_probs = []
    
    for pt_path, n_windows in tqdm(eval_files, desc=f"Eval (vote={voting_window})", leave=False):
        y_true, _, raw_probs = get_model_probabilities(
            models_config, pt_path, transform, device
        )
        
        smoothed_probs = apply_temporal_voting(raw_probs, voting_window)
        
        all_y_true.append(y_true)
        all_probs.append(smoothed_probs)
    
    all_y_true = np.concatenate(all_y_true)
    all_probs = np.concatenate(all_probs)
    
    best_thresh, _ = find_best_threshold(all_y_true, all_probs)
    y_pred = (all_probs[:, 1] >= best_thresh).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y_true, y_pred, average="macro", zero_division=0
    )
    bal_acc = balanced_accuracy_score(all_y_true, y_pred)
    cm = confusion_matrix(all_y_true, y_pred)
    
    try:
        auc = roc_auc_score(all_y_true, all_probs[:, 1])
    except:
        auc = float('nan')
    
    return {
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "threshold": best_thresh,
        "confusion_matrix": cm,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}\nUnified Ensemble & Voting Evaluator\n{'='*70}")
    print(f"Device: {device}")
    
    # Load all models into the config dictionaries
    print("\nLoading Models...")
    for cfg in MODELS:
        print(f"  → [{cfg['type']}] {cfg['name']} (weight: {cfg['weight']})")
        cfg["model_instance"] = load_model(cfg, device)
        
    # Setup transform for Spectrogram models (only used if type == 'cnn_lstm')
    transform = EEGToSpectrogram().to(device)
    transform.eval()
    
    eval_files = load_eval_manifest(EVAL_MANIFEST)
    print(f"\nEvaluating {len(eval_files)} recordings...\n")
    
    results = {}
    for window_size in VOTING_WINDOW_SIZES:
        metrics = evaluate_with_voting(MODELS, eval_files, transform, device, window_size)
        results[window_size] = metrics
        
        label = "No voting" if window_size == 1 else f"Voting window={window_size}"
        print(f"--- {label} ---")
        print(f"  F1-macro:     {metrics['f1_macro']:.4f}  |  AUC: {metrics['auc']:.4f}")
        print(f"  Threshold:    {metrics['threshold']:.2f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Window':<12} {'F1-macro':<12} {'Bal-Acc':<12} {'AUC':<12} {'Thresh':<12}")
    print("-" * 60)
    for ws, m in results.items():
        label = "None" if ws == 1 else str(ws)
        print(f"{label:<12} {m['f1_macro']:<12.4f} {m['balanced_accuracy']:<12.4f} {m['auc']:<12.4f} {m['threshold']:<12.2f}")
    
    best_ws = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_f1 = results[best_ws]['f1_macro']
    baseline_f1 = results[1]['f1_macro']
    
    print(f"\n🏆 Best: voting_window={best_ws} → F1={best_f1:.4f} (+{best_f1 - baseline_f1:.4f} over baseline)")
    print("\nBest Confusion Matrix:")
    print(results[best_ws]["confusion_matrix"])


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
